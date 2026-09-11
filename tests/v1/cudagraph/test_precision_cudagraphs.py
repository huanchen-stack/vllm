# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Precision-keyed CUDA-graph keys and dispatch (component C3), CPU only.

Oracles (all under ``/data/huanchen/verl/.codex-report/new-storyline-experiments/``):

* ``dynamic_tail8k_heatmap_20260823/runs/{b32,b64,b128}_*/logs/*.log``:
  ``Profiling CUDA graph memory: PIECEWISE=29 (largest=64), FULL=21 (largest=32)``
  (b32), ``45 (128) / 29 (64)`` (b64), ``77 (256) / 45 (128)`` (b128), every
  policy with ``capture_max_batch=32``, LoRA cases ``[0, 2]``.
* ``hardmath_lora_accuracy_100step_20260825/.../cap9k.log``: eleven
  ``No matching dynamic-precision CUDA graph`` warnings, one per
  ``(num_tokens, num_reqs, int4)`` key above the ceiling.
* ``eos_hazard_extensibility/logs/supplement_preflight/
  smollm3_3b_gsm8k_tail_w4_t8_cg.log``:
  the fixed-threshold path captured ``FULL=38`` (one precision per size) and
  fell to eager for the scheduler's ``bf16`` at ``num_tokens=1``; the clean
  branch registers both precisions (``45``) and never falls back there.
* vanilla ``tests/v1/cudagraph/test_cudagraph_dispatch.py`` (10 keys with
  LoRA specialisation, 2 without) with every dual-precision flag off.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from unittest.mock import MagicMock

import pytest

import vllm.envs as envs
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    CUDAGraphMode,
    ParallelConfig,
    SchedulerConfig,
    VllmConfig,
)
from vllm.config.lora import LoRAConfig
from vllm.forward_context import BatchDescriptor
from vllm.model_executor.dual_precision import (
    BASE_PRECISION_BF16,
    BASE_PRECISION_INT4,
    check_dual_precision_model_runner,
)
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher

_MISSING = object()
_ENV_NAMES = ("VLLM_DUAL_PRECISION_ROLLOUT", "VLLM_DUAL_PRECISION_POLICY")


@pytest.fixture
def dual_precision_env():
    """Set the two dispatcher knobs on the lazy ``vllm.envs`` module.

    ``monkeypatch.setattr`` would restore a *concrete* module attribute on
    undo (the lazy ``__getattr__`` value it read), which then shadows later
    ``os.environ`` changes in other tests; this fixture removes the
    attributes again so lazy lookup resumes.
    """
    saved = {name: envs.__dict__.get(name, _MISSING) for name in _ENV_NAMES}

    def configure(*, enabled: bool = True, policy: str = "") -> None:
        envs.VLLM_DUAL_PRECISION_ROLLOUT = enabled
        envs.VLLM_DUAL_PRECISION_POLICY = policy

    yield configure
    for name, value in saved.items():
        if value is _MISSING:
            envs.__dict__.pop(name, None)
        else:
            setattr(envs, name, value)


def default_capture_ladder(max_num_seqs: int) -> list[int]:
    """The vanilla default ladder (``VllmConfig._set_cudagraph_sizes``)."""
    max_graph_size = min(max_num_seqs * 2, 512)
    sizes = (
        [1, 2, 4] + list(range(8, 256, 8)) + list(range(256, max_graph_size + 1, 16))
    )
    return [s for s in sizes if s <= max_graph_size]


def make_config(
    cudagraph_mode: str,
    compilation_mode: CompilationMode,
    capture_sizes: list[int],
    max_num_seqs: int,
    lora: LoRAConfig | None,
) -> MagicMock:
    comp = CompilationConfig(
        cudagraph_mode=cudagraph_mode,
        mode=compilation_mode,
        cudagraph_capture_sizes=list(capture_sizes),
    )
    config = MagicMock(spec=VllmConfig)
    config.compilation_config = comp
    config.scheduler_config = SchedulerConfig.default_factory(max_num_seqs=max_num_seqs)
    config.parallel_config = ParallelConfig()
    config.speculative_config = None
    config.lora_config = lora
    if comp.mode == CompilationMode.VLLM_COMPILE:
        comp.set_splitting_ops_for_v1(
            all2all_backend=config.parallel_config.all2all_backend,
            data_parallel_size=config.parallel_config.data_parallel_size,
        )
    comp.max_cudagraph_capture_size = comp.cudagraph_capture_sizes[-1]
    comp.post_init_cudagraph_sizes()
    return config


def headline_lora() -> LoRAConfig:
    """LoRA cases ``[0, 2]``: ``max_loras=1`` without active-count specialisation."""
    return LoRAConfig(max_loras=1, specialize_active_lora=False)


def build(config: MagicMock) -> CudagraphDispatcher:
    dispatcher = CudagraphDispatcher(config)
    dispatcher.initialize_cudagraph_keys(
        cudagraph_mode=config.compilation_config.cudagraph_mode,
        uniform_decode_query_len=1,
    )
    return dispatcher


def keys(dispatcher: CudagraphDispatcher, mode: CUDAGraphMode) -> set[BatchDescriptor]:
    return dispatcher.cudagraph_keys[mode]


def int4_keys(dispatcher: CudagraphDispatcher, mode: CUDAGraphMode) -> set[int]:
    """``num_tokens`` of every INT4 key of ``mode``."""
    return {
        k.num_tokens
        for k in keys(dispatcher, mode)
        if k.base_precision == BASE_PRECISION_INT4
    }


def int4_descs(
    dispatcher: CudagraphDispatcher, mode: CUDAGraphMode
) -> set[BatchDescriptor]:
    return {
        k for k in keys(dispatcher, mode) if k.base_precision == BASE_PRECISION_INT4
    }


# ---------------------------------------------------------------------------
# Key sets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "max_num_seqs, piecewise, full, largest_piecewise, largest_full",
    [
        (32, 29, 21, 64, 32),
        (64, 45, 29, 128, 64),
        (128, 77, 45, 256, 128),
    ],
)
def test_key_sets_match_the_headline_ladders(
    dual_precision_env, max_num_seqs, piecewise, full, largest_piecewise, largest_full
):
    dual_precision_env(enabled=True, policy="fixed_frontier:8000")  # ceiling 32
    ladder = default_capture_ladder(max_num_seqs)
    assert ladder[-1] == largest_piecewise
    config = make_config(
        "FULL_AND_PIECEWISE",
        CompilationMode.VLLM_COMPILE,
        ladder,
        max_num_seqs,
        headline_lora(),
    )
    dispatcher = build(config)

    assert len(keys(dispatcher, CUDAGraphMode.PIECEWISE)) == piecewise
    assert len(keys(dispatcher, CUDAGraphMode.FULL)) == full
    assert dispatcher.int4_capture_max_batch == 32

    for mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
        twins = int4_descs(dispatcher, mode)
        assert len(twins) == 7  # sizes 1, 2, 4, 8, 16, 24, 32
        assert all(k.has_lora and k.num_tokens <= 32 for k in twins)
        # every INT4 key has a BF16 twin with the same shape
        for k in twins:
            assert replace(k, base_precision=BASE_PRECISION_BF16) in keys(
                dispatcher, mode
            )

    descs = dict(dispatcher.get_capture_descs())
    assert set(descs[CUDAGraphMode.PIECEWISE]) == keys(
        dispatcher, CUDAGraphMode.PIECEWISE
    )
    assert set(descs[CUDAGraphMode.FULL]) == keys(dispatcher, CUDAGraphMode.FULL)
    assert descs[CUDAGraphMode.PIECEWISE][0].num_tokens == largest_piecewise
    assert descs[CUDAGraphMode.FULL][0].num_tokens == largest_full


def test_fixed_threshold_mode_registers_both_precisions(dual_precision_env, caplog):
    """SmolLM3 preflight: the old static path captured FULL=38 (one precision
    per size) and the scheduler's ``bf16`` at ``num_tokens=1`` ran eager."""
    dual_precision_env(enabled=True, policy="fixed_threshold:8")  # ceiling 32
    ladder = default_capture_ladder(64)  # 19 sizes up to 128
    config = make_config(
        "FULL_DECODE_ONLY", CompilationMode.NONE, ladder, 128, headline_lora()
    )
    dispatcher = build(config)

    assert len(keys(dispatcher, CUDAGraphMode.PIECEWISE)) == 0
    assert len(keys(dispatcher, CUDAGraphMode.FULL)) == 19 * 2 + 7  # 45, was 38

    with caplog.at_level(logging.WARNING, logger="vllm"):
        for precision in (BASE_PRECISION_BF16, BASE_PRECISION_INT4):
            mode, desc = dispatcher.dispatch(
                1,
                num_reqs=1,
                uniform_decode=True,
                has_lora=True,
                num_active_loras=1,
                base_precision=precision,
            )
            assert mode == CUDAGraphMode.FULL
            assert desc.base_precision == precision
            assert desc.num_tokens == 1 and desc.num_reqs == 1 and desc.uniform
    assert "No matching dynamic-precision CUDA graph" not in caplog.text


def test_fixed_threshold_above_32_lifts_the_ceiling(dual_precision_env):
    dual_precision_env(enabled=True, policy="fixed_threshold:48")
    config = make_config(
        "PIECEWISE", CompilationMode.VLLM_COMPILE, [8, 32, 48, 64], 64, headline_lora()
    )
    dispatcher = build(config)
    assert dispatcher.int4_capture_max_batch == 48
    assert int4_keys(dispatcher, CUDAGraphMode.PIECEWISE) == {8, 32, 48}


def test_uniform_w4_ceiling_is_clamped_to_the_capture_list(dual_precision_env):
    dual_precision_env(enabled=True, policy="uniform_w4")  # capture_max_batch = 2**30
    config = make_config(
        "PIECEWISE", CompilationMode.VLLM_COMPILE, [8, 32, 64], 64, headline_lora()
    )
    dispatcher = build(config)
    assert dispatcher.int4_capture_max_batch == 64
    assert int4_keys(dispatcher, CUDAGraphMode.PIECEWISE) == {8, 32, 64}


def test_policy_file_ceiling_is_read_from_the_json(dual_precision_env, tmp_path):
    from vllm.v1.core.sched.precision_policy import load_precision_policy

    policy = load_precision_policy("fixed_frontier:8000")
    raw = policy.to_json()
    raw["capture_max_batch"] = 16
    path = tmp_path / "policy.json"
    import json

    path.write_text(json.dumps(raw))
    dual_precision_env(enabled=True, policy=str(path))
    config = make_config(
        "PIECEWISE", CompilationMode.VLLM_COMPILE, [8, 16, 32, 64], 64, headline_lora()
    )
    dispatcher = build(config)
    assert dispatcher.int4_capture_max_batch == 16
    assert int4_keys(dispatcher, CUDAGraphMode.PIECEWISE) == {8, 16}


def test_int4_keys_need_lora_wrappers(dual_precision_env):
    """The INT4 shadow binds LoRA wrappers only; without LoRA there is
    nothing to bind, so no INT4 key is registered."""
    dual_precision_env(enabled=True, policy="fixed_frontier:8000")
    config = make_config("PIECEWISE", CompilationMode.VLLM_COMPILE, [8, 32], 32, None)
    dispatcher = build(config)
    assert dispatcher.int4_capture_max_batch == 32
    assert int4_keys(dispatcher, CUDAGraphMode.PIECEWISE) == set()
    assert len(keys(dispatcher, CUDAGraphMode.PIECEWISE)) == 2


def test_enabled_without_a_policy_registers_no_int4_keys(dual_precision_env):
    """With no policy the scheduler never publishes ``int4``; capturing INT4
    graphs would only cost memory."""
    dual_precision_env(enabled=True, policy="")
    config = make_config(
        "PIECEWISE", CompilationMode.VLLM_COMPILE, [8, 32], 32, headline_lora()
    )
    dispatcher = build(config)
    assert dispatcher.int4_capture_max_batch is None
    assert int4_keys(dispatcher, CUDAGraphMode.PIECEWISE) == set()
    assert len(keys(dispatcher, CUDAGraphMode.PIECEWISE)) == 4


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


@pytest.fixture
def b64_dispatcher(dual_precision_env) -> CudagraphDispatcher:
    dual_precision_env(enabled=True, policy="fixed_frontier:8000")
    config = make_config(
        "FULL_AND_PIECEWISE",
        CompilationMode.VLLM_COMPILE,
        default_capture_ladder(64),
        64,
        headline_lora(),
    )
    return build(config)


def test_scheduler_override_is_the_only_precision_source(b64_dispatcher):
    dispatcher = b64_dispatcher
    # A tiny decode batch that the old fixed threshold would have made INT4
    # stays BF16 when the scheduler says so ...
    mode, desc = dispatcher.dispatch(
        2,
        num_reqs=2,
        uniform_decode=True,
        has_lora=True,
        num_active_loras=1,
        base_precision=BASE_PRECISION_BF16,
    )
    assert (mode, desc.base_precision) == (CUDAGraphMode.FULL, BASE_PRECISION_BF16)
    # ... and a large one that the threshold would have kept BF16 goes INT4
    # (ceiling 32) when the scheduler says so.
    mode, desc = dispatcher.dispatch(
        32,
        num_reqs=32,
        uniform_decode=True,
        has_lora=True,
        num_active_loras=1,
        base_precision=BASE_PRECISION_INT4,
    )
    assert (mode, desc.base_precision) == (CUDAGraphMode.FULL, BASE_PRECISION_INT4)
    assert desc == BatchDescriptor(
        num_tokens=32,
        num_reqs=32,
        uniform=True,
        has_lora=True,
        num_active_loras=2,
        base_precision=BASE_PRECISION_INT4,
    )
    # Mixed batches dispatch to the relaxed PIECEWISE key of the precision.
    mode, desc = dispatcher.dispatch(
        20,
        num_reqs=5,
        has_lora=True,
        num_active_loras=1,
        base_precision=BASE_PRECISION_INT4,
    )
    assert mode == CUDAGraphMode.PIECEWISE
    assert desc == BatchDescriptor(
        num_tokens=24,
        num_reqs=None,
        uniform=False,
        has_lora=True,
        num_active_loras=2,
        base_precision=BASE_PRECISION_INT4,
    )
    # None means BF16: never anything else, never derived from a count.
    for num_tokens in (1, 2, 8, 64, 128):
        mode, desc = dispatcher.dispatch(
            num_tokens,
            num_reqs=min(num_tokens, 64),
            uniform_decode=True,
            has_lora=True,
            num_active_loras=1,
        )
        assert mode != CUDAGraphMode.NONE
        assert desc.base_precision == BASE_PRECISION_BF16


def test_bad_override_raises(b64_dispatcher):
    with pytest.raises(ValueError, match="fp8"):
        b64_dispatcher.dispatch(8, num_reqs=8, has_lora=True, base_precision="fp8")


def test_dispatch_is_keyword_only_after_num_tokens(b64_dispatcher):
    with pytest.raises(TypeError):
        b64_dispatcher.dispatch(8, 8)  # type: ignore[misc]
    with pytest.raises(TypeError):
        b64_dispatcher.dispatch(8, True)  # type: ignore[misc]


def test_eager_fallback_logs_once_per_key(b64_dispatcher, caplog):
    """Hardmath run: eleven distinct warnings for eleven ``(num_tokens, int4)``
    keys above ``capture_max_batch=32``, each logged once."""
    dispatcher = b64_dispatcher
    warning = "No matching dynamic-precision CUDA graph"

    def count() -> int:
        return sum(warning in r.getMessage() for r in caplog.records)

    with caplog.at_level(logging.WARNING, logger="vllm"):
        for _ in range(3):
            mode, desc = dispatcher.dispatch(
                48,
                num_reqs=48,
                uniform_decode=True,
                has_lora=True,
                num_active_loras=1,
                base_precision=BASE_PRECISION_INT4,
            )
            assert mode == CUDAGraphMode.NONE
            assert desc == BatchDescriptor(
                num_tokens=48, num_reqs=48, base_precision=BASE_PRECISION_INT4
            )
        assert count() == 1
        assert "num_tokens=48, num_reqs=48, base_precision=int4" in caplog.text

        mode, desc = dispatcher.dispatch(
            53,
            num_reqs=53,
            uniform_decode=True,
            has_lora=True,
            num_active_loras=1,
            base_precision=BASE_PRECISION_INT4,
        )
        assert mode == CUDAGraphMode.NONE
        assert desc.base_precision == BASE_PRECISION_INT4
        assert count() == 2

        # BF16 at the same shape still replays a graph, silently.
        mode, desc = dispatcher.dispatch(
            48,
            num_reqs=48,
            uniform_decode=True,
            has_lora=True,
            num_active_loras=1,
            base_precision=BASE_PRECISION_BF16,
        )
        assert mode == CUDAGraphMode.FULL
        assert desc.base_precision == BASE_PRECISION_BF16
        assert count() == 2

        # Above the capture list: eager as in vanilla, precision carried,
        # no warning (there was never a graph to miss).
        mode, desc = dispatcher.dispatch(
            200, num_reqs=64, has_lora=True, base_precision=BASE_PRECISION_INT4
        )
        assert mode == CUDAGraphMode.NONE
        assert desc == BatchDescriptor(
            num_tokens=200, num_reqs=64, base_precision=BASE_PRECISION_INT4
        )
        assert count() == 2


def test_forced_eager_carries_the_precision(b64_dispatcher):
    mode, desc = b64_dispatcher.dispatch(
        8,
        num_reqs=8,
        has_lora=True,
        base_precision=BASE_PRECISION_INT4,
        valid_modes={CUDAGraphMode.NONE},
    )
    assert mode == CUDAGraphMode.NONE
    assert desc == BatchDescriptor(
        num_tokens=8, num_reqs=8, base_precision=BASE_PRECISION_INT4
    )


# ---------------------------------------------------------------------------
# Vanilla equivalence when disabled
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cudagraph_mode_str, compilation_mode, lora",
    [
        ("FULL", CompilationMode.NONE, False),
        ("FULL_DECODE_ONLY", CompilationMode.NONE, False),
        ("PIECEWISE", CompilationMode.VLLM_COMPILE, False),
        ("PIECEWISE", CompilationMode.VLLM_COMPILE, True),
        ("FULL_AND_PIECEWISE", CompilationMode.VLLM_COMPILE, True),
    ],
)
@pytest.mark.parametrize("policy", ["", "fixed_frontier:8000"])
def test_vanilla_equivalence_when_disabled(
    dual_precision_env, cudagraph_mode_str, compilation_mode, lora, policy
):
    """Flag off (even with a policy string present): the vanilla key sets
    (10 with LoRA specialisation, 2 without), every key BF16, and the
    positional ``dispatch(num_tokens)`` form of the spec-decode callers."""
    dual_precision_env(enabled=False, policy=policy)
    lora_config = LoRAConfig(max_loras=4, specialize_active_lora=True) if lora else None
    config = make_config(cudagraph_mode_str, compilation_mode, [1, 8], 8, lora_config)
    dispatcher = build(config)
    assert dispatcher.int4_capture_max_batch is None

    expected = 10 if lora else 2
    if cudagraph_mode_str in ("FULL_AND_PIECEWISE", "PIECEWISE"):
        assert len(keys(dispatcher, CUDAGraphMode.PIECEWISE)) == expected
    else:
        assert len(keys(dispatcher, CUDAGraphMode.PIECEWISE)) == 0
    if cudagraph_mode_str != "PIECEWISE":
        assert len(keys(dispatcher, CUDAGraphMode.FULL)) == expected
    else:
        assert len(keys(dispatcher, CUDAGraphMode.FULL)) == 0
    for mode in (CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL):
        assert all(
            k.base_precision == BASE_PRECISION_BF16 for k in keys(dispatcher, mode)
        )

    # bare eager path: exactly the vanilla descriptor
    mode, desc = dispatcher.dispatch(15)
    assert (mode, desc) == (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=15))
    mode, desc = dispatcher.dispatch(8, valid_modes={CUDAGraphMode.NONE})
    assert (mode, desc) == (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=8))
    mode, desc = dispatcher.dispatch(8, invalid_modes={CUDAGraphMode.FULL})
    if "PIECEWISE" in cudagraph_mode_str:
        assert mode == CUDAGraphMode.PIECEWISE
        assert desc == BatchDescriptor(num_tokens=8, num_reqs=None, uniform=False)
    else:
        assert (mode, desc) == (CUDAGraphMode.NONE, BatchDescriptor(num_tokens=8))
    # the runner's call with num_reqs and no precision
    mode, desc = dispatcher.dispatch(8, num_reqs=8, uniform_decode=True)
    assert mode != CUDAGraphMode.NONE
    assert desc.base_precision == BASE_PRECISION_BF16
    # a precision override is rejected only when it is not a precision
    with pytest.raises(ValueError):
        dispatcher.dispatch(8, num_reqs=8, base_precision="fp8")


# ---------------------------------------------------------------------------
# V2 model runner guard
# ---------------------------------------------------------------------------


def _engine_config(
    *,
    use_v2: bool = False,
    spec_decode: bool = False,
    fast_prefill: bool = False,
    dp: int = 1,
) -> MagicMock:
    config = MagicMock(spec=VllmConfig)
    config.use_v2_model_runner = use_v2
    config.speculative_config = MagicMock() if spec_decode else None
    config.cache_config = MagicMock()
    config.cache_config.kv_sharing_fast_prefill = fast_prefill
    config.parallel_config = MagicMock()
    config.parallel_config.data_parallel_size = dp
    return config


def test_v2_model_runner_guard(dual_precision_env):
    dual_precision_env(enabled=True, policy="fixed_frontier:8000")
    with pytest.raises(NotImplementedError, match="V2 model runner"):
        check_dual_precision_model_runner(_engine_config(use_v2=True))
    check_dual_precision_model_runner(_engine_config())

    dual_precision_env(enabled=False)
    check_dual_precision_model_runner(_engine_config(use_v2=True))
    check_dual_precision_model_runner(_engine_config())


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"spec_decode": True}, "speculative"),
        ({"fast_prefill": True}, "kv_sharing_fast_prefill"),
        ({"dp": 2}, "data parallelism"),
    ],
)
def test_unsupported_engine_features_are_refused(dual_precision_env, kwargs, match):
    """Paths that dispatch forwards without a base precision (implicitly
    BF16) or choose it per DP rank are refused at worker init."""
    dual_precision_env(enabled=True, policy="fixed_frontier:8000")
    with pytest.raises(NotImplementedError, match=match):
        check_dual_precision_model_runner(_engine_config(**kwargs))
    dual_precision_env(enabled=False)
    check_dual_precision_model_runner(_engine_config(**kwargs))

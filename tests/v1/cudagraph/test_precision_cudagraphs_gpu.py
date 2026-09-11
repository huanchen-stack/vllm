# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU smoke for the precision-keyed CUDA graphs (component C3).

1. ``CUDAGraphWrapper`` captures two graphs for the same shape under the two
   bindings of C2's binder (``BatchDescriptor.base_precision`` is the only
   difference between the keys) and each replays its own precision.
2. End to end on Qwen3.5-9B BF16 + Intel AutoRound INT4 shadow with a zero
   LoRA adapter, ``VLLM_DUAL_PRECISION_POLICY=fixed_frontier:32``, 8 prompts
   x 128 tokens: the graph counts are the archived b64 ladder (45 PIECEWISE
   / 29 FULL), the switch happens (cohort JSONL), every decode step after it
   replays a captured INT4 graph (no eager-fallback warning) and generation
   stays coherent.

Launch under the decision-13 launcher on one free GPU (about 25 GiB, ~5 min):

    run_gpu.sh --gpus 3 --timeout 2100 -- python -m pytest -p no:cacheprovider -q \
        tests/v1/cudagraph/test_precision_cudagraphs_gpu.py -m gpu_smoke
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.gpu_smoke

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

HF_HUB = "/data/huggingface/hub"
QWEN35_9B_BF16 = (
    f"{HF_HUB}/models--Qwen--Qwen3.5-9B/snapshots/"
    "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
QWEN35_9B_AUTOROUND = (
    f"{HF_HUB}/models--Intel--Qwen3.5-9B-int4-AutoRound/snapshots/"
    "29688b8959bebb6d019ddd8f174a5b4bfd670456"
)
REPO_ROOT = Path(__file__).resolve().parents[3]
MAKE_ZERO_LORA = REPO_ROOT / "tools" / "rollout_lora" / "make_zero_lora.py"

FRONTIER = 32
NUM_PROMPTS = 8
MAX_TOKENS = 128
MAX_NUM_SEQS = 64  # the archived b64 ladder: 19 sizes up to 128
CAPTURE_LOG_RE = re.compile(
    r"PIECEWISE=(\d+) \(largest=(\d+)\), FULL=(\d+) \(largest=(\d+)\)"
)
FALLBACK_WARNING = "No matching dynamic-precision CUDA graph"


def _gpu_smoke_requested(request) -> None:
    markexpr = request.config.getoption("-m", default="") or ""
    if "gpu_smoke" not in markexpr and os.environ.get("PS_RUN_GPU_SMOKE") != "1":
        pytest.skip("gpu_smoke tier: run with -m gpu_smoke under run_gpu.sh")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("run under run_gpu.sh (decision 13): CUDA_VISIBLE_DEVICES is unset")


# ---------------------------------------------------------------------------
# 1. wrapper: two graphs for one shape
# ---------------------------------------------------------------------------


def test_wrapper_captures_one_graph_per_precision_for_the_same_shape(request):
    _gpu_smoke_requested(request)
    from tests.model_executor.dual_precision.fakes import (
        build_int4_model,
        build_wrapped_model,
    )
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.compilation.monitor import set_cudagraph_capturing_enabled
    from vllm.config import CUDAGraphMode, VllmConfig
    from vllm.forward_context import BatchDescriptor, set_forward_context
    from vllm.model_executor.dual_precision import (
        BASE_PRECISION_BF16,
        BASE_PRECISION_INT4,
        bind_dual_precision,
        get_binding,
    )
    from vllm.model_executor.dual_precision.loader import attach_shadow_layers

    names = ["model.layers.0.mlp.gate_up_proj", "model.layers.0.mlp.down_proj"]
    hidden, dtype, device = 64, torch.bfloat16, "cuda"
    torch.manual_seed(0)
    model = build_wrapped_model(names, hidden, device=device, dtype=dtype)
    int4_model = build_int4_model(names, set(names), hidden, device=device, dtype=dtype)
    vllm_config = VllmConfig()
    state = attach_shadow_layers(
        model,
        int4_model,
        bf16_layer_policy="none",
        module_policy="all",
        num_layers=1,
        static_forward_context=vllm_config.compilation_config.static_forward_context,
        dtype=dtype,
    )
    assert state.attached == 2
    for name in names:
        wrapper = model.get_submodule(name)
        wrapper.base_layer.weight.data.mul_(hidden**-0.5)
        get_binding(wrapper).int4_or_fallback.weight.data.mul_(hidden**-0.5)
        wrapper.lora_a.normal_(std=0.1)
        wrapper.lora_b.normal_(std=0.1)

    class Block(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.model = m

        def forward(self, x):
            up = self.model.get_submodule(names[0])(x)
            return self.model.get_submodule(names[1])(F.silu(up))

    def reference(x, precision):
        h = x
        for name in names:
            wrapper = model.get_submodule(name)
            binding = get_binding(wrapper)
            base = (
                binding.int4_or_fallback
                if precision == BASE_PRECISION_INT4
                else binding.bf16
            )
            out = F.linear(h, base.weight) + F.linear(
                F.linear(h, wrapper.lora_a), wrapper.lora_b
            )
            h = F.silu(out) if name == names[0] else out
        return h

    block = Block(model)
    wrapper = CUDAGraphWrapper(block, vllm_config, runtime_mode=CUDAGraphMode.FULL)
    static_x = torch.randn(10, hidden, device=device, dtype=dtype)
    descs = {
        p: BatchDescriptor(num_tokens=10, base_precision=p)
        for p in (BASE_PRECISION_BF16, BASE_PRECISION_INT4)
    }
    assert descs[BASE_PRECISION_BF16] != descs[BASE_PRECISION_INT4]
    references = {p: reference(static_x, p) for p in descs}
    assert not torch.allclose(
        references[BASE_PRECISION_BF16], references[BASE_PRECISION_INT4]
    )

    set_cudagraph_capturing_enabled(True)
    try:
        # warmup (eager), then capture one graph per precision, bound
        # exactly as the runner binds before each capture.
        for precision, desc in descs.items():
            bind_dual_precision(model, precision)
            with set_forward_context(
                None,
                vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.NONE,
                batch_descriptor=None,
            ):
                wrapper(static_x)
            with set_forward_context(
                None,
                vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=desc,
            ):
                wrapper(static_x)
            assert desc in wrapper.concrete_cudagraph_entries
        assert len(wrapper.concrete_cudagraph_entries) == 2

        # Replay each key under its own binding: matches the eager reference.
        for precision, desc in descs.items():
            bind_dual_precision(model, precision)
            entry = wrapper.concrete_cudagraph_entries[desc]
            assert entry.cudagraph is not None
            with set_forward_context(
                None,
                vllm_config,
                cudagraph_runtime_mode=CUDAGraphMode.FULL,
                batch_descriptor=desc,
            ):
                out = wrapper(static_x)
            torch.cuda.synchronize()
            torch.testing.assert_close(out, references[precision], rtol=5e-2, atol=5e-2)

        # Replaying the BF16 key while bound to INT4 still yields the BF16
        # result: the selection is baked into the graph, which is exactly
        # why the precision must be part of the key.
        bind_dual_precision(model, BASE_PRECISION_INT4)
        with set_forward_context(
            None,
            vllm_config,
            cudagraph_runtime_mode=CUDAGraphMode.FULL,
            batch_descriptor=descs[BASE_PRECISION_BF16],
        ):
            out = wrapper(static_x)
        torch.cuda.synchronize()
        torch.testing.assert_close(
            out, references[BASE_PRECISION_BF16], rtol=5e-2, atol=5e-2
        )
    finally:
        set_cudagraph_capturing_enabled(False)


# ---------------------------------------------------------------------------
# 2. engine: capture counts, switch, INT4 graphs replayed, coherent text
# ---------------------------------------------------------------------------


class _Records(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _coherent(text: str) -> bool:
    words = text.split()
    if len(words) < 10:
        return False
    counts = collections.Counter(words)
    most_common = counts.most_common(1)[0][1]
    ascii_ratio = sum(ch.isascii() for ch in text) / max(len(text), 1)
    return most_common <= len(words) // 3 and ascii_ratio > 0.9


def test_engine_switch_replays_captured_int4_graphs(request, tmp_path: Path):
    _gpu_smoke_requested(request)
    for path in (QWEN35_9B_BF16, QWEN35_9B_AUTOROUND):
        if not Path(path).is_dir():
            pytest.skip(f"checkpoint not available: {path}")

    adapter = tmp_path / "zero_lora"
    subprocess.run(
        [
            sys.executable,
            str(MAKE_ZERO_LORA),
            "--model",
            QWEN35_9B_BF16,
            "--output",
            str(adapter),
            "--dtype",
            "bfloat16",
        ],
        check=True,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
    )
    cohorts = tmp_path / "obs" / "online_switch_cohorts.jsonl"

    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["ROLLOUT_QLORA"] = "1"
    os.environ["VLLM_DUAL_PRECISION_ROLLOUT"] = "1"
    os.environ["VLLM_DUAL_PRECISION_INT4_MODEL"] = QWEN35_9B_AUTOROUND
    os.environ["VLLM_DUAL_PRECISION_BF16_LAYERS"] = "none"
    os.environ["VLLM_DUAL_PRECISION_INT4_MODULES"] = "all"
    os.environ["VLLM_DUAL_PRECISION_POLICY"] = f"fixed_frontier:{FRONTIER}"
    os.environ["VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS"] = str(cohorts)
    os.environ["VLLM_DUAL_PRECISION_RELOAD_POLICY_EACH_ROLLOUT"] = "0"

    records = _Records()
    logging.getLogger("vllm").addHandler(records)

    from vllm import LLM, SamplingParams
    from vllm.config import CUDAGraphMode
    from vllm.lora.request import LoRARequest
    from vllm.model_executor.dual_precision import (
        BASE_PRECISION_BF16,
        BASE_PRECISION_INT4,
        get_active_precision,
    )

    llm = LLM(
        model=QWEN35_9B_BF16,
        dtype="bfloat16",
        enable_lora=True,
        max_loras=1,
        max_lora_rank=16,
        max_model_len=1024,
        max_num_seqs=MAX_NUM_SEQS,
        gpu_memory_utilization=0.75,
        seed=0,
    )
    try:
        engine_core = llm.llm_engine.engine_core.engine_core
        scheduler = engine_core.scheduler
        assert scheduler.precision_switcher is not None
        runner = engine_core.model_executor.driver_worker.worker.model_runner
        dispatcher = runner.cudagraph_dispatcher
        assert runner.dual_precision_enabled
        assert dispatcher.int4_capture_max_batch == 32
        piecewise = len(dispatcher.cudagraph_keys[CUDAGraphMode.PIECEWISE])
        full = len(dispatcher.cudagraph_keys[CUDAGraphMode.FULL])
        assert (piecewise, full) == (45, 29), (piecewise, full)
        capture_lines = [m for m in records.messages if CAPTURE_LOG_RE.search(m)]
        assert capture_lines, "no 'Profiling CUDA graph memory' line"
        assert CAPTURE_LOG_RE.search(capture_lines[-1]).groups() == (
            "45",
            "128",
            "29",
            "64",
        )
        finished = [m for m in records.messages if "Graph capturing finished" in m]
        assert finished, "no 'Graph capturing finished' line"
        # capture left the model bound to BF16
        assert get_active_precision(runner.get_model()) == BASE_PRECISION_BF16
        assert not dispatcher._missing_precision_keys_logged

        # record every dispatch of the generation
        dispatched: list[tuple[str, object, object]] = []
        real_dispatch = dispatcher.dispatch

        def recording_dispatch(num_tokens, **kwargs):
            mode, desc = real_dispatch(num_tokens, **kwargs)
            dispatched.append((kwargs.get("base_precision"), mode, desc))
            return mode, desc

        dispatcher.dispatch = recording_dispatch  # type: ignore[method-assign]

        prompts = [
            f"Question {i}: explain in a few sentences why the sky is blue."
            for i in range(NUM_PROMPTS)
        ]
        params = SamplingParams(max_tokens=MAX_TOKENS, ignore_eos=True, temperature=0.0)
        outputs = llm.generate(
            prompts, params, lora_request=LoRARequest("zero", 1, str(adapter))
        )
        dispatcher.dispatch = real_dispatch  # type: ignore[method-assign]

        assert len(outputs) == NUM_PROMPTS
        texts = [o.outputs[0].text for o in outputs]
        assert all(len(o.outputs[0].token_ids) == MAX_TOKENS for o in outputs)
        for text in texts:
            assert _coherent(text), text

        # the switch happened at the frontier, in one cohort of 8
        switches = scheduler.precision_switcher.switches
        assert len(switches) == 1
        assert switches[0].committed_frontier == FRONTIER
        assert switches[0].applied_live_requests == NUM_PROMPTS
        rows = [json.loads(line) for line in cohorts.read_text().splitlines()]
        assert len(rows) == 1 and rows[0]["event"] == "switch_cohort"
        assert len(rows[0]["requests"]) == NUM_PROMPTS

        # dispatch: BF16 before, INT4 after; the INT4 decode steps replayed
        # captured graphs (FULL at the padded decode size) and no key fell
        # back to eager.
        precisions = [p for p, _, _ in dispatched]
        assert precisions[0] == BASE_PRECISION_BF16
        first_int4 = precisions.index(BASE_PRECISION_INT4)
        assert first_int4 > 0
        assert all(p == BASE_PRECISION_BF16 for p in precisions[:first_int4])
        assert all(p == BASE_PRECISION_INT4 for p in precisions[first_int4:])
        int4_steps = dispatched[first_int4:]
        assert len(int4_steps) >= (MAX_TOKENS - FRONTIER) - 4
        for _, mode, desc in int4_steps:
            assert mode == CUDAGraphMode.FULL, (mode, desc)
            assert desc.base_precision == BASE_PRECISION_INT4
            assert desc.num_tokens == NUM_PROMPTS and desc.uniform and desc.has_lora
        bf16_graph_steps = [
            m for _, m, _ in dispatched[:first_int4] if m == CUDAGraphMode.FULL
        ]
        assert bf16_graph_steps, "no BF16 decode step replayed a graph"
        assert not dispatcher._missing_precision_keys_logged
        assert not any(FALLBACK_WARNING in m for m in records.messages)
        # the last forward ran INT4
        assert get_active_precision(runner.get_model()) == BASE_PRECISION_INT4

        summary = {
            "piecewise": piecewise,
            "full": full,
            "capture_line": capture_lines[-1],
            "finished_line": finished[-1],
            "dispatches": len(dispatched),
            "first_int4_dispatch": first_int4,
            "int4_full_graph_steps": len(int4_steps),
            "switch": {
                "committed_frontier": switches[0].committed_frontier,
                "applied_response_tokens": switches[0].applied_response_tokens,
                "applied_live_requests": switches[0].applied_live_requests,
            },
            "texts": texts,
        }
        (tmp_path / "summary.json").write_text(json.dumps(summary, indent=2))
        print("C3_E2E_SUMMARY", json.dumps(summary))
    finally:
        logging.getLogger("vllm").removeHandler(records)
        del llm


if __name__ == "__main__":
    raise SystemExit(
        pytest.main([__file__, "-m", "gpu_smoke", "-p", "no:cacheprovider", "-q", "-s"])
    )

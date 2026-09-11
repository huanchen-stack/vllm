# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-model residency gate: Qwen3.5-9B BF16 + Intel AutoRound INT4 shadow.

Loads both checkpoints through the engine (``VLLM_DUAL_PRECISION_ROLLOUT=1``),
then inspects the attached state inside the worker: attach counts must match
the archived runs (152 attached / 134 fallback, ``BF16_LAYERS=none``) and every
attached INT4 linear must agree with its BF16 twin on a random input (worst
cosine >= 0.95; measured 0.9856). One GPU, ~25 GiB, about one minute.
"""

import os
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.gpu_smoke

HF_HUB = "/data/huggingface/hub"
QWEN35_9B_BF16 = (
    f"{HF_HUB}/models--Qwen--Qwen3.5-9B/snapshots/"
    "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
)
QWEN35_9B_AUTOROUND = (
    f"{HF_HUB}/models--Intel--Qwen3.5-9B-int4-AutoRound/snapshots/"
    "29688b8959bebb6d019ddd8f174a5b4bfd670456"
)

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
for _path in (QWEN35_9B_BF16, QWEN35_9B_AUTOROUND):
    if not os.path.isdir(_path):
        pytest.skip(f"checkpoint not available: {_path}", allow_module_level=True)


def _inspect_shadow(model) -> dict:
    """Runs inside the worker via ``LLM.apply_model``."""
    from vllm.model_executor.dual_precision import (
        SHADOW_MODULE_NAME,
        get_dual_precision_state,
    )
    from vllm.model_executor.dual_precision.validation import (
        compare_shadow_numerics,
    )

    state = get_dual_precision_state(model)
    assert state is not None
    store = model._modules[SHADOW_MODULE_NAME]
    generator = torch.Generator(device="cuda").manual_seed(0)
    results = []
    for binding in state.bindings:
        if not binding.shadow_active:
            continue
        item = compare_shadow_numerics(
            binding.module_name,
            binding.bf16,
            binding.int4_or_fallback,
            torch.bfloat16,
            generator=generator,
        )
        results.append((item.name, item.cosine, item.relative_rmse))
    results.sort(key=lambda item: item[1])
    return {
        "num_shadow_linears": state.num_shadow_linears,
        "attached": state.attached,
        "policy_bf16": state.policy_bf16,
        "fallback": state.fallback,
        "unwrapped": state.unwrapped,
        "store_layers": len(store),
        "store_gib": state.shadow_bytes / (1 << 30),
        "store_is_registered_once": list(model._modules).count(SHADOW_MODULE_NAME),
        "worst": results[:5],
        "lora_wrapped_store": any(hasattr(m, "base_layer") for m in store.modules()),
    }


@pytest.fixture(scope="module")
def llm(monkeypatch_module):
    from vllm import LLM

    # apply_model ships a function to the worker; allow pickle for the test.
    monkeypatch_module.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch_module.setenv("VLLM_DUAL_PRECISION_ROLLOUT", "1")
    monkeypatch_module.setenv("VLLM_DUAL_PRECISION_INT4_MODEL", QWEN35_9B_AUTOROUND)
    monkeypatch_module.setenv("VLLM_DUAL_PRECISION_BF16_LAYERS", "none")
    monkeypatch_module.setenv("VLLM_DUAL_PRECISION_INT4_MODULES", "all")
    monkeypatch_module.setenv("VLLM_DUAL_PRECISION_VALIDATE_LIFECYCLE", "1")
    llm = LLM(
        model=QWEN35_9B_BF16,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
        enable_sleep_mode=True,
        # Keep the vision tower (as the archived runs did): its 110 linears
        # are the bulk of the 134 fallback linears in the recorded log line.
        seed=0,
    )
    yield llm
    del llm


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_qwen35_9b_autoround_shadow_attach_counts_and_numerics(llm):
    (report,) = llm.apply_model(_inspect_shadow)

    assert (report["num_shadow_linears"], report["attached"], report["fallback"]) == (
        286,
        152,
        134,
    )
    assert report["policy_bf16"] == 0 and report["unwrapped"] == 0
    assert report["store_layers"] == 152 and report["store_is_registered_once"] == 1
    assert not report["lora_wrapped_store"]
    # 152 GPTQ-packed linears of a 9B model: ~4 GiB of packed weights+scales.
    assert 3.0 < report["store_gib"] < 6.0, report["store_gib"]
    # Measured 2026-09-11 on GPU 3: store 3.32 GiB; worst cosine 0.9856
    # (layers.30.linear_attn.in_proj_ba, rel_rmse 0.169); the AWQ-labelled
    # cyankiwi shadow in the archived guard run bottomed out at 0.902.
    worst_name, worst_cos, worst_rmse = report["worst"][0]
    assert worst_cos >= 0.95, report["worst"]
    print("shadow store GiB:", report["store_gib"], "worst:", report["worst"])


def test_engine_generates_with_shadow_attached(llm):
    from vllm import SamplingParams

    outputs = llm.generate(
        ["The capital of France is"], SamplingParams(temperature=0, max_tokens=8)
    )
    assert len(outputs[0].outputs[0].token_ids) == 8


def _lifecycle_after_weight_sync(model) -> dict:
    """Runs inside the worker after sleep(1)/wake_up: replay verl's weight-sync
    post-processing with and without the shadow hidden, then probe."""
    from vllm.model_executor.dual_precision import (
        BASE_PRECISION_BF16,
        BASE_PRECISION_INT4,
        SHADOW_MODULE_NAME,
        bind_dual_precision,
        get_dual_precision_state,
        mark_lifecycle_event,
    )
    from vllm.model_executor.dual_precision.validation import run_lifecycle_probes
    from vllm.model_executor.model_loader.utils import process_weights_after_loading

    state = get_dual_precision_state(model)
    device = next(model.parameters()).device
    # process_weights_after_loading only reads dtype/quantization from it.
    model_config = SimpleNamespace(dtype=torch.bfloat16, quantization=None)

    # The worker marked the sleep/wake-up; the first INT4 bind is the baseline
    # validation and consumes them.
    events_after_wake = list(state.pending_lifecycle_events)
    bind_dual_precision(model, BASE_PRECISION_INT4)
    events_after_baseline = list(state.pending_lifecycle_events)
    bind_dual_precision(model, BASE_PRECISION_BF16)

    # verl's _hide_dual_precision_shadow_model: pop the store around the call.
    shadow = model._modules.pop(SHADOW_MODULE_NAME)
    try:
        process_weights_after_loading(model, model_config, device)
    finally:
        model._modules[SHADOW_MODULE_NAME] = shadow
    # verl streams the base weights through model.load_weights (wrapped at
    # attach to mark the event); arm the re-validation the same way.
    load_weights_wrapped = "load_weights" in model.__dict__
    mark_lifecycle_event(model, "load_weights")
    bind_dual_precision(model, BASE_PRECISION_INT4)  # re-validation after sync
    events_after_sync_bind = list(state.pending_lifecycle_events)
    with_hide = [
        (r.name, r.exact, r.max_abs)
        for r in run_lifecycle_probes(state.lifecycle_probes)
    ]

    # Without the helper the Marlin repack visits the packed shadow again: on
    # this vLLM base it asserts (the packed parameter is no longer a
    # BasevLLMParameter after the first repack); older bases silently
    # re-packed and corrupted the store. Either way the probes cannot pass.
    without_hide_error = None
    try:
        process_weights_after_loading(model, model_config, device)
    except Exception as exc:  # noqa: BLE001 - the failure mode is the point
        without_hide_error = f"{type(exc).__name__}: {exc}"
    without_hide = [
        (r.name, r.exact, r.max_abs)
        for r in run_lifecycle_probes(state.lifecycle_probes)
    ]
    return {
        "num_probes": len(state.lifecycle_probes),
        "validated": state.lifecycle_validated,
        "sanity_probe_pending": state.sanity_probe_pending,
        "events_after_wake": events_after_wake,
        "events_after_baseline": events_after_baseline,
        "events_after_sync_bind": events_after_sync_bind,
        "load_weights_wrapped": load_weights_wrapped,
        "with_hide": with_hide,
        "without_hide": without_hide,
        "without_hide_error": without_hide_error,
    }


def test_lifecycle_probes_exact_after_sleep_wake_and_weight_sync(llm):
    """Audit test 5. Runs last: it deliberately corrupts the shadow store."""
    llm.sleep(level=1)
    llm.wake_up()
    (report,) = llm.apply_model(_lifecycle_after_weight_sync)

    assert report["num_probes"] == 6 and report["validated"]
    # The engine loaded real base weights, so the sanity probe ran at attach.
    assert report["sanity_probe_pending"] is False
    # Defect 4: the worker's sleep/wake_up armed a re-validation, the bind
    # consumed it, and the weight-sync mark armed another one.
    assert report["events_after_wake"] == ["sleep", "wake_up"], report
    assert report["events_after_baseline"] == []
    assert report["events_after_sync_bind"] == []
    assert report["load_weights_wrapped"] is True
    assert all(exact and max_abs == 0 for _, exact, max_abs in report["with_hide"]), (
        report
    )
    # The hide helper is load-bearing: a second repack either raises inside
    # the Marlin kernel (this base) or breaks the probes.
    assert report["without_hide_error"] is not None or not any(
        exact for _, exact, _ in report["without_hide"]
    ), report
    print("without hide:", report["without_hide_error"], report["without_hide"][:2])

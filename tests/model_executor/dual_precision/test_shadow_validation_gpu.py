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
    llm = LLM(
        model=QWEN35_9B_BF16,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=1024,
        max_num_seqs=4,
        gpu_memory_utilization=0.75,
        enforce_eager=True,
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

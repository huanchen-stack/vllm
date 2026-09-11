# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""verl's rollout flow on a real model (integration defect 1): the engine loads
``load_format=dummy`` base weights, the INT4 shadow must still be real, the
sanity probe is deferred until the trainer's base sync arrives through
``model.load_weights``, and the INT4 decode after the sync is coherent.

Qwen3.5-4B BF16 (dummy) + Intel AutoRound INT4 shadow, uniform_w4, a zero LoRA
adapter, ``VLLM_LOGGING_LEVEL=WARN`` as verl sets it. One GPU, ~4 minutes.
"""

import glob
import json
import logging
import os
import re

import pytest
import torch

pytestmark = pytest.mark.gpu_smoke

HF_HUB = "/data/huggingface/hub"
QWEN35_4B_BF16 = (
    f"{HF_HUB}/models--Qwen--Qwen3.5-4B/snapshots/"
    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)
QWEN35_4B_AUTOROUND = (
    f"{HF_HUB}/models--Intel--Qwen3.5-4B-int4-AutoRound/snapshots/"
    "0a857215cab37e8caa857a053005a73cf98ff2c4"
)
# The recipe's LoRA targets (examples/precision_scheduler/models/qwen3_5_4b.yaml
# in verl); vLLM wraps only these, and verl's base sync renames exactly these
# to ``<module>.base_layer.weight`` (fsdp_utils.replace_lora_wrapper).
TARGETS = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj",
)  # fmt: skip
PROMPTS = [
    (
        "Question: Tina has 12 red pens, 9 fewer green pens than red, and 3 more "
        "blue pens than green. How many pens in total? Answer step by step."
    ),
    "Explain in two sentences why the sky is blue.",
    "Write one sentence about the history of Rome.",
    "List three fruits and say which one is your favourite and why.",
]
WORD = re.compile(r"[A-Za-z]{3,}")

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)
for _path in (QWEN35_4B_BF16, QWEN35_4B_AUTOROUND):
    if not os.path.isdir(_path):
        pytest.skip(f"checkpoint not available: {_path}", allow_module_level=True)


def _coherent(text: str) -> bool:
    words = WORD.findall(text)
    return len(words) >= 8 and len({w.lower() for w in words}) >= max(
        5, len(words) // 3
    )


def _write_zero_adapter(path: str, rank: int = 16) -> str:
    """A rank-16 adapter with zero B matrices on every target module: the
    QLoRA path is exercised and the output equals the base model's."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    shapes: dict[str, tuple[int, ...]] = {}
    for shard in sorted(glob.glob(f"{QWEN35_4B_BF16}/*.safetensors")):
        with safe_open(shard, "pt") as f:
            for key in f.keys():  # noqa: SIM118 - safe_open is not a dict
                module = key[: -len(".weight")]
                if key.endswith(".weight") and module.split(".")[-1] in TARGETS:
                    shapes[module] = tuple(f.get_slice(key).get_shape())
    tensors = {}
    for module, (out_f, in_f) in shapes.items():
        if module.startswith("mtp."):
            continue
        peft = "base_model.model." + module
        dtype = torch.bfloat16
        tensors[f"{peft}.lora_A.weight"] = torch.zeros(rank, in_f, dtype=dtype)
        tensors[f"{peft}.lora_B.weight"] = torch.zeros(out_f, rank, dtype=dtype)
    os.makedirs(path, exist_ok=True)
    save_file(tensors, f"{path}/adapter_model.safetensors")
    config = {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "bias": "none",
        "target_modules": list(TARGETS),
        "base_model_name_or_path": QWEN35_4B_BF16,
    }
    with open(f"{path}/adapter_config.json", "w") as f:
        json.dump(config, f)
    return path


def _verl_name(key: str) -> str:
    if key.endswith(".weight") and key[: -len(".weight")].split(".")[-1] in TARGETS:
        return key[: -len(".weight")] + ".base_layer.weight"
    return key


def _state_report(model) -> dict:
    from vllm.model_executor.dual_precision import get_dual_precision_state

    state = get_dual_precision_state(model)
    return {
        "sanity_probe_pending": state.sanity_probe_pending,
        "pending": list(state.pending_lifecycle_events),
        "load_weights_wrapped": "load_weights" in model.__dict__,
        "shadow_load_format": state.shadow_load_format,
        "attached": state.attached,
    }


def _sync_bf16_like_verl(model) -> int:
    from safetensors.torch import load_file

    loaded = 0
    for shard in sorted(glob.glob(f"{QWEN35_4B_BF16}/*.safetensors")):
        tensors = load_file(shard, device="cuda")
        items = [
            (_verl_name(k), v) for k, v in tensors.items() if not k.startswith("mtp.")
        ]
        loaded += len(model.load_weights(items) or ())
        del tensors
        torch.cuda.empty_cache()
    return loaded


def test_dummy_engine_loads_real_shadow_and_decodes_after_base_sync(
    tmp_path, monkeypatch, caplog
):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("ROLLOUT_QLORA", "1")
    monkeypatch.setenv("VLLM_DUAL_PRECISION_ROLLOUT", "1")
    monkeypatch.setenv("VLLM_DUAL_PRECISION_INT4_MODEL", QWEN35_4B_AUTOROUND)
    monkeypatch.setenv("VLLM_DUAL_PRECISION_BF16_LAYERS", "none")
    monkeypatch.setenv("VLLM_DUAL_PRECISION_INT4_MODULES", "all")
    monkeypatch.setenv("VLLM_DUAL_PRECISION_POLICY", "uniform_w4")
    monkeypatch.setenv("VLLM_DUAL_PRECISION_VALIDATE_LIFECYCLE", "1")
    monkeypatch.setenv("VLLM_ALLOW_RUNTIME_LORA_UPDATING", "true")
    monkeypatch.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "WARN")
    adapter = _write_zero_adapter(str(tmp_path / "zero_lora"))
    # Records are collected at the level verl runs at; what the test asserts
    # on is therefore exactly what a verl log would contain.
    caplog.set_level(logging.WARNING, logger="vllm")

    llm = LLM(
        model=QWEN35_4B_BF16,
        dtype="bfloat16",
        enable_lora=True,
        max_loras=1,
        max_lora_rank=16,
        lora_target_modules=list(TARGETS),
        max_model_len=2048,
        max_num_seqs=32,
        gpu_memory_utilization=0.6,
        seed=0,
        enable_sleep_mode=True,
        enable_prefix_caching=False,
        load_format="dummy",
    )
    try:
        params = SamplingParams(max_tokens=64, temperature=0.0)
        lora = LoRARequest("zero", 1, adapter)

        (init,) = llm.apply_model(_state_report)
        assert init["shadow_load_format"] == "auto" and init["attached"] == 152
        assert init["sanity_probe_pending"] and init["load_weights_wrapped"]
        err = caplog.text
        assert "engine load_format=dummy, but the INT4 shadow is loaded with" in err
        assert (
            "sanity probe on language_model.model.layers.0." in err
            and "deferred" in err
        )
        assert (
            "precision=int4, lora_base_layers=152" in err
        )  # capture-time bind at WARN
        assert "int4_shadow_active=152" in err

        # Graph capture bound INT4 before any sync: the probe must stay pending.
        (before,) = llm.apply_model(_state_report)
        assert before["sanity_probe_pending"] and before["pending"] == []

        (loaded,) = llm.apply_model(_sync_bf16_like_verl)
        assert loaded > 600, loaded
        (after_sync,) = llm.apply_model(_state_report)
        assert (
            after_sync["pending"] == ["load_weights"]
            and after_sync["sanity_probe_pending"]
        )

        outputs = llm.generate(PROMPTS, params, lora_request=lora)
        texts = [o.outputs[0].text for o in outputs]
        assert all(_coherent(t) for t in texts), texts
        assert "Green pens" in texts[0] or "green pens" in texts[0], texts[0]
        (final,) = llm.apply_model(_state_report)
        assert final == {**after_sync, "pending": [], "sanity_probe_pending": False}
        err = caplog.text
        assert "lifecycle validation at first INT4 bind after load_weights" in err
        assert "exact=False" not in err
        print("post-sync INT4 decode:", texts[0][:80])
    finally:
        del llm

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for tools/precision_scheduler/models/prepare_phi4mini.py."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "precision_scheduler" / "models"))

import prepare_phi4mini as tool  # noqa: E402


class _Tok:
    def __init__(self, chat_template="T", eos=1, size=100):
        self.eos_token_id = eos
        self.bos_token_id = 0
        self.pad_token_id = 0
        self.chat_template = chat_template
        self._size = size

    def __len__(self):
        return self._size


def _cfg(model_type="phi3", vocab=100, quant=None, archs=("Phi3ForCausalLM",)):
    ns = SimpleNamespace(
        model_type=model_type,
        vocab_size=vocab,
        architectures=list(archs),
        max_position_embeddings=131072,
    )
    if quant is not None:
        ns.quantization_config = quant
    return ns


CT_W4 = {
    "quant_method": "compressed-tensors",
    "config_groups": {"group_0": {"weights": {"num_bits": 4, "group_size": 128}}},
}


def test_audit_accepts_compressed_tensors_w4_pair():
    audit = tool.audit_pair(_cfg(), _cfg(quant=CT_W4), _Tok(), _Tok())
    assert audit["quant_method"] == "compressed-tensors"
    assert audit["num_bits"] == 4
    assert audit["bf16_tokenizer_fingerprint"] == audit["w4_tokenizer_fingerprint"]


def test_audit_rejects_awq_and_mismatches():
    awq = {"quant_method": "awq", "bits": 4}
    with pytest.raises(RuntimeError, match="AWQ"):
        tool.audit_pair(_cfg(), _cfg(quant=awq), _Tok(), _Tok())
    with pytest.raises(RuntimeError, match="architecture mismatch"):
        tool.audit_pair(_cfg(vocab=100), _cfg(vocab=101, quant=CT_W4), _Tok(), _Tok())
    with pytest.raises(RuntimeError, match="chat template"):
        tool.audit_pair(_cfg(), _cfg(quant=CT_W4), _Tok("A"), _Tok("B"))
    with pytest.raises(RuntimeError, match="not supported"):
        tool.audit_pair(
            _cfg(archs=("Phi4FlashForCausalLM",)), _cfg(quant=CT_W4), _Tok(), _Tok()
        )


def test_zero_lora_uses_native_fused_names_and_zero_lora_b(tmp_path):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    save_file(
        {
            "model.layers.0.self_attn.qkv_proj.weight": torch.zeros(12, 8),
            "model.layers.0.self_attn.o_proj.weight": torch.zeros(8, 8),
            "model.layers.0.mlp.gate_up_proj.weight": torch.zeros(32, 8),
            "model.layers.0.mlp.down_proj.weight": torch.zeros(8, 16),
            "model.layers.0.input_layernorm.weight": torch.zeros(8),
            "model.embed_tokens.weight": torch.zeros(100, 8),
        },
        str(snapshot / "model.safetensors"),
    )
    shapes = tool.linear_shapes(snapshot)
    assert set(shapes) == {
        "model.layers.0.self_attn.qkv_proj",
        "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_up_proj",
        "model.layers.0.mlp.down_proj",
    }

    inventory = tool.build_zero_lora(shapes, tmp_path / "adapter", "base", rank=4)

    config = json.loads((tmp_path / "adapter" / "adapter_config.json").read_text())
    assert config["target_modules"] == [
        "down_proj",
        "gate_up_proj",
        "o_proj",
        "qkv_proj",
    ]
    assert config["r"] == config["lora_alpha"] == 4
    tensors = load_file(str(tmp_path / "adapter" / "adapter_model.safetensors"))
    assert tensors[
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight"
    ].shape == (4, 16)
    b = tensors["base_model.model.model.layers.0.mlp.down_proj.lora_B.weight"]
    assert b.shape == (8, 4) and torch.count_nonzero(b) == 0
    assert inventory["all_lora_B_zero"] and inventory["linear_modules"] == 4


def test_zero_lora_restricts_multimodal_checkpoints_to_language_tower(tmp_path):
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    save_file(
        {
            "model.language_model.layers.0.mlp.down_proj.weight": torch.zeros(8, 16),
            "model.vision_tower.layers.0.mlp.fc1.weight": torch.zeros(8, 16),
        },
        str(snapshot / "model.safetensors"),
    )
    assert list(tool.linear_shapes(snapshot)) == [
        "model.language_model.layers.0.mlp.down_proj"
    ]

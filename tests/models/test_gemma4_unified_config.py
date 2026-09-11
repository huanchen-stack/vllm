# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""gemma4_unified registry glue and the Gemma4ForCausalLM YOCO k_norm fix (CPU)."""

import json
import shutil
from pathlib import Path

import pytest
import torch

from vllm.model_executor.models.registry import _TEXT_GENERATION_MODELS, ModelRegistry
from vllm.transformers_utils.config import get_config
from vllm.transformers_utils.model_arch_config_convertor import (
    MODEL_ARCH_CONFIG_CONVERTORS,
    Gemma4ModelArchConfigConvertor,
)

FIXTURE = Path(__file__).parent / "fixtures" / "gemma4_unified_12b_config.json"

UNIFIED_ARCH = "Gemma4UnifiedForConditionalGeneration"

# The module-scoped distributed env below is torn down once at module end
# instead of by the per-test cleanup fixture in tests/conftest.py.
pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture(scope="module")
def unified_model_dir(tmp_path_factory):
    target = tmp_path_factory.mktemp("gemma4_unified_12b")
    shutil.copy(FIXTURE, target / "config.json")
    return str(target)


def test_get_config_resolves_gemma4_unified(unified_model_dir):
    from vllm.transformers_utils.configs import Gemma4UnifiedConfig

    config = get_config(unified_model_dir, trust_remote_code=False)

    assert isinstance(config, Gemma4UnifiedConfig)
    assert config.model_type == "gemma4_unified"
    assert config.architectures == [UNIFIED_ARCH]
    text = config.text_config
    assert text.hidden_size == 3840
    assert text.num_hidden_layers == 48
    assert text.global_head_dim == 512
    raw = json.loads(FIXTURE.read_text())
    assert raw["text_config"]["model_type"] == "gemma4_unified_text"


def test_registry_maps_unified_arch_to_text_only_gemma4():
    assert _TEXT_GENERATION_MODELS[UNIFIED_ARCH] == ("gemma4", "Gemma4ForCausalLM")
    assert UNIFIED_ARCH in ModelRegistry.get_supported_archs()
    model_cls = ModelRegistry._try_load_model_cls(UNIFIED_ARCH)
    from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM

    assert model_cls is Gemma4ForCausalLM


def test_arch_config_convertors_cover_unified_types():
    convertors = MODEL_ARCH_CONFIG_CONVERTORS
    assert convertors["gemma4_unified"] is Gemma4ModelArchConfigConvertor
    assert convertors["gemma4_unified_text"] is Gemma4ModelArchConfigConvertor


def test_models_config_map_covers_unified_arch():
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, Gemma4Config

    assert MODELS_CONFIG_MAP[UNIFIED_ARCH] is Gemma4Config


# --------------------------------------------------------------------------- #
# k_norm completeness on a tiny Gemma4ForCausalLM with YOCO KV-shared layers   #
# --------------------------------------------------------------------------- #

TINY_TEXT_CONFIG = {
    "architectures": ["Gemma4ForCausalLM"],
    "model_type": "gemma4_text",
    "hidden_size": 64,
    "intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_global_key_value_heads": 1,
    "head_dim": 64,
    "global_head_dim": 64,
    "num_kv_shared_layers": 2,
    "layer_types": [
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "full_attention",
    ],
    "sliding_window": 32,
    "vocab_size": 256,
    "max_position_embeddings": 128,
    "rms_norm_eps": 1e-6,
    "attention_bias": False,
    "rope_parameters": {
        "full_attention": {
            "rope_theta": 10000.0,
            "rope_type": "default",
            "partial_rotary_factor": 0.25,
        },
        "sliding_attention": {
            "rope_theta": 10000.0,
            "rope_type": "default",
            "partial_rotary_factor": 1.0,
        },
    },
    "hidden_size_per_layer_input": 0,
    "final_logit_softcapping": 30.0,
    "tie_word_embeddings": True,
    "attention_k_eq_v": True,
    "dtype": "bfloat16",
    "pad_token_id": 0,
    "bos_token_id": 2,
    "eos_token_id": 1,
    "hidden_activation": "gelu_pytorch_tanh",
    "enable_moe_block": False,
}


@pytest.fixture(scope="module")
def tiny_gemma4(tmp_path_factory):
    from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.model_executor.models.gemma4 import Gemma4ForCausalLM
    from vllm.utils.network_utils import get_open_port

    model_dir = tmp_path_factory.mktemp("tiny_gemma4")
    (model_dir / "config.json").write_text(json.dumps(TINY_TEXT_CONFIG))
    model_config = ModelConfig(
        model=str(model_dir),
        tokenizer=str(model_dir),
        skip_tokenizer_init=True,
        dtype="bfloat16",
        seed=0,
        enforce_eager=True,
    )
    vllm_config = VllmConfig(model_config=model_config)
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            local_rank=0,
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1)
        with torch.device("cpu"):
            model = Gemma4ForCausalLM(vllm_config=vllm_config, prefix="")
    yield model
    cleanup_dist_env_and_memory()


def _hf_style_checkpoint(model, drop_kv_shared_k_norm: bool):
    """Turn the model's own parameters into HF-style checkpoint tensors."""
    state = {name: p.detach().clone() for name, p in model.named_parameters()}
    num_layers = model.config.num_hidden_layers
    first_shared = num_layers - model.config.num_kv_shared_layers
    tensors = []
    for name, tensor in state.items():
        if "qkv_proj" in name or "gate_up_proj" in name:
            continue
        if drop_kv_shared_k_norm and "k_norm" in name:
            layer = int(name.split("layers.")[1].split(".")[0])
            if layer >= first_shared:
                continue
        tensors.append((name, tensor))
    for layer_idx in range(num_layers):
        attn = model.model.layers[layer_idx].self_attn
        qkv = state[f"model.layers.{layer_idx}.self_attn.qkv_proj.weight"]
        q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], dim=0)
        tensors.append((f"model.layers.{layer_idx}.self_attn.q_proj.weight", q))
        tensors.append((f"model.layers.{layer_idx}.self_attn.k_proj.weight", k))
        if model.config.layer_types[layer_idx] != "full_attention":
            # k_eq_v: full-attention layers carry no v_proj in the checkpoint.
            tensors.append((f"model.layers.{layer_idx}.self_attn.v_proj.weight", v))
        gate_up = state[f"model.layers.{layer_idx}.mlp.gate_up_proj.weight"]
        gate, up = gate_up.chunk(2, dim=0)
        tensors.append((f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate))
        tensors.append((f"model.layers.{layer_idx}.mlp.up_proj.weight", up))
    return tensors


def test_kv_shared_layers_are_flagged(tiny_gemma4):
    flags = [layer.self_attn.is_kv_shared_layer for layer in tiny_gemma4.model.layers]
    assert flags == [False, False, True, True]


def test_load_weights_is_complete_without_kv_shared_k_norm(tiny_gemma4):
    checkpoint = _hf_style_checkpoint(tiny_gemma4, drop_kv_shared_k_norm=True)
    names = {name for name, _ in checkpoint}
    assert "model.layers.3.self_attn.k_norm.weight" not in names

    loaded = tiny_gemma4.load_weights(iter(checkpoint))

    expected = {name for name, _ in tiny_gemma4.named_parameters()}
    assert expected - loaded == set(), expected - loaded
    shared_k_norms = {
        "model.layers.2.self_attn.k_norm.weight",
        "model.layers.3.self_attn.k_norm.weight",
    }
    assert shared_k_norms <= loaded


def test_vision_embedder_tensors_are_skipped(tiny_gemma4):
    checkpoint = _hf_style_checkpoint(tiny_gemma4, drop_kv_shared_k_norm=False)
    checkpoint.append(
        ("model.vision_embedder.patch_projection.weight", torch.zeros(4, 4))
    )

    loaded = tiny_gemma4.load_weights(iter(checkpoint))

    expected = {name for name, _ in tiny_gemma4.named_parameters()}
    assert expected <= loaded
    assert not any("vision_embedder" in name for name in loaded)


def test_kv_shared_attention_never_reads_k_norm(tiny_gemma4, monkeypatch):
    attn = tiny_gemma4.model.layers[3].self_attn
    assert attn.is_kv_shared_layer

    def _boom(*args, **kwargs):
        raise AssertionError("k_norm must not be evaluated on a KV-shared layer")

    monkeypatch.setattr(attn.k_norm, "forward", _boom)
    monkeypatch.setattr(attn.attn, "forward", lambda q, k, v: q)
    # The fused rotary op has no CPU kernel; RoPE is irrelevant to this check.
    monkeypatch.setattr(attn.rotary_emb, "forward", lambda positions, q, k: (q, k))
    dtype = next(attn.parameters()).dtype
    hidden = torch.zeros(3, tiny_gemma4.config.hidden_size, dtype=dtype)
    positions = torch.arange(3)

    out = attn(positions, hidden)

    assert out.shape == hidden.shape

    # And a non-shared layer does read it.
    non_shared = tiny_gemma4.model.layers[1].self_attn
    monkeypatch.setattr(non_shared.k_norm, "forward", _boom)
    monkeypatch.setattr(non_shared.attn, "forward", lambda q, k, v: q)
    monkeypatch.setattr(
        non_shared.rotary_emb, "forward", lambda positions, q, k: (q, k)
    )
    with pytest.raises(AssertionError, match="KV-shared"):
        non_shared(positions, hidden)

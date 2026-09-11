# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Binding never mutates module topology (GEMMA4 audit invariant 1)."""

import pytest
import torch
import torch.nn.functional as F

from vllm.model_executor.dual_precision import (
    BASE_PRECISION_BF16,
    BASE_PRECISION_INT4,
    bind_dual_precision,
    get_active_base_layer,
    get_active_precision,
    get_binding,
    get_dual_precision_state,
    set_analysis_bf16_layers,
)
from vllm.model_executor.dual_precision.binding import (
    BINDING_ATTR,
    OP_LAYER_SUFFIX,
    OVERRIDE_ATTR,
)
from vllm.model_executor.dual_precision.loader import (
    SHADOW_MODULE_NAME,
    attach_shadow_layers,
)

from .fakes import (
    FakeInt4Linear,
    build_int4_model,
    build_wrapped_model,
    module_dict_snapshot,
)

NAMES = [
    "model.layers.0.mlp.gate_up_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.1.self_attn.qkv_proj",
    "model.layers.1.mlp.down_proj",
    "visual.blocks.0.attn.qkv",
]
QUANTIZED = {n for n in NAMES if n.startswith("model.layers")}
HIDDEN = 8


def _attach(bf16_layers="none", module_policy="all"):
    model = build_wrapped_model(NAMES, HIDDEN)
    int4_model = build_int4_model(NAMES, QUANTIZED, HIDDEN)
    static_forward_context: dict = {}
    state = attach_shadow_layers(
        model,
        int4_model,
        bf16_layer_policy=bf16_layers,
        module_policy=module_policy,
        num_layers=2,
        static_forward_context=static_forward_context,
        dtype=torch.float32,
    )
    return model, state, static_forward_context


def test_attach_installs_overrides_and_shadow_store_without_lora_wrapping_shadow():
    model, state, sfc = _attach()

    assert (state.attached, state.policy_bf16, state.fallback) == (4, 0, 1)
    assert state.unwrapped == 0
    assert list(model._modules).count(SHADOW_MODULE_NAME) == 1
    store = model._modules[SHADOW_MODULE_NAME]
    assert store is state.shadow_store and len(store) == 4
    assert all(isinstance(layer, FakeInt4Linear) for layer in store.layers)
    # Shadow linears are bare LinearBase modules, never LoRA wrappers.
    assert all(not hasattr(layer, "base_layer") for layer in store.modules())
    for binding in state.bindings:
        wrapper = model.get_submodule(binding.module_name)
        assert callable(wrapper.base_forward_override)
        assert getattr(wrapper, BINDING_ATTR) is binding
        assert sfc[binding.module_name + OP_LAYER_SUFFIX] is binding
        assert OVERRIDE_ATTR not in wrapper._modules
        assert BINDING_ATTR not in wrapper._modules
    fallback = [b for b in state.bindings if not b.shadow_active]
    assert [b.module_name for b in fallback] == ["visual.blocks.0.attn.qkv"]
    assert fallback[0].int4_or_fallback is fallback[0].bf16


def test_bind_never_mutates_module_dicts_and_flips_active_layer():
    model, state, _ = _attach()
    before = module_dict_snapshot(model)
    wrapper = model.get_submodule("model.layers.1.mlp.down_proj")
    bf16 = wrapper.base_layer
    int4 = get_binding(wrapper).int4_or_fallback
    assert int4 is not bf16

    assert get_active_precision(model) == BASE_PRECISION_BF16
    for precision, expected in (
        (BASE_PRECISION_INT4, int4),
        (BASE_PRECISION_BF16, bf16),
        (BASE_PRECISION_INT4, int4),
    ):
        bind_dual_precision(model, precision, no_compile_layers=None)
        assert get_active_precision(model) == precision
        assert get_active_base_layer(wrapper) is expected
        assert wrapper.base_layer is bf16
        assert module_dict_snapshot(model) == before
    assert get_dual_precision_state(model).bound_state_key == (
        BASE_PRECISION_INT4,
        None,
    )


def test_committed_modules_swap_violates_the_invariant():
    """Regression guard: the 34e66a3 binder replaced ``_modules['base_layer']``
    which is exactly what broke Gemma4 under ``@support_torch_compile``."""
    model, state, _ = _attach()
    before = module_dict_snapshot(model)
    for binding in state.bindings:
        wrapper = model.get_submodule(binding.module_name)
        wrapper._modules["base_layer"] = binding.int4_or_fallback  # the old way

    assert module_dict_snapshot(model) != before


def test_bind_rejects_unknown_precision_and_is_noop_without_state():
    model, _, _ = _attach()
    with pytest.raises(ValueError, match="Unknown base precision"):
        bind_dual_precision(model, "fp8")
    plain = build_wrapped_model(NAMES, HIDDEN)
    bind_dual_precision(plain, BASE_PRECISION_INT4)
    assert get_active_precision(plain) == BASE_PRECISION_BF16


def test_override_runs_selected_base_outside_forward_context_as_bf16():
    """Outside a forward context (tower modules) the override uses BF16 even
    under the INT4 bind; inside one it goes through the custom op (GPU test)."""
    model, _, _ = _attach()
    wrapper = model.get_submodule("model.layers.0.mlp.down_proj")
    x = torch.randn(3, HIDDEN)
    bind_dual_precision(model, BASE_PRECISION_INT4)
    assert torch.equal(wrapper(x), F.linear(x, wrapper.base_layer.weight))


def test_analysis_mask_forces_blocks_to_bf16_under_int4_bind():
    model, state, _ = _attach()
    result = set_analysis_bf16_layers(model, [1, 1, 0], "rescue")
    assert result == {"label": "rescue", "bf16_layers": [0, 1]}
    bind_dual_precision(model, BASE_PRECISION_INT4)
    assert all(b.active is b.bf16 for b in state.bindings)

    set_analysis_bf16_layers(model, [0], "layer0")
    bind_dual_precision(model, BASE_PRECISION_INT4)
    active = {b.module_name: b.active is b.int4_or_fallback for b in state.bindings}
    assert active["model.layers.0.mlp.down_proj"] is False
    assert active["model.layers.1.mlp.down_proj"] is True
    with pytest.raises(ValueError):
        set_analysis_bf16_layers(model, [-1], "bad")


def test_policy_and_mlp_only_reach_bindings():
    model, state, _ = _attach(bf16_layers="0", module_policy="mlp_only")
    assert (state.attached, state.policy_bf16, state.fallback) == (1, 3, 1)
    bind_dual_precision(model, BASE_PRECISION_INT4)
    shadow_active = {b.module_name for b in state.bindings if b.active is not b.bf16}
    assert shadow_active == {"model.layers.1.mlp.down_proj"}


def test_attach_raises_when_nothing_attaches():
    model = build_wrapped_model(NAMES, HIDDEN)
    int4_model = build_int4_model(NAMES, set(), HIDDEN)
    with pytest.raises(RuntimeError, match="no quantized LinearBase"):
        attach_shadow_layers(
            model,
            int4_model,
            bf16_layer_policy="none",
            module_policy="all",
            num_layers=2,
            static_forward_context={},
            dtype=torch.float32,
        )

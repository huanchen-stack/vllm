# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import gc
from dataclasses import fields
from typing import Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase

logger = init_logger(__name__)

BASE_PRECISION_BF16 = "bf16"
BASE_PRECISION_INT4 = "int4"

DUAL_PRECISION_INT4_LAYER_ATTR = "_vllm_dual_precision_int4_layer"
DUAL_PRECISION_BF16_LAYER_ATTR = "_vllm_dual_precision_bf16_base_layer"
DUAL_PRECISION_INT4_FALLBACK_LAYER_ATTR = "_vllm_dual_precision_int4_or_fallback_layer"
DUAL_PRECISION_BOUND_PRECISION_ATTR = "_vllm_dual_precision_bound_precision"
DUAL_PRECISION_MODEL_BOUND_STATE_ATTR = "_vllm_dual_precision_model_bound_state"
DUAL_PRECISION_MODEL_LORA_LAYERS_ATTR = "_vllm_dual_precision_lora_layers"
DUAL_PRECISION_MODEL_LORA_REGISTRY_STATE_ATTR = (
    "_vllm_dual_precision_lora_registry_state"
)
DUAL_PRECISION_MODEL_LORA_REGISTRY_STATS_ATTR = (
    "_vllm_dual_precision_lora_registry_stats"
)
DUAL_PRECISION_SHADOW_MODEL_REF_ATTR = "_vllm_dual_precision_int4_model_ref"
DUAL_PRECISION_SHADOW_MODEL_MODULE = "_vllm_dual_precision_int4_model"

_QUANTIZED_WEIGHT_NAMES = ("qweight", "w13_qweight", "w2_qweight")
_BIND_LOGGED: set[tuple[str, int, int, int]] = set()


class Int4ShadowLayerStore(nn.Module):
    """Owns only the quantized shadow layers needed by dual precision."""

    def __init__(self, named_layers: list[tuple[str, LinearBase]]) -> None:
        super().__init__()
        self.layer_names = [name for name, _ in named_layers]
        self.layers = nn.ModuleList(layer for _, layer in named_layers)


def dual_precision_rollout_enabled(has_lora: bool = True) -> bool:
    return (
        envs.VLLM_DUAL_PRECISION_ROLLOUT
        and envs.ROLLOUT_QLORA
        and has_lora
    )


def select_base_precision(num_reqs: int | None, has_lora: bool) -> str:
    if (
        dual_precision_rollout_enabled(has_lora)
        and num_reqs is not None
        and num_reqs <= envs.VLLM_DUAL_PRECISION_THRESHOLD
    ):
        return BASE_PRECISION_INT4
    return BASE_PRECISION_BF16


def register_dual_precision_lora_layer(
    layer: nn.Module,
    base_layer: LinearBase,
) -> None:
    """Register immutable BF16 and INT4/fallback base choices on a LoRA layer."""
    int4_layer = getattr(base_layer, DUAL_PRECISION_INT4_LAYER_ATTR, base_layer)
    object.__setattr__(layer, DUAL_PRECISION_BF16_LAYER_ATTR, base_layer)
    object.__setattr__(layer, DUAL_PRECISION_INT4_FALLBACK_LAYER_ATTR, int4_layer)
    object.__setattr__(layer, DUAL_PRECISION_BOUND_PRECISION_ATTR, BASE_PRECISION_BF16)


def _get_dual_precision_lora_layers(
    model: nn.Module,
    no_compile_layers: dict[str, Any] | None,
) -> tuple[list[nn.Module], int, int, int, bool]:
    no_compile_layer_count = len(no_compile_layers or {})
    registry_state = (id(no_compile_layers), no_compile_layer_count)
    cached_layers = getattr(model, DUAL_PRECISION_MODEL_LORA_LAYERS_ATTR, None)
    if (
        cached_layers is not None
        and getattr(model, DUAL_PRECISION_MODEL_LORA_REGISTRY_STATE_ATTR, None)
        == registry_state
    ):
        model_module_count, discovered, eligible = getattr(
            model,
            DUAL_PRECISION_MODEL_LORA_REGISTRY_STATS_ATTR,
            (0, 0, len(cached_layers)),
        )
        return cached_layers, model_module_count, discovered, eligible, False

    layers: list[nn.Module] = []
    discovered = 0
    modules: list[nn.Module] = list(model.modules())
    model_module_count = len(modules)
    if no_compile_layers:
        modules.extend(
            module
            for module in no_compile_layers.values()
            if isinstance(module, nn.Module)
        )

    seen: set[int] = set()
    for module in modules:
        module_id = id(module)
        if module_id in seen:
            continue
        seen.add(module_id)
        if not hasattr(module, DUAL_PRECISION_BF16_LAYER_ATTR):
            base_layer = getattr(module, "base_layer", None)
            if isinstance(base_layer, LinearBase):
                register_dual_precision_lora_layer(module, base_layer)
                discovered += 1
            else:
                continue
        layers.append(module)

    object.__setattr__(model, DUAL_PRECISION_MODEL_LORA_LAYERS_ATTR, layers)
    object.__setattr__(
        model, DUAL_PRECISION_MODEL_LORA_REGISTRY_STATE_ATTR, registry_state
    )
    object.__setattr__(
        model,
        DUAL_PRECISION_MODEL_LORA_REGISTRY_STATS_ATTR,
        (model_module_count, discovered, len(layers)),
    )
    return layers, model_module_count, discovered, len(layers), True


def bind_dual_precision_lora_base_layer(
    model: nn.Module,
    base_precision: str,
    no_compile_layers: dict[str, Any] | None = None,
) -> None:
    """Bind LoRA wrappers to a static base layer before capture/execution.

    The forward hot path reads only ``self.base_layer``. INT4 fallback for
    unquantized GPTQ layers is resolved here by binding the BF16 layer into the
    INT4 slot, avoiding per-linear Python branches during replay/capture.
    """
    if not dual_precision_rollout_enabled():
        return

    no_compile_layer_count = len(no_compile_layers or {})
    registry_state = (id(no_compile_layers), no_compile_layer_count)
    bound_state = (base_precision, registry_state)
    if getattr(model, DUAL_PRECISION_MODEL_BOUND_STATE_ATTR, None) == bound_state:
        return

    (
        lora_layers,
        model_module_count,
        discovered,
        eligible,
        refreshed_registry,
    ) = _get_dual_precision_lora_layers(model, no_compile_layers)
    bound = 0
    int4_bound = 0
    for module in lora_layers:
        if getattr(module, DUAL_PRECISION_BOUND_PRECISION_ATTR, None) == base_precision:
            continue
        if base_precision == BASE_PRECISION_INT4:
            active_layer = getattr(module, DUAL_PRECISION_INT4_FALLBACK_LAYER_ATTR)
        else:
            active_layer = getattr(module, DUAL_PRECISION_BF16_LAYER_ATTR)
        module._modules["base_layer"] = active_layer
        object.__setattr__(
            module, DUAL_PRECISION_BOUND_PRECISION_ATTR, base_precision
        )
        bound += 1
        bf16_layer = getattr(module, DUAL_PRECISION_BF16_LAYER_ATTR)
        if active_layer is not bf16_layer:
            int4_bound += 1

    object.__setattr__(model, DUAL_PRECISION_MODEL_BOUND_STATE_ATTR, bound_state)
    log_key = (base_precision, model_module_count, no_compile_layer_count, eligible)
    if log_key in _BIND_LOGGED:
        return
    _BIND_LOGGED.add(log_key)
    logger.info(
        "Dual precision QLoRA base path bound: precision=%s, "
        "model_modules=%d, no_compile_layers=%d, lora_base_layers=%d, "
        "newly_discovered=%d, refreshed_registry=%s, rebound_layers=%d, "
        "int4_shadow_active=%d.",
        base_precision,
        model_module_count,
        no_compile_layer_count,
        eligible,
        discovered,
        refreshed_registry,
        bound,
        int4_bound,
    )


def _clone_init_dataclass(config_obj: Any, **overrides: Any) -> Any:
    config_cls = type(config_obj)
    kwargs = {
        field.name: getattr(config_obj, field.name)
        for field in fields(config_obj)
        if field.init
    }
    kwargs.update(overrides)
    return config_cls(**kwargs)


def _make_int4_vllm_config(vllm_config: VllmConfig) -> VllmConfig:
    int4_model = envs.VLLM_DUAL_PRECISION_INT4_MODEL
    if not int4_model:
        raise ValueError(
            "VLLM_DUAL_PRECISION_ROLLOUT=1 requires "
            "VLLM_DUAL_PRECISION_INT4_MODEL to point at the INT4 checkpoint."
        )

    int4_model_config = _clone_init_dataclass(
        vllm_config.model_config,
        model=int4_model,
        model_weights="",
        hf_config_path=int4_model,
        quantization=None,
    )

    int4_compilation_config = _clone_init_dataclass(
        vllm_config.compilation_config
    )

    vllm_config_cls = type(vllm_config)
    vllm_config_kwargs = {
        field.name: getattr(vllm_config, field.name)
        for field in fields(vllm_config)
        if field.init
    }
    vllm_config_kwargs["model_config"] = int4_model_config
    vllm_config_kwargs["compilation_config"] = int4_compilation_config
    return vllm_config_cls(**vllm_config_kwargs)


def _linear_modules_by_name(model: nn.Module) -> dict[str, LinearBase]:
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, LinearBase)
    }


def _is_quantized_shadow_layer(layer: LinearBase) -> bool:
    return any(hasattr(layer, name) for name in _QUANTIZED_WEIGHT_NAMES)


def _module_tensor_bytes(module: nn.Module) -> int:
    tensors = list(module.parameters(recurse=True)) + list(
        module.buffers(recurse=True)
    )
    seen: set[int] = set()
    total = 0
    for tensor in tensors:
        tensor_id = id(tensor)
        if tensor_id in seen:
            continue
        seen.add(tensor_id)
        total += tensor.numel() * tensor.element_size()
    return total


def _format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1 << 30):.2f} GiB"


def load_and_attach_int4_shadow_model(
    model: nn.Module,
    vllm_config: VllmConfig,
) -> nn.Module:
    if not dual_precision_rollout_enabled(
        has_lora=vllm_config.lora_config is not None
    ):
        return model

    int4_vllm_config = _make_int4_vllm_config(vllm_config)
    logger.info(
        "Loading dual precision INT4 shadow model from %s...",
        int4_vllm_config.model_config.model,
    )
    from vllm.model_executor.model_loader import get_model_loader

    int4_loader = get_model_loader(int4_vllm_config.load_config)
    int4_model = int4_loader.load_model(
        vllm_config=int4_vllm_config,
        model_config=int4_vllm_config.model_config,
    )
    int4_model.eval()
    for param in int4_model.parameters():
        param.requires_grad_(False)

    bf16_linears = _linear_modules_by_name(model)
    int4_linears = _linear_modules_by_name(int4_model)
    quantized_int4_linears = {
        name: layer
        for name, layer in int4_linears.items()
        if _is_quantized_shadow_layer(layer)
    }

    fallback = 0
    attached = 0
    attached_layers: list[tuple[str, LinearBase]] = []
    for name, bf16_layer in bf16_linears.items():
        int4_layer = quantized_int4_linears.get(name)
        if int4_layer is None:
            fallback += 1
            continue
        object.__setattr__(bf16_layer, DUAL_PRECISION_INT4_LAYER_ATTR, int4_layer)
        attached_layers.append((name, int4_layer))
        attached += 1

    if attached == 0:
        raise RuntimeError(
            "Dual precision INT4 shadow model loaded, but no quantized LinearBase "
            "layers were attached."
        )

    shadow_store = Int4ShadowLayerStore(attached_layers)
    shadow_bytes = _module_tensor_bytes(shadow_store)
    object.__setattr__(model, DUAL_PRECISION_SHADOW_MODEL_REF_ATTR, shadow_store)

    logger.info(
        "Loaded %d GPTQ shadow linear layers; attached %d quantized INT4 "
        "layers and left %d BF16 fallback layers.",
        len(int4_linears),
        attached,
        fallback,
    )
    logger.info(
        "Dual precision steady-state INT4 shadow store holds %d layers and %s "
        "of parameters/buffers.",
        attached,
        _format_gib(shadow_bytes),
    )

    del int4_linears
    del quantized_int4_linears
    del attached_layers
    del int4_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


def register_int4_shadow_model(model: nn.Module) -> None:
    shadow_model: nn.Module | None = getattr(
        model, DUAL_PRECISION_SHADOW_MODEL_REF_ATTR, None
    )
    if shadow_model is None:
        return
    modules: dict[str, Any] = model._modules
    if DUAL_PRECISION_SHADOW_MODEL_MODULE not in modules:
        model.add_module(DUAL_PRECISION_SHADOW_MODEL_MODULE, shadow_model)

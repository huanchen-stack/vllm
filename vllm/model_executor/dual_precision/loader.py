# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Load the INT4 shadow checkpoint and attach it to a LoRA-wrapped model.

Residency model: the INT4 model is loaded through the regular model loader
(inside the runner's ``weights`` CuMem pool, so it sleeps and wakes with the
BF16 weights), its GPTQ-packed linears are matched by module name onto the
BF16 model's LoRA wrappers, and only the attached linears are kept alive in an
:class:`Int4ShadowLayerStore` registered as a submodule of the model. The
temporary INT4 model is then dropped.

Supported shadow formats are GPTQ packings only: Intel AutoRound
``auto_round:auto_gptq`` and compressed-tensors ``pack-quantized``. AWQ is
refused: its activation-aware scales live in the norms next to the linears,
so a W4 linear against the BF16 norm is a different model.
"""

from __future__ import annotations

import gc
from dataclasses import fields
from typing import Any

import torch
import torch.nn as nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.dual_precision.binding import (
    DualPrecisionBinding,
    DualPrecisionState,
    find_lora_wrappers,
    install_binding,
    set_dual_precision_state,
)
from vllm.model_executor.dual_precision.policy_layers import (
    format_layer_indices,
    is_quantized_shadow_layer,
    resolve_bf16_layer_indices,
    should_attach_int4_shadow,
    transformer_layer_index,
)
from vllm.model_executor.dual_precision.validation import (
    MAX_LIFECYCLE_PROBES,
    LifecycleProbe,
    ShadowValidation,
    compare_shadow_numerics,
    log_shadow_validation,
    record_lifecycle_probe,
)
from vllm.model_executor.layers.linear import LinearBase

logger = init_logger(__name__)

SHADOW_MODULE_NAME = "_vllm_dual_precision_int4_model"
"""Attribute under which the shadow store hangs on the model.

Contract with verl's ``_hide_dual_precision_shadow_model`` (weight sync pops
this submodule while it re-runs ``process_weights_after_loading``).
"""

_GPTQ_RESOLVED_METHODS = frozenset(
    {"gptq", "gptq_marlin", "auto_gptq", "inc", "compressed-tensors"}
)
_AWQ_RESOLVED_METHODS = frozenset({"awq", "awq_marlin"})


class Int4ShadowLayerStore(nn.Module):
    """Strongly owns the attached INT4 linears so they share the model's
    lifecycle (CuMem ``weights`` pool, sleep level 1 offload/restore)."""

    def __init__(self, named_layers: list[tuple[str, LinearBase]]) -> None:
        super().__init__()
        self.layer_names = [name for name, _ in named_layers]
        self.layers = nn.ModuleList(layer for _, layer in named_layers)

    def __len__(self) -> int:
        return len(self.layers)


def dual_precision_rollout_enabled() -> bool:
    """Feature gate: ``VLLM_DUAL_PRECISION_ROLLOUT=1``."""
    return bool(envs.VLLM_DUAL_PRECISION_ROLLOUT)


def check_dual_precision_model_runner(vllm_config: VllmConfig) -> None:
    """Refuse engine configurations dual precision does not support.

    Only the V1 ``GPUModelRunner`` attaches the shadow, binds the base
    precision before every forward and captures precision-keyed CUDA graphs.
    The V2 runner (``VLLM_USE_V2_MODEL_RUNNER=1``, or the default for
    unquantized ``Qwen3ForCausalLM``) would silently serve BF16 for every
    step; fail at worker init instead. Speculative decoding, the KV-sharing
    fast-prefill path and data parallelism dispatch extra forwards with
    descriptors that carry no precision (implicitly BF16) or choose the
    precision per DP rank, so they are refused as well.

    With the feature off, a configured ``VLLM_DUAL_PRECISION_POLICY`` on a
    LoRA-enabled engine is refused too: the scheduler would publish ``int4``
    with no shadow attached and every post-switch step would run eager.
    Without a LoRA config nothing could ever bind, so the scheduler-only
    smokes (C4, C7) may set a policy alone.
    """
    if not dual_precision_rollout_enabled():
        if envs.VLLM_DUAL_PRECISION_POLICY and vllm_config.lora_config is not None:
            raise NotImplementedError(
                "VLLM_DUAL_PRECISION_POLICY requires VLLM_DUAL_PRECISION_ROLLOUT=1 "
                "on a LoRA-enabled engine: without the INT4 shadow the "
                "scheduler's int4 steps would have no graph and run eager."
            )
        return
    if vllm_config.use_v2_model_runner:
        raise NotImplementedError(
            "VLLM_DUAL_PRECISION_ROLLOUT=1 is not supported with the V2 model "
            "runner: the INT4 shadow is attached, bound and graph-captured by "
            "the V1 GPUModelRunner only. Set VLLM_USE_V2_MODEL_RUNNER=0."
        )
    if vllm_config.speculative_config is not None:
        raise NotImplementedError(
            "VLLM_DUAL_PRECISION_ROLLOUT=1 is not supported with speculative "
            "decoding: draft forwards carry no base precision."
        )
    if vllm_config.cache_config.kv_sharing_fast_prefill:
        raise NotImplementedError(
            "VLLM_DUAL_PRECISION_ROLLOUT=1 is not supported with "
            "kv_sharing_fast_prefill: the decoder-portion dispatch carries no "
            "base precision."
        )
    if vllm_config.parallel_config.data_parallel_size > 1:
        raise NotImplementedError(
            "VLLM_DUAL_PRECISION_ROLLOUT=1 is not supported with data "
            "parallelism: the precision is chosen per DP rank's scheduler and "
            "is not synchronised across ranks."
        )


# --------------------------------------------------------------------------- #
# Config cloning and format validation                                         #
# --------------------------------------------------------------------------- #


def clone_init_dataclass(config_obj: Any, **overrides: Any) -> Any:
    """Re-construct ``config_obj`` from its init fields with overrides."""
    config_cls = type(config_obj)
    kwargs = {
        field.name: getattr(config_obj, field.name)
        for field in fields(config_obj)
        if field.init
    }
    kwargs.update(overrides)
    return config_cls(**kwargs)


def validate_shadow_quantization(
    hf_quant_config: dict[str, Any] | None, resolved_method: str | None
) -> str:
    """Accept GPTQ packings, reject everything else with a clear message.

    Returns a short label of the accepted format for logging.
    """
    if not hf_quant_config:
        raise ValueError(
            "VLLM_DUAL_PRECISION_INT4_MODEL must point at a quantized checkpoint "
            "(no quantization_config found)."
        )
    quant_method = str(hf_quant_config.get("quant_method", "")).lower()
    resolved = (resolved_method or "").lower()

    if quant_method in _AWQ_RESOLVED_METHODS or resolved in _AWQ_RESOLVED_METHODS:
        raise ValueError(
            "Dual precision shadow checkpoints must be GPTQ-packed; AWQ "
            f"(quant_method={quant_method!r}, resolved={resolved!r}) is not "
            "supported because its activation scales are folded into the "
            "norms of the quantized checkpoint."
        )
    if quant_method == "auto-round":
        packing = str(hf_quant_config.get("packing_format", "auto_round:auto_gptq"))
        backend = str(
            hf_quant_config.get("backend", hf_quant_config.get("vllm_backend", "auto"))
        )
        if "awq" in packing.lower() or "awq" in backend.lower():
            raise ValueError(
                "Dual precision shadow checkpoints must be GPTQ-packed; "
                f"AutoRound packing_format={packing!r} backend={backend!r} is AWQ."
            )
        return f"auto-round:{packing}"
    if quant_method == "compressed-tensors":
        fmt = str(hf_quant_config.get("format", "")).lower()
        groups = hf_quant_config.get("config_groups") or {}
        group_formats = {
            str(group.get("format", "")).lower()
            for group in groups.values()
            if isinstance(group, dict)
        }
        if fmt != "pack-quantized" and "pack-quantized" not in group_formats:
            raise ValueError(
                "Dual precision shadow checkpoints must be compressed-tensors "
                f"pack-quantized (GPTQ); got format={fmt!r}."
            )
        return "compressed-tensors:pack-quantized"
    if quant_method in ("gptq", "gptq_marlin") or resolved in _GPTQ_RESOLVED_METHODS:
        return f"gptq:{resolved or quant_method}"
    raise ValueError(
        "Dual precision shadow checkpoints must be GPTQ-packed (AutoRound "
        "auto_round:auto_gptq or compressed-tensors pack-quantized); got "
        f"quant_method={quant_method!r} (resolved {resolved!r})."
    )


def make_int4_vllm_config(vllm_config: VllmConfig, int4_model: str) -> VllmConfig:
    """Clone ``vllm_config`` so it loads ``int4_model`` with auto-detected
    quantization and its own compilation config (own static forward context).
    """
    if not int4_model:
        raise ValueError(
            "VLLM_DUAL_PRECISION_ROLLOUT=1 requires "
            "VLLM_DUAL_PRECISION_INT4_MODEL to point at the INT4 checkpoint."
        )

    int4_model_config = clone_init_dataclass(
        vllm_config.model_config,
        model=int4_model,
        model_weights="",
        hf_config_path=int4_model,
        quantization=None,
    )
    validate_shadow_quantization(
        int4_model_config.model_arch_config.quantization_config,
        int4_model_config.quantization,
    )
    int4_compilation_config = clone_init_dataclass(vllm_config.compilation_config)
    return clone_init_dataclass(
        vllm_config,
        model_config=int4_model_config,
        compilation_config=int4_compilation_config,
    )


def load_int4_shadow_model(int4_vllm_config: VllmConfig) -> nn.Module:
    from vllm.model_executor.model_loader import get_model_loader

    logger.info(
        "Loading dual precision INT4 shadow model from %s...",
        int4_vllm_config.model_config.model,
    )
    loader = get_model_loader(int4_vllm_config.load_config)
    int4_model = loader.load_model(
        vllm_config=int4_vllm_config, model_config=int4_vllm_config.model_config
    )
    int4_model.eval()
    for param in int4_model.parameters():
        param.requires_grad_(False)
    return int4_model


# --------------------------------------------------------------------------- #
# Attach                                                                       #
# --------------------------------------------------------------------------- #


def module_tensor_bytes(module: nn.Module) -> int:
    seen: set[int] = set()
    total = 0
    for tensor in list(module.parameters()) + list(module.buffers()):
        if id(tensor) in seen:
            continue
        seen.add(id(tensor))
        total += tensor.numel() * tensor.element_size()
    return total


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1 << 30):.2f} GiB"


def register_shadow_store(model: nn.Module, store: Int4ShadowLayerStore) -> None:
    """Hang the store on the model (after LoRA wrapping, so the LoRA manager
    never sees the shadow linears) under :data:`SHADOW_MODULE_NAME`."""
    if SHADOW_MODULE_NAME in model._modules:
        raise RuntimeError("Dual precision shadow store is already registered.")
    model.add_module(SHADOW_MODULE_NAME, store)


def attach_shadow_layers(
    model: nn.Module,
    int4_model: nn.Module,
    *,
    bf16_layer_policy: str,
    module_policy: str,
    num_layers: int,
    static_forward_context: dict[str, Any],
    dtype: torch.dtype,
    validate_shadow: bool = False,
    validate_lifecycle: bool = False,
) -> DualPrecisionState:
    """Match, bind and store. Pure of env access; the runner entry point
    :func:`attach_dual_precision` supplies the knobs."""
    bf16_layer_indices = resolve_bf16_layer_indices(bf16_layer_policy, num_layers)
    logger.info(
        "Dual precision BF16 layer policy %r resolved to transformer blocks %s of %d.",
        bf16_layer_policy,
        format_layer_indices(bf16_layer_indices),
        num_layers,
    )
    logger.info("Dual precision INT4 module policy: %s.", module_policy)

    int4_linears = {
        name: module
        for name, module in int4_model.named_modules()
        if isinstance(module, LinearBase)
    }
    quantized = {
        name: layer
        for name, layer in int4_linears.items()
        if is_quantized_shadow_layer(layer)
    }
    wrappers = find_lora_wrappers(model)
    wrapped_base_names = {f"{name}.base_layer" for name in wrappers}

    # Counts follow the archived log line and cover every BF16 ``LinearBase``
    # (wrapped or not): ``fallback`` has no quantized peer, ``policy_bf16`` is
    # excluded by the layer/module policy, ``attached`` joins the store, and
    # ``unwrapped`` would attach but has no LoRA wrapper to switch through.
    attached = policy_bf16 = fallback = unwrapped = 0
    bare_linears = [
        name
        for name, module in model.named_modules()
        if isinstance(module, LinearBase) and name not in wrapped_base_names
    ]
    for name in list(wrappers) + bare_linears:
        if name not in quantized:
            fallback += 1
        elif not should_attach_int4_shadow(name, bf16_layer_indices, module_policy):
            policy_bf16 += 1
        elif name in wrappers:
            attached += 1
        else:
            unwrapped += 1

    if attached == 0:
        # Checked before anything is installed so a failed attach leaves the
        # model exactly as it was (no overrides, no bindings, no store).
        raise RuntimeError(
            "Dual precision INT4 shadow model loaded, but no quantized LinearBase "
            "layers were attached to a LoRA wrapper."
        )

    bindings: list[DualPrecisionBinding] = []
    attached_layers: list[tuple[str, LinearBase]] = []
    validations: list[ShadowValidation] = []
    probes: list[LifecycleProbe] = []
    for name, wrapper in wrappers.items():
        layer_index = transformer_layer_index(name)
        int4_layer = quantized.get(name)
        if int4_layer is not None and should_attach_int4_shadow(
            name, bf16_layer_indices, module_policy
        ):
            attached_layers.append((name, int4_layer))
            if validate_shadow:
                validations.append(
                    compare_shadow_numerics(name, wrapper.base_layer, int4_layer, dtype)
                )
            if validate_lifecycle and len(probes) < MAX_LIFECYCLE_PROBES:
                probes.append(record_lifecycle_probe(name, int4_layer, dtype))
        else:
            int4_layer = None
        bindings.append(
            install_binding(
                wrapper, name, int4_layer, layer_index, static_forward_context
            )
        )

    store = Int4ShadowLayerStore(attached_layers)
    state = DualPrecisionState(
        shadow_store=store,
        bindings=bindings,
        num_shadow_linears=len(int4_linears),
        attached=attached,
        policy_bf16=policy_bf16,
        fallback=fallback,
        unwrapped=unwrapped,
        shadow_bytes=module_tensor_bytes(store),
        lifecycle_probes=probes,
    )
    set_dual_precision_state(model, state)
    register_shadow_store(model, store)

    logger.info(
        "Loaded %d GPTQ shadow linear layers; attached %d quantized INT4 "
        "layers, kept %d quantized layers in BF16 by policy, and left %d "
        "layers in BF16 because no INT4 shadow was available.",
        state.num_shadow_linears,
        attached,
        policy_bf16,
        fallback,
    )
    if unwrapped:
        logger.warning(
            "Dual precision: %d quantized linears have no LoRA wrapper and stay BF16.",
            unwrapped,
        )
    logger.info(
        "Dual precision steady-state INT4 shadow store holds %d layers and %s "
        "of parameters/buffers.",
        attached,
        format_gib(state.shadow_bytes),
    )
    log_shadow_validation(validations)
    return state


def attach_dual_precision(
    model: nn.Module, vllm_config: VllmConfig
) -> DualPrecisionState | None:
    """Runner entry point; call once, after ``load_lora_model``.

    Returns ``None`` (and changes nothing) when dual precision is off or the
    engine has no LoRA config.
    """
    if not dual_precision_rollout_enabled():
        return None
    if vllm_config.lora_config is None:
        logger.warning(
            "VLLM_DUAL_PRECISION_ROLLOUT=1 but LoRA is not enabled; the INT4 "
            "shadow model is not loaded."
        )
        return None

    int4_vllm_config = make_int4_vllm_config(
        vllm_config, envs.VLLM_DUAL_PRECISION_INT4_MODEL
    )
    int4_model = load_int4_shadow_model(int4_vllm_config)
    try:
        state = attach_shadow_layers(
            model,
            int4_model,
            bf16_layer_policy=envs.VLLM_DUAL_PRECISION_BF16_LAYERS,
            module_policy=envs.VLLM_DUAL_PRECISION_INT4_MODULES,
            num_layers=vllm_config.model_config.get_total_num_hidden_layers(),
            static_forward_context=vllm_config.compilation_config.static_forward_context,
            dtype=vllm_config.model_config.dtype,
            validate_shadow=envs.VLLM_DUAL_PRECISION_VALIDATE_SHADOW,
            validate_lifecycle=envs.VLLM_DUAL_PRECISION_VALIDATE_LIFECYCLE,
        )
    finally:
        # Only the attached linears survive, owned by the shadow store.
        del int4_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return state

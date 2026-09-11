# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-forward base-precision selection for LoRA wrappers.

Topology invariant (GEMMA4_DUAL_PRECISION_AUDIT, invariant 1): binding never
mutates ``_modules``, ``_parameters`` or ``_buffers`` of any module. The
compiled model graph and the captured CUDA graphs see one stable opaque op,
``torch.ops.vllm.dual_precision_base_linear``; the op body reads a plain
Python selection (:class:`DualPrecisionBinding.active`) that lives in the
wrapper's ``__dict__`` and picks the BF16 base or its INT4 shadow at run time.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.model_executor.dual_precision.loader import Int4ShadowLayerStore
    from vllm.model_executor.dual_precision.validation import LifecycleProbe

logger = init_logger(__name__)

BASE_PRECISION_BF16 = "bf16"
BASE_PRECISION_INT4 = "int4"
BASE_PRECISIONS: tuple[str, ...] = (BASE_PRECISION_BF16, BASE_PRECISION_INT4)

BINDING_ATTR = "_vllm_dual_precision_binding"
"""``__dict__`` slot on a LoRA wrapper holding its :class:`DualPrecisionBinding`."""
STATE_ATTR = "_vllm_dual_precision_state"
"""``__dict__`` slot on the model holding its :class:`DualPrecisionState`."""
OP_LAYER_SUFFIX = ".dual_precision_base_linear"
"""Suffix appended to ``base_layer.prefix`` to key the wrapper in
``compilation_config.static_forward_context`` (mirrors ``.lora_linear_async``)."""
OVERRIDE_SETTER = "set_base_forward_override"
OVERRIDE_ATTR = "base_forward_override"


@dataclass
class DualPrecisionBinding:
    """Immutable base choices plus the mutable selection for one wrapper."""

    layer_name: str
    """Key in ``static_forward_context`` used by the custom op."""
    module_name: str
    """Name of the LoRA wrapper in ``model.named_modules()``."""
    bf16: LinearBase
    int4_or_fallback: LinearBase
    """INT4 shadow linear, or ``bf16`` again when no shadow was attached."""
    layer_index: int | None
    active: LinearBase
    bound_precision: str = BASE_PRECISION_BF16

    @property
    def shadow_active(self) -> bool:
        return self.int4_or_fallback is not self.bf16


@dataclass
class DualPrecisionState:
    """Everything dual precision knows about one model; hangs off the model."""

    shadow_store: Int4ShadowLayerStore
    bindings: list[DualPrecisionBinding]
    num_shadow_linears: int
    attached: int
    policy_bf16: int
    fallback: int
    unwrapped: int
    shadow_bytes: int
    active_precision: str = BASE_PRECISION_BF16
    analysis_bf16_layers: frozenset[int] | None = None
    analysis_label: str | None = None
    lifecycle_probes: list[LifecycleProbe] = field(default_factory=list)
    lifecycle_validated: bool = False
    pending_lifecycle_events: list[str] = field(default_factory=list)
    """Events since the last INT4 bind (``mark_lifecycle_event``); the next
    INT4 bind re-runs the lifecycle probes and clears the list."""
    probe_dtype: torch.dtype | None = None
    shadow_load_format: str = "auto"
    sanity_probe_pending: bool = False
    """The attach-time sanity probe was deferred (dummy base weights): run it
    at the first INT4 bind after a weight-load event."""
    shadow_validation_pending: bool = False
    """``VLLM_DUAL_PRECISION_VALIDATE_SHADOW`` deferred the same way."""
    bound_state_key: tuple[Any, ...] | None = None
    logged_bind_keys: set[tuple[Any, ...]] = field(default_factory=set)


LIFECYCLE_WEIGHT_EVENTS: frozenset[str] = frozenset(
    {"load_weights", "reload_weights", "update_weights"}
)
"""Lifecycle events after which the base weights may differ from attach time."""


# --------------------------------------------------------------------------- #
# Custom op                                                                    #
# --------------------------------------------------------------------------- #


def dual_precision_base_linear(
    layer_name: str,
    output_size: int,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the currently selected BF16/INT4 base outside the compiled graph."""
    binding: DualPrecisionBinding = get_forward_context().no_compile_layers[layer_name]
    active = binding.active
    return active.quant_method.apply(active, x, bias)


def dual_precision_base_linear_fake(
    layer_name: str,
    output_size: int,
    x: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty((*x.shape[:-1], output_size), device=x.device, dtype=x.dtype)


direct_register_custom_op(
    op_name="dual_precision_base_linear",
    op_func=dual_precision_base_linear,
    fake_impl=dual_precision_base_linear_fake,
)


def make_base_forward_override(
    binding: DualPrecisionBinding, output_size: int
) -> Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor]:
    """Closure installed as the wrapper's ``base_forward_override``.

    Inside a forward context the base GEMM goes through the opaque op so the
    precision choice is invisible to ``torch.compile``. Outside one (e.g.
    multimodal tower modules run eagerly) the BF16 base is used directly, as
    the plain sync path would.
    """
    layer_name = binding.layer_name
    bf16 = binding.bf16

    def dual_precision_base_forward(
        x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        if is_forward_context_available():
            return torch.ops.vllm.dual_precision_base_linear(
                layer_name, output_size, x, bias
            )
        return bf16.quant_method.apply(bf16, x, bias)

    return dual_precision_base_forward


# --------------------------------------------------------------------------- #
# Wrapper discovery and override installation                                  #
# --------------------------------------------------------------------------- #


def find_lora_wrappers(model: nn.Module) -> dict[str, nn.Module]:
    """LoRA wrappers keyed by module name (modules with a ``LinearBase`` base)."""
    return {
        name: module
        for name, module in model.named_modules()
        if isinstance(getattr(module, "base_layer", None), LinearBase)
    }


def base_output_size(wrapper: nn.Module) -> int:
    slices = getattr(wrapper, "output_slices", None)
    if slices:
        return int(sum(slices))
    return int(wrapper.base_layer.output_size_per_partition)


def install_binding(
    wrapper: nn.Module,
    module_name: str,
    int4_layer: LinearBase | None,
    layer_index: int | None,
    static_forward_context: dict[str, Any],
) -> DualPrecisionBinding:
    """Bind ``wrapper`` to its BF16 base and optional INT4 shadow.

    Registers the binding under ``base_layer.prefix + OP_LAYER_SUFFIX`` in the
    static forward context and installs the override closure on the wrapper.
    Only ``__dict__`` entries are added; the module tree is untouched.
    """
    base_layer: LinearBase = wrapper.base_layer
    prefix = getattr(base_layer, "prefix", "") or module_name
    layer_name = prefix + OP_LAYER_SUFFIX
    if layer_name in static_forward_context:
        raise ValueError(f"Duplicate dual precision layer name: {layer_name}")
    binding = DualPrecisionBinding(
        layer_name=layer_name,
        module_name=module_name,
        bf16=base_layer,
        int4_or_fallback=int4_layer if int4_layer is not None else base_layer,
        layer_index=layer_index,
        active=base_layer,
    )
    static_forward_context[layer_name] = binding
    override = make_base_forward_override(binding, base_output_size(wrapper))
    setter = getattr(wrapper, OVERRIDE_SETTER, None)
    if callable(setter):
        setter(override)
    else:
        object.__setattr__(wrapper, OVERRIDE_ATTR, override)
    object.__setattr__(wrapper, BINDING_ATTR, binding)
    return binding


def get_binding(wrapper: nn.Module) -> DualPrecisionBinding | None:
    return getattr(wrapper, BINDING_ATTR, None)


def get_active_base_layer(wrapper: nn.Module) -> LinearBase:
    """Selected base of ``wrapper`` without touching module topology."""
    binding = get_binding(wrapper)
    if binding is None:
        base_layer = getattr(wrapper, "base_layer", None)
        if not isinstance(base_layer, LinearBase):
            raise TypeError(
                f"LoRA wrapper has no LinearBase base layer: {type(wrapper)!r}"
            )
        return base_layer
    return binding.active


# --------------------------------------------------------------------------- #
# Model-level state and bind                                                   #
# --------------------------------------------------------------------------- #


def set_dual_precision_state(model: nn.Module, state: DualPrecisionState) -> None:
    object.__setattr__(model, STATE_ATTR, state)


def get_dual_precision_state(model: nn.Module) -> DualPrecisionState | None:
    return getattr(model, STATE_ATTR, None)


def get_active_precision(model: nn.Module) -> str:
    state = get_dual_precision_state(model)
    return BASE_PRECISION_BF16 if state is None else state.active_precision


def mark_lifecycle_event(model: nn.Module, kind: str) -> None:
    """Arm one re-validation at the next INT4 bind.

    Called by the worker (``sleep`` / ``wake_up``), the runner's weight
    reload and the model's ``load_weights`` (wrapped at attach). No-op when
    dual precision is not attached, so vanilla paths pay nothing. Repeated
    events of one kind between two INT4 binds collapse into one entry.
    """
    state = get_dual_precision_state(model)
    if state is None:
        return
    if kind not in state.pending_lifecycle_events:
        state.pending_lifecycle_events.append(kind)


def _run_int4_bind_validations(state: DualPrecisionState) -> None:
    """Lifecycle probes (baseline at the first INT4 bind, again after every
    marked event) and the deferred sanity probe, before the wrappers flip."""
    from vllm.model_executor.dual_precision.validation import (
        compare_shadow_numerics,
        log_shadow_validation,
        sanity_probe_shadow,
        validate_shadow_lifecycle,
    )

    events = list(state.pending_lifecycle_events)
    state.pending_lifecycle_events.clear()
    if not state.lifecycle_validated:
        validate_shadow_lifecycle(state)
    elif events:
        validate_shadow_lifecycle(state, after=events)
    weight_events = [kind for kind in events if kind in LIFECYCLE_WEIGHT_EVENTS]
    if not weight_events or state.probe_dtype is None:
        return
    when = "at the first INT4 bind after " + ",".join(weight_events)
    shadow_bindings = [b for b in state.bindings if b.shadow_active]
    if state.sanity_probe_pending:
        state.sanity_probe_pending = False
        if shadow_bindings:
            first = shadow_bindings[0]
            sanity_probe_shadow(
                first.module_name,
                first.bf16,
                first.int4_or_fallback,
                state.probe_dtype,
                shadow_load_format=state.shadow_load_format,
                when=when,
                deferred=True,
            )
    if state.shadow_validation_pending:
        state.shadow_validation_pending = False
        log_shadow_validation(
            [
                compare_shadow_numerics(
                    b.module_name, b.bf16, b.int4_or_fallback, state.probe_dtype
                )
                for b in shadow_bindings
            ],
            when=when,
        )


def bind_dual_precision(
    model: nn.Module,
    precision: str,
    no_compile_layers: dict[str, Any] | None = None,
) -> None:
    """Select ``precision`` for every bound wrapper before capture/execution.

    Idempotent per (precision, analysis mask). Never mutates ``_modules``,
    ``_parameters`` or ``_buffers``. ``no_compile_layers`` is accepted for
    call-site symmetry with the runner (which passes its static forward
    context) and is not needed: every binding was registered at attach time.
    """
    state = get_dual_precision_state(model)
    if state is None:
        return
    if precision not in BASE_PRECISIONS:
        raise ValueError(
            f"Unknown base precision {precision!r}; expected one of {BASE_PRECISIONS}."
        )

    # Validations come before the idempotence check: an INT4 -> INT4 rebind
    # across a wake-up or weight sync must still re-run the probes.
    if precision == BASE_PRECISION_INT4 and (
        not state.lifecycle_validated
        or state.pending_lifecycle_events
        or state.sanity_probe_pending
        or state.shadow_validation_pending
    ):
        _run_int4_bind_validations(state)

    analysis_layers = state.analysis_bf16_layers
    analysis_key = None if analysis_layers is None else tuple(sorted(analysis_layers))
    bound_state_key = (precision, analysis_key)
    if state.bound_state_key == bound_state_key:
        return

    rebound = 0
    shadow_active = 0
    for binding in state.bindings:
        force_bf16 = (
            analysis_layers is not None and binding.layer_index in analysis_layers
        )
        if precision == BASE_PRECISION_INT4 and not force_bf16:
            active = binding.int4_or_fallback
        else:
            active = binding.bf16
        if binding.active is not active or binding.bound_precision != precision:
            rebound += 1
        binding.active = active
        binding.bound_precision = precision
        if active is not binding.bf16:
            shadow_active += 1

    state.active_precision = precision
    state.bound_state_key = bound_state_key
    log_key = (precision, analysis_key)
    if log_key in state.logged_bind_keys:
        return
    state.logged_bind_keys.add(log_key)
    # Log-line contract (WARNING, as archived): verl's validate_rollout_run.py
    # reads ``lora_base_layers=`` and ``precision=int4 ... int4_shadow_active=``
    # from a run that sets VLLM_LOGGING_LEVEL=WARN.
    logger.warning(
        "Dual precision QLoRA base path bound: precision=%s, "
        "lora_base_layers=%d, rebound_layers=%d, int4_shadow_active=%d, "
        "analysis_bf16_layers=%s.",
        precision,
        len(state.bindings),
        rebound,
        shadow_active,
        state.analysis_label if analysis_layers is not None else "off",
    )


def set_analysis_bf16_layers(
    model: nn.Module,
    layer_indices: list[int] | tuple[int, ...],
    label: str,
) -> dict[str, Any]:
    """Eager-only diagnostic: force these blocks to BF16 under the INT4 bind.

    Used by offline layer-sensitivity studies through ``llm.apply_model``.
    Not for CUDA-graph runs: the mask changes the kernels a graph would
    capture, so callers must run eagerly. The next :func:`bind_dual_precision`
    applies the mask.
    """
    indices = frozenset(int(index) for index in layer_indices)
    if any(index < 0 for index in indices):
        raise ValueError(f"Layer indices must be non-negative, got {indices}.")
    state = get_dual_precision_state(model)
    if state is None:
        raise RuntimeError("Dual precision is not attached to this model.")
    state.analysis_bf16_layers = indices
    state.analysis_label = label
    state.bound_state_key = None
    return {"label": label, "bf16_layers": sorted(indices)}

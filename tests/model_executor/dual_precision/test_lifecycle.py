# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Always-on shadow sanity probe and event-armed lifecycle re-validation
(integration defects 1 and 4), on CPU fakes."""

import logging

import pytest
import torch

from vllm.model_executor.dual_precision import (
    BASE_PRECISION_BF16,
    BASE_PRECISION_INT4,
    bind_dual_precision,
    get_binding,
    get_dual_precision_state,
)
from vllm.model_executor.dual_precision.binding import mark_lifecycle_event
from vllm.model_executor.dual_precision.loader import (
    SHADOW_MODULE_NAME,
    attach_shadow_layers,
    wrap_load_weights_for_lifecycle,
)
from vllm.model_executor.dual_precision.validation import SANITY_MIN_COSINE

from .fakes import build_int4_model, build_wrapped_model, module_dict_snapshot

NAMES = [
    "model.layers.0.mlp.gate_up_proj",
    "model.layers.0.mlp.down_proj",
    "model.layers.1.mlp.down_proj",
]
HIDDEN = 64
VALIDATION_LOGGER = "vllm.model_executor.dual_precision.validation"
BINDING_LOGGER = "vllm.model_executor.dual_precision.binding"


def _attach(int4_model=None, **kwargs):
    model = build_wrapped_model(NAMES, HIDDEN)
    if int4_model is None:
        int4_model = build_int4_model(NAMES, set(NAMES), HIDDEN)
    sfc: dict = {}
    state = attach_shadow_layers(
        model,
        int4_model,
        bf16_layer_policy="none",
        module_policy="all",
        num_layers=2,
        static_forward_context=sfc,
        dtype=torch.float32,
        **kwargs,
    )
    return model, state, sfc


def _lifecycle_lines(caplog) -> list[str]:
    return [
        rec.getMessage()
        for rec in caplog.records
        if rec.name == VALIDATION_LOGGER and "lifecycle validation" in rec.getMessage()
    ]


# --------------------------------------------------------------- sanity probe


def test_random_shadow_is_refused_at_attach_and_leaves_model_untouched():
    model = build_wrapped_model(NAMES, HIDDEN)
    random_shadow = build_int4_model(NAMES, set(NAMES), HIDDEN, base_seed=None)
    before = module_dict_snapshot(model)
    sfc: dict = {}
    with pytest.raises(RuntimeError) as excinfo:
        attach_shadow_layers(
            model,
            random_shadow,
            bf16_layer_policy="none",
            module_policy="all",
            num_layers=2,
            static_forward_context=sfc,
            dtype=torch.float32,
            shadow_load_format="auto",
        )
    message = str(excinfo.value)
    assert "sanity probe failed at attach" in message
    assert NAMES[0] in message and "load_format=auto" in message
    assert f"> {SANITY_MIN_COSINE}" in message
    assert module_dict_snapshot(model) == before
    assert sfc == {} and get_dual_precision_state(model) is None
    assert SHADOW_MODULE_NAME not in model._modules
    assert all(model.get_submodule(n).base_forward_override is None for n in NAMES)


def test_sanity_probe_passes_for_a_correlated_shadow_and_is_not_deferred():
    _, state, _ = _attach()
    assert state.sanity_probe_pending is False
    assert state.shadow_load_format == "auto"


def test_dummy_engine_defers_sanity_probe_to_first_int4_bind_after_load_weights(
    caplog,
):
    """verl's flow: dummy base -> graph capture (INT4 bind, BF16 is noise so
    no probe) -> trainer syncs base weights -> rollout's INT4 bind probes."""
    random_shadow = build_int4_model(NAMES, set(NAMES), HIDDEN, base_seed=None)
    with caplog.at_level(logging.WARNING, logger="vllm"):
        model, state, _ = _attach(random_shadow, engine_load_format="dummy")
    assert state.sanity_probe_pending is True
    assert any("deferred" in rec.getMessage() for rec in caplog.records)

    # Capture-time INT4 bind: no weight event yet, nothing to compare against.
    bind_dual_precision(model, BASE_PRECISION_INT4)
    assert state.sanity_probe_pending is True
    bind_dual_precision(model, BASE_PRECISION_BF16)
    mark_lifecycle_event(model, "wake_up")  # not a weight event
    bind_dual_precision(model, BASE_PRECISION_INT4)
    assert state.sanity_probe_pending is True

    mark_lifecycle_event(model, "load_weights")
    with pytest.raises(RuntimeError, match="after load_weights"):
        bind_dual_precision(model, BASE_PRECISION_INT4)
    assert state.sanity_probe_pending is False


def test_dummy_engine_deferred_probe_passes_once_base_weights_are_real(caplog):
    model, state, _ = _attach(engine_load_format="dummy")
    bind_dual_precision(model, BASE_PRECISION_INT4)
    assert state.sanity_probe_pending
    mark_lifecycle_event(model, "load_weights")
    with caplog.at_level(logging.INFO, logger=VALIDATION_LOGGER):
        bind_dual_precision(model, BASE_PRECISION_INT4)
    assert not state.sanity_probe_pending
    assert any(
        "sanity probe passed at the first INT4 bind after load_weights"
        in r.getMessage()
        for r in caplog.records
    )
    # Idempotent: a second weight event does not re-run the sanity probe.
    caplog.clear()
    mark_lifecycle_event(model, "load_weights")
    with caplog.at_level(logging.INFO, logger=VALIDATION_LOGGER):
        bind_dual_precision(model, BASE_PRECISION_INT4)
    assert not any("sanity probe" in r.getMessage() for r in caplog.records)


# ------------------------------------------------------ lifecycle re-validation


def test_lifecycle_probes_rerun_only_after_a_marked_event(caplog):
    model, state, _ = _attach(validate_lifecycle=True)
    assert len(state.lifecycle_probes) == 3
    with caplog.at_level(logging.WARNING, logger=VALIDATION_LOGGER):
        bind_dual_precision(model, BASE_PRECISION_INT4)
        baseline = _lifecycle_lines(caplog)
        assert len(baseline) == 1 and "at first INT4 bind:" in baseline[0]
        assert baseline[0].count("exact=True") == 3

        # Without an event: BF16 -> INT4 -> BF16 -> INT4 never re-validates.
        bind_dual_precision(model, BASE_PRECISION_BF16)
        bind_dual_precision(model, BASE_PRECISION_INT4)
        bind_dual_precision(model, BASE_PRECISION_INT4)
        assert len(_lifecycle_lines(caplog)) == 1

        # After wake_up the next INT4 bind re-validates, exactly once, even
        # though the bound precision is unchanged (INT4 -> INT4).
        mark_lifecycle_event(model, "wake_up")
        assert state.pending_lifecycle_events == ["wake_up"]
        bind_dual_precision(model, BASE_PRECISION_INT4)
        lines = _lifecycle_lines(caplog)
        assert len(lines) == 2 and "after wake_up:" in lines[1]
        assert state.pending_lifecycle_events == []
        bind_dual_precision(model, BASE_PRECISION_INT4)
        assert len(_lifecycle_lines(caplog)) == 2

        # Events collapse per kind and are all named in the line.
        for kind in ("sleep", "wake_up", "load_weights", "load_weights"):
            mark_lifecycle_event(model, kind)
        assert state.pending_lifecycle_events == ["sleep", "wake_up", "load_weights"]
        bind_dual_precision(model, BASE_PRECISION_BF16)  # BF16 bind: not yet
        assert len(_lifecycle_lines(caplog)) == 2
        bind_dual_precision(model, BASE_PRECISION_INT4)
        lines = _lifecycle_lines(caplog)
        assert len(lines) == 3 and "after sleep,wake_up,load_weights:" in lines[2]


def test_lifecycle_revalidation_flags_a_changed_shadow_at_error(caplog):
    model, state, _ = _attach(validate_lifecycle=True)
    bind_dual_precision(model, BASE_PRECISION_INT4)
    probe = state.lifecycle_probes[0]
    with torch.no_grad():
        probe.layer.weight.add_(1.0)  # corrupt the shadow store in place
    mark_lifecycle_event(model, "wake_up")
    with caplog.at_level(logging.WARNING, logger=VALIDATION_LOGGER):
        bind_dual_precision(model, BASE_PRECISION_INT4)
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1
    assert "1/3 probes not exact" in errors[0].getMessage()
    assert f"{probe.name}: exact=False" in errors[0].getMessage()


def test_mark_lifecycle_event_is_noop_without_state():
    plain = build_wrapped_model(NAMES, HIDDEN)
    mark_lifecycle_event(plain, "wake_up")  # no state, nothing to arm
    assert get_dual_precision_state(plain) is None


def test_wrapped_load_weights_marks_the_event_without_touching_topology():
    class Model(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.calls = []

        def load_weights(self, weights):
            self.calls.append(list(weights))
            return {"loaded"}

    model, state, _ = _attach()
    wrapped = Model(model)
    # Hang the state where the runner would find it (the model itself).
    from vllm.model_executor.dual_precision.binding import set_dual_precision_state

    set_dual_precision_state(wrapped, state)
    before = module_dict_snapshot(wrapped)
    wrap_load_weights_for_lifecycle(wrapped)
    wrap_load_weights_for_lifecycle(wrapped)  # idempotent
    assert module_dict_snapshot(wrapped) == before
    assert wrapped.load_weights.__wrapped__.__func__ is Model.load_weights
    assert wrapped.load_weights([("a", torch.zeros(1))]) == {"loaded"}
    assert wrapped.calls and state.pending_lifecycle_events == ["load_weights"]


# ------------------------------------------------------------- log contract


def test_int4_bind_line_is_logged_at_warning(caplog):
    model, _, _ = _attach()
    with caplog.at_level(logging.WARNING, logger=BINDING_LOGGER):
        bind_dual_precision(model, BASE_PRECISION_INT4)
    lines = [r for r in caplog.records if "QLoRA base path bound" in r.getMessage()]
    assert len(lines) == 1 and lines[0].levelno == logging.WARNING
    message = lines[0].getMessage()
    assert "precision=int4" in message and "lora_base_layers=3" in message
    assert "int4_shadow_active=3" in message
    wrapper = model.get_submodule(NAMES[0])
    assert get_binding(wrapper).active is get_binding(wrapper).int4_or_fallback

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical and lifecycle validation of the INT4 shadow store.

One check is always on: the *sanity probe* (:func:`sanity_probe_shadow`)
compares one attached INT4 linear with its BF16 twin on one random input and
raises when the cosine is not above :data:`SANITY_MIN_COSINE`; a shadow that
is noise (a dummy-loaded or mismatched checkpoint) can therefore never start
serving. It runs at attach time, or, when the engine itself loaded dummy base
weights (verl syncs the real ones later), at the first INT4 bind after the
first weight-load lifecycle event.

Two further checks are off by default and gated by registered env vars:

* ``VLLM_DUAL_PRECISION_VALIDATE_SHADOW``: at attach time, compare every
  attached INT4 linear with its BF16 twin on one random input and log the
  worst cosine / relative RMSE (a mis-matched or mis-packed shadow shows up
  as a cosine far below 0.9).
* ``VLLM_DUAL_PRECISION_VALIDATE_LIFECYCLE``: at attach time, record
  fixed-input probes for the first attached INT4 linears; at the first INT4
  bind re-run them as the baseline, and re-run them again at the first INT4
  bind after every sleep/wake-up and weight-load lifecycle event (the worker
  and the runner mark those through ``mark_lifecycle_event``), logging
  ``exact=True`` per probe when the shadow survived untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase

if TYPE_CHECKING:
    from vllm.model_executor.dual_precision.binding import DualPrecisionState

logger = init_logger(__name__)

MAX_LIFECYCLE_PROBES = 6
SANITY_MIN_COSINE = 0.5
"""A real GPTQ shadow scores >= 0.9 against its BF16 twin (measured 0.9856
worst layer on Qwen3.5-9B); random weights score ~0."""


@dataclass
class ShadowValidation:
    name: str
    cosine: float
    relative_rmse: float


@dataclass
class LifecycleProbe:
    name: str
    layer: LinearBase
    sample_cpu: torch.Tensor
    expected: torch.Tensor


@dataclass
class LifecycleResult:
    name: str
    exact: bool
    cosine: float
    max_abs: float


def _apply(layer: LinearBase, x: torch.Tensor) -> torch.Tensor:
    bias = getattr(layer, "bias", None)
    with torch.no_grad():
        return layer.quant_method.apply(layer, x, bias)


def compare_shadow_numerics(
    name: str,
    bf16_layer: LinearBase,
    int4_layer: LinearBase,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> ShadowValidation:
    """Cosine and relative RMSE of the INT4 output against the BF16 output."""
    input_size = int(bf16_layer.input_size)
    parameter = next(bf16_layer.parameters(recurse=False))
    sample = torch.randn(
        2, input_size, device=parameter.device, dtype=dtype, generator=generator
    )
    bf16_flat = _apply(bf16_layer, sample).float().flatten()
    int4_flat = _apply(int4_layer, sample).float().flatten()
    cosine = torch.nn.functional.cosine_similarity(bf16_flat, int4_flat, dim=0).item()
    relative_rmse = (
        (bf16_flat - int4_flat).square().mean().sqrt()
        / bf16_flat.square().mean().sqrt().clamp_min(1e-12)
    ).item()
    return ShadowValidation(name, cosine, relative_rmse)


def sanity_probe_shadow(
    name: str,
    bf16_layer: LinearBase,
    int4_layer: LinearBase,
    dtype: torch.dtype,
    *,
    shadow_load_format: str,
    when: str,
) -> ShadowValidation:
    """Always-on guard: one random input through one attached layer, both
    bases; raise unless the INT4 output tracks the BF16 one.

    ``when`` names the moment for the log/error ("at attach", "at the first
    INT4 bind after load_weights").
    """
    result = compare_shadow_numerics(name, bf16_layer, int4_layer, dtype)
    if not result.cosine > SANITY_MIN_COSINE:  # NaN fails too
        raise RuntimeError(
            f"Dual precision INT4 shadow sanity probe failed {when}: layer "
            f"{name} cosine vs its BF16 twin = {result.cosine:.4f} "
            f"(rel_rmse={result.relative_rmse:.4f}, required > "
            f"{SANITY_MIN_COSINE}; shadow load_format={shadow_load_format}). "
            "The shadow does not match the base model (dummy-loaded, "
            "mis-packed or the wrong checkpoint); refusing to serve it."
        )
    logger.info(
        "Dual precision INT4 shadow sanity probe passed %s: layer %s "
        "cos=%.4f rel_rmse=%.4f (shadow load_format=%s).",
        when,
        name,
        result.cosine,
        result.relative_rmse,
        shadow_load_format,
    )
    return result


def log_shadow_validation(results: list[ShadowValidation], worst_k: int = 10) -> None:
    if not results:
        return
    worst = sorted(results, key=lambda item: item.cosine)[:worst_k]
    logger.warning(
        "Dual precision shadow numerical validation (worst cosine): %s",
        "; ".join(
            f"{item.name}: cos={item.cosine:.6f}, rel_rmse={item.relative_rmse:.6f}"
            for item in worst
        ),
    )


def shadow_probe_input(layer: LinearBase, dtype: torch.dtype) -> torch.Tensor:
    """Small deterministic input used to detect shadow corruption on wake-up."""
    input_size = int(layer.input_size)
    values = torch.arange(input_size, device=next(layer.parameters()).device)
    values = ((values % 251).float() - 125.0) / 125.0
    return values.to(dtype=dtype).unsqueeze(0)


def record_lifecycle_probe(
    name: str, layer: LinearBase, dtype: torch.dtype
) -> LifecycleProbe:
    sample = shadow_probe_input(layer, dtype)
    expected = _apply(layer, sample).float().cpu()
    return LifecycleProbe(name, layer, sample.cpu(), expected)


def run_lifecycle_probes(probes: list[LifecycleProbe]) -> list[LifecycleResult]:
    results: list[LifecycleResult] = []
    for probe in probes:
        device = next(probe.layer.parameters()).device
        actual = _apply(probe.layer, probe.sample_cpu.to(device=device)).float().cpu()
        expected_flat = probe.expected.flatten()
        actual_flat = actual.flatten()
        cosine = torch.nn.functional.cosine_similarity(
            expected_flat, actual_flat, dim=0
        ).item()
        max_abs = (expected_flat - actual_flat).abs().max().item()
        results.append(
            LifecycleResult(
                probe.name, torch.equal(probe.expected, actual), cosine, max_abs
            )
        )
    return results


def validate_shadow_lifecycle(
    state: DualPrecisionState, after: list[str] | None = None
) -> list[LifecycleResult]:
    """Re-run the recorded probes: at the first INT4 bind (baseline) and at
    the first INT4 bind after the lifecycle events in ``after``."""
    state.lifecycle_validated = True
    if not state.lifecycle_probes:
        return []
    results = run_lifecycle_probes(state.lifecycle_probes)
    moment = "at first INT4 bind"
    if after:
        moment += " after " + ",".join(after)
    summary = "; ".join(
        f"{item.name}: exact={item.exact}, cos={item.cosine:.8f}, "
        f"max_abs={item.max_abs:.6g}"
        for item in results
    )
    if all(item.exact for item in results):
        logger.warning(
            "Dual precision shadow lifecycle validation %s: %s", moment, summary
        )
    else:
        logger.error(
            "Dual precision shadow lifecycle validation %s: shadow store "
            "changed since attach (%d/%d probes not exact): %s",
            moment,
            sum(not item.exact for item in results),
            len(results),
            summary,
        )
    return results

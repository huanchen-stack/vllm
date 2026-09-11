# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch-free switching policy for dual-precision (BF16 -> INT4) rollouts.

This module is the single definition of the policy contract shared by the
scheduler (which decides *when* a rollout flips its base precision), the GPU
worker (which reads ``capture_max_batch`` to bound INT4 CUDA-graph capture),
and the offline/online policy toolkit (which writes policy JSON files).

It imports only the standard library so it can be loaded from a file path by
tooling that has no torch or CUDA available.  Do not add heavy imports.

Policy kinds (decision 6 of the migration plan) all run through one runtime
path: a dense lookup table indexed by (response frontier, prompt bucket,
live batch) returning the committed switch frontier, plus a live-batch guard.

* ``fixed_frontier:<K>``  every table cell is ``K``: switch when the longest
  response reaches ``K`` tokens.
* ``fixed_threshold:<t>`` every table cell is the first frontier and the
  guard is ``t``: switch as soon as the live batch has drained to ``t``.
* ``uniform_w4``          INT4 from the first token; no table.
* ``<path>.json``         a calibrated table (schema 2/4/5/6, see the design
  document ``docs/design/precision_policy.md``).

The receding-horizon cost model that defines the calibrated tables is kept as
a pure offline function (``CostModel``); it is never evaluated on the
scheduler thread.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ACCEPTED_SCHEMA_VERSIONS",
    "CostModel",
    "DEFAULT_CAPTURE_MAX_BATCH",
    "DEFAULT_PROMPT_BUCKET_TOKENS",
    "DEFAULT_SCAN_INTERVAL_TOKENS",
    "DROPPED_POLICY_KEYS",
    "Decision",
    "KIND_COST_MODEL",
    "KIND_FIXED_FRONTIER",
    "KIND_FIXED_THRESHOLD",
    "KIND_LOOKUP",
    "KIND_UNIFORM_W4",
    "LOOKUP_LAYOUT",
    "LookupTable",
    "PolicyDecider",
    "PolicyRevisionError",
    "PolicySpec",
    "PolicyStore",
    "PrecisionPolicy",
    "Prediction",
    "REASON_ALREADY_SWITCHED",
    "REASON_COMMITTED",
    "REASON_GUARD_BLOCKED",
    "REASON_NOT_OBSERVED",
    "REASON_NO_COMMITMENT",
    "REASON_SWITCH",
    "UNBOUNDED_CAPTURE_MAX_BATCH",
    "load_precision_policy",
    "policy_from_json",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Policy JSON schema versions the loader accepts.  Schema 2/3 carry a cost
#: model (offline oracle only); 4/5/6 carry a dense lookup table.
ACCEPTED_SCHEMA_VERSIONS: tuple[int, ...] = (2, 3, 4, 5, 6)

#: The only supported flat layout of ``lookup_table.committed_frontiers``.
LOOKUP_LAYOUT = "frontier_major,prompt_bucket,live_batch"

#: Top-level policy keys that were dropped by decision 4 of the migration
#: plan.  A file that still carries them is rejected with a clear error.
DROPPED_POLICY_KEYS: dict[str, str] = {
    "switch_thresholds": (
        "the switch_thresholds table mode was dropped (decision 4); use "
        "fixed_switch_frontier or a lookup_table"
    ),
    "lookup_hierarchy": (
        "the lookup_hierarchy mode was dropped (decision 4); use a single lookup_table"
    ),
}

DEFAULT_SCAN_INTERVAL_TOKENS = 250
DEFAULT_CAPTURE_MAX_BATCH = 32
DEFAULT_PROMPT_BUCKET_TOKENS = 128
#: Degenerate tables built from inline specs cover this many response tokens.
DEFAULT_SPEC_RESPONSE_CAP = 1 << 20
#: ``capture_max_batch`` of the ``uniform_w4`` kind: no INT4 capture ceiling.
UNBOUNDED_CAPTURE_MAX_BATCH = 1 << 30

KIND_LOOKUP = "lookup"
KIND_FIXED_FRONTIER = "fixed_frontier"
KIND_FIXED_THRESHOLD = "fixed_threshold"
KIND_UNIFORM_W4 = "uniform_w4"
#: Schema 2/3 file with a cost model and no runtime table (offline oracle).
KIND_COST_MODEL = "cost_model"

COMMITMENT_MONOTONE = "monotone"
COMMITMENT_RECEDING = "receding"

REASON_ALREADY_SWITCHED = "already_switched"
REASON_SWITCH = "switch"
REASON_NOT_OBSERVED = "not_observed"
REASON_NO_COMMITMENT = "no_commitment"
REASON_COMMITTED = "committed"
REASON_GUARD_BLOCKED = "guard_blocked"

BASE_PRECISION_BF16 = "bf16"
BASE_PRECISION_INT4 = "int4"


class PolicyRevisionError(RuntimeError):
    """A policy reload did not produce an acceptable revision."""


# ---------------------------------------------------------------------------
# Policy spec (decision 6)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicySpec:
    """Parsed form of the single policy flag.

    ``kind`` is one of the ``KIND_*`` constants or ``""`` when disabled.
    ``value`` carries the integer of ``fixed_frontier:<K>`` /
    ``fixed_threshold:<t>``; ``path`` carries the JSON path.
    """

    kind: str
    value: int | None = None
    path: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.kind)

    @property
    def is_file(self) -> bool:
        return self.path is not None

    @classmethod
    def parse(cls, spec: str | None) -> PolicySpec:
        text = (spec or "").strip()
        if not text:
            return cls(kind="")
        if text == KIND_UNIFORM_W4:
            return cls(kind=KIND_UNIFORM_W4)
        for kind in (KIND_FIXED_FRONTIER, KIND_FIXED_THRESHOLD):
            prefix = kind + ":"
            if text.startswith(prefix):
                raw_value = text[len(prefix) :].strip()
                try:
                    value = int(raw_value)
                except ValueError as error:
                    raise ValueError(
                        f"Precision policy spec {text!r}: {kind} needs an "
                        f"integer, got {raw_value!r}"
                    ) from error
                if value <= 0:
                    raise ValueError(
                        f"Precision policy spec {text!r}: {kind} must be positive"
                    )
                return cls(kind=kind, value=value)
        if text in (KIND_FIXED_FRONTIER, KIND_FIXED_THRESHOLD):
            raise ValueError(
                f"Precision policy spec {text!r} needs a value, e.g. {text}:8"
            )
        if text.endswith(".json") or "/" in text or Path(text).exists():
            return cls(kind=KIND_LOOKUP, path=text)
        raise ValueError(
            f"Unrecognized precision policy spec {text!r}; expected "
            f"'fixed_threshold:<int>', 'fixed_frontier:<int>', "
            f"'uniform_w4', or a path to a policy JSON file"
        )

    def __str__(self) -> str:
        if not self.kind:
            return ""
        if self.path is not None:
            return self.path
        if self.value is not None:
            return f"{self.kind}:{self.value}"
        return self.kind


# ---------------------------------------------------------------------------
# Dense lookup table
# ---------------------------------------------------------------------------

_LOOKUP_REQUIRED_FIELDS = (
    "frontier_start",
    "frontier_step",
    "frontier_count",
    "prompt_bucket_start",
    "prompt_bucket_step",
    "prompt_bucket_count",
    "live_batch_start",
    "live_batch_count",
    "committed_frontiers",
)


@dataclass(frozen=True)
class LookupTable:
    """Dense ``(frontier, prompt bucket, live batch) -> committed frontier``.

    Indexing is constant time.  The frontier axis is *not* clamped: a
    frontier outside ``[frontier_start, frontier_start + step * count)``
    yields ``None``.  The prompt axis is rounded to the nearest bucket
    (Python ``round``, half to even) and clamped; the live axis is clamped.
    A stored ``0`` means "no planned switch" and is returned as ``None``.
    """

    frontier_start: int
    frontier_step: int
    frontier_count: int
    prompt_bucket_start: int
    prompt_bucket_step: int
    prompt_bucket_count: int
    live_batch_start: int
    live_batch_count: int
    committed_frontiers: tuple[int, ...]
    layout: str = LOOKUP_LAYOUT

    def __post_init__(self) -> None:
        if self.layout != LOOKUP_LAYOUT:
            raise ValueError(
                f"lookup_table layout must be {LOOKUP_LAYOUT!r}, got {self.layout!r}"
            )
        dims = (self.frontier_count, self.prompt_bucket_count, self.live_batch_count)
        if any(value <= 0 for value in dims) or any(
            value <= 0 for value in (self.frontier_step, self.prompt_bucket_step)
        ):
            raise ValueError("lookup_table dimensions and steps must be positive")
        if self.frontier_start < 0 or self.live_batch_start < 0:
            raise ValueError("lookup_table axis starts must be non-negative")
        expected = math.prod(dims)
        if len(self.committed_frontiers) != expected:
            raise ValueError(
                f"lookup_table committed_frontiers has "
                f"{len(self.committed_frontiers)} entries; expected {expected}"
            )
        if any(value < 0 for value in self.committed_frontiers):
            raise ValueError("lookup_table committed_frontiers must be >= 0")

    @classmethod
    def from_json(cls, raw: Any, label: str = "lookup_table") -> LookupTable:
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be an object")
        missing = [key for key in _LOOKUP_REQUIRED_FIELDS if key not in raw]
        if missing:
            raise ValueError(f"{label} is missing fields: " + ", ".join(missing))
        decisions = raw["committed_frontiers"]
        if not isinstance(decisions, list):
            raise ValueError(f"{label} committed_frontiers must be a list")
        try:
            cells = tuple(int(value) for value in decisions)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"{label} committed_frontiers must hold integers"
            ) from error
        try:
            return cls(
                frontier_start=int(raw["frontier_start"]),
                frontier_step=int(raw["frontier_step"]),
                frontier_count=int(raw["frontier_count"]),
                prompt_bucket_start=int(raw["prompt_bucket_start"]),
                prompt_bucket_step=int(raw["prompt_bucket_step"]),
                prompt_bucket_count=int(raw["prompt_bucket_count"]),
                live_batch_start=int(raw["live_batch_start"]),
                live_batch_count=int(raw["live_batch_count"]),
                committed_frontiers=cells,
                layout=str(raw.get("layout", LOOKUP_LAYOUT)),
            )
        except ValueError as error:
            raise ValueError(f"{label}: {error}") from error

    @classmethod
    def constant(
        cls,
        committed_frontier: int,
        *,
        scan_interval_tokens: int,
        response_cap: int,
    ) -> LookupTable:
        """A degenerate table returning ``committed_frontier`` everywhere.

        One prompt bucket and one live column: clamping maps every input to
        the single cell.  The frontier axis spans ``[scan, response_cap)``.
        """
        count = max(1, (response_cap - scan_interval_tokens) // scan_interval_tokens)
        return cls(
            frontier_start=scan_interval_tokens,
            frontier_step=scan_interval_tokens,
            frontier_count=count,
            prompt_bucket_start=0,
            prompt_bucket_step=DEFAULT_PROMPT_BUCKET_TOKENS,
            prompt_bucket_count=1,
            live_batch_start=1,
            live_batch_count=1,
            committed_frontiers=(int(committed_frontier),) * count,
        )

    @property
    def frontier_tokens(self) -> tuple[int, ...]:
        return tuple(
            self.frontier_start + index * self.frontier_step
            for index in range(self.frontier_count)
        )

    @property
    def prompt_bucket_tokens(self) -> tuple[int, ...]:
        return tuple(
            self.prompt_bucket_start + index * self.prompt_bucket_step
            for index in range(self.prompt_bucket_count)
        )

    def frontier_index(self, frontier_tokens: int) -> int | None:
        index = (int(frontier_tokens) - self.frontier_start) // self.frontier_step
        if index < 0 or index >= self.frontier_count:
            return None
        return index

    def prompt_index(self, prompt_tokens: float) -> int:
        index = round(
            (prompt_tokens - self.prompt_bucket_start) / self.prompt_bucket_step
        )
        return max(0, min(self.prompt_bucket_count - 1, index))

    def live_index(self, live: int) -> int:
        return max(0, min(self.live_batch_count - 1, int(live) - self.live_batch_start))

    def flat_index(
        self, frontier_index: int, prompt_index: int, live_index: int
    ) -> int:
        return (
            frontier_index * self.prompt_bucket_count + prompt_index
        ) * self.live_batch_count + live_index

    def committed_frontier(
        self, frontier_tokens: int, prompt_tokens: float, live: int
    ) -> int | None:
        """Return the planned switch frontier or ``None`` (no planned switch)."""
        frontier_index = self.frontier_index(frontier_tokens)
        if frontier_index is None:
            return None
        planned = self.committed_frontiers[
            self.flat_index(
                frontier_index,
                self.prompt_index(prompt_tokens),
                self.live_index(live),
            )
        ]
        return planned if planned > 0 else None

    def to_json(self) -> dict[str, Any]:
        return {
            "layout": self.layout,
            "frontier_start": self.frontier_start,
            "frontier_step": self.frontier_step,
            "frontier_count": self.frontier_count,
            "prompt_bucket_start": self.prompt_bucket_start,
            "prompt_bucket_step": self.prompt_bucket_step,
            "prompt_bucket_count": self.prompt_bucket_count,
            "live_batch_start": self.live_batch_start,
            "live_batch_count": self.live_batch_count,
            "committed_frontiers": list(self.committed_frontiers),
        }


# ---------------------------------------------------------------------------
# Offline cost model (receding-horizon full-RL-step cost)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Prediction:
    """Result of ``CostModel.predict_receding_horizon``."""

    switch_now: bool
    current_frontier: int
    planned_frontier: int | None
    planned_live_batch: int
    bf16_remaining_rollout_seconds: float
    bf16_remaining_tokens: float
    bf16_remaining_cost_seconds: float
    planned_remaining_rollout_seconds: float
    planned_remaining_tokens: float
    planned_remaining_cost_seconds: float
    predicted_gain_seconds: float


def _interp_log(value: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Piecewise-linear interpolation in log2(x), clamped to the grid ends."""
    value = max(xs[0], min(xs[-1], value))
    log_value = math.log2(value)
    for index in range(1, len(xs)):
        if value <= xs[index]:
            left = math.log2(xs[index - 1])
            right = math.log2(xs[index])
            weight = 0.0 if right == left else (log_value - left) / (right - left)
            return ys[index - 1] + weight * (ys[index] - ys[index - 1])
    return ys[-1]


class CostModel:
    """Receding-horizon full-RL-step cost model.  **Offline only.**

    A single scalar prediction costs 0.25-0.5 s in pure Python; the scheduler
    never calls this.  Tables are built from it offline and the archived
    online predictions serve as its golden oracle.

    ``predict_receding_horizon`` reproduces the archived online predictor
    exactly (archived response-length samples, chunked survival integration,
    speed scales, tail-correction anchors, strict ``<`` tie rule).
    ``predict`` is the per-bin expected cost used by the hazard-based table
    builder (``cost(p, fi, pj, live, alive)`` in the design document) and
    ``plan_switch`` is its global search with the same tie rule.
    """

    def __init__(
        self,
        cost_model: dict[str, Any],
        *,
        scan_interval_tokens: int,
        capture_max_batch: int,
        live_batch_max: int | None = None,
    ) -> None:
        if not isinstance(cost_model, dict):
            raise ValueError("cost_model must be an object")
        for key in ("response_cap", "downstream_seconds_per_token"):
            if key not in cost_model:
                raise ValueError(f"cost_model is missing field {key!r}")
        for key in ("tpot_batches", "tpot_contexts", "bf16_tpot_ms", "w4_tpot_ms"):
            if key not in cost_model:
                raise ValueError(f"cost_model is missing field {key!r}")
        if scan_interval_tokens <= 0:
            raise ValueError("scan_interval_tokens must be positive")
        self.raw = cost_model
        self.scan_interval_tokens = int(scan_interval_tokens)
        self.capture_max_batch = int(capture_max_batch)
        self.live_batch_max = live_batch_max
        self.response_cap = int(cost_model["response_cap"])
        self.chunk_tokens = int(cost_model.get("integration_chunk_tokens", 64))
        self.min_conditional_samples = int(cost_model.get("min_conditional_samples", 5))
        self.downstream_seconds_per_token = float(
            cost_model["downstream_seconds_per_token"]
        )
        self.switch_overhead_seconds = float(
            cost_model.get("switch_overhead_seconds", 0.0)
        )
        self.required_gain_seconds = float(cost_model.get("required_gain_seconds", 0.0))
        self.prompt_bucket_tokens = int(
            cost_model.get("prompt_bucket_tokens", DEFAULT_PROMPT_BUCKET_TOKENS)
        )
        self._tpot_batches = [float(value) for value in cost_model["tpot_batches"]]
        self._tpot_contexts = [float(value) for value in cost_model["tpot_contexts"]]
        self._lengths: dict[str, list[int]] = {}
        for precision in ("bf16", "w4"):
            lengths = cost_model.get(f"{precision}_lengths")
            if lengths is not None:
                self._lengths[precision] = [int(value) for value in lengths]
        self._speed_scale = {
            precision: float(cost_model.get(f"{precision}_speed_scale", 1.0))
            for precision in ("bf16", "w4")
        }
        # Instance-level memoization (the experimental code used module-level
        # dicts keyed by id(policy), which alias after a reload).
        self._tpot_cache: dict[tuple[str, int, float], float] = {}
        self._suffix_cache: dict[
            tuple[str, int, int, int], tuple[float, float] | None
        ] = {}
        self._prefix_cache: dict[
            tuple[int, int, int, int], tuple[float, float, int] | None
        ] = {}

    # -- primitives -------------------------------------------------------

    def tpot_ms(self, batch: int, context: float, precision: str) -> float:
        """Log2-interpolated TPOT; non-finite grid cells are skipped per row."""
        key = (precision, int(batch), float(context))
        cached = self._tpot_cache.get(key)
        if cached is not None:
            return cached
        grid = self.raw[f"{precision}_tpot_ms"]
        supported_batches: list[float] = []
        along_context: list[float] = []
        for batch_value, row in zip(self._tpot_batches, grid, strict=True):
            supported = [
                (context_value, float(tpot))
                for context_value, tpot in zip(self._tpot_contexts, row, strict=True)
                if math.isfinite(float(tpot))
            ]
            if not supported:
                continue
            supported_contexts, supported_tpot = zip(*supported, strict=True)
            supported_batches.append(batch_value)
            along_context.append(
                _interp_log(context, list(supported_contexts), list(supported_tpot))
            )
        if not supported_batches:
            raise ValueError(f"cost_model {precision}_tpot_ms has no finite cells")
        result = _interp_log(float(batch), supported_batches, along_context)
        self._tpot_cache[key] = result
        return result

    def expected_tpot_ms(
        self, live: int, alive_probability: float, context: float, precision: str
    ) -> float:
        """E[TPOT] over the binomial number of survivors (approximated).

        The expectation is taken at ``E[live | at least one alive]`` and
        scaled by ``P(at least one alive)``, preserving the probability mass
        of an empty future batch.  (No ``live_batch_max`` clip here: this is
        the archived online predictor's exact arithmetic.)
        """
        if alive_probability <= 0.0 or live <= 0:
            return 0.0
        if alive_probability >= 1.0:
            return self.tpot_ms(live, context, precision)
        probability_nonempty = 1.0 - (1.0 - alive_probability) ** live
        conditional_live = live * alive_probability / probability_nonempty
        effective_live = max(1, round(conditional_live))
        return probability_nonempty * self.tpot_ms(effective_live, context, precision)

    def tail_correction(self, live: int, name: str) -> float:
        anchors = self.raw.get("tail_correction_anchors", [])
        if not anchors:
            return 1.0
        xs = [float(anchor["live_batch"]) for anchor in anchors]
        ys = [float(anchor[name]) for anchor in anchors]
        return _interp_log(float(live), xs, ys)

    def prompt_bucket(self, median_prompt_tokens: float) -> int:
        step = self.prompt_bucket_tokens
        return int(round(median_prompt_tokens / step) * step)

    @staticmethod
    def _conditional(lengths: list[int], frontier: int) -> list[int]:
        return [length for length in lengths if length > frontier]

    def _lengths_for(self, precision: str) -> list[int]:
        try:
            return self._lengths[precision]
        except KeyError as error:
            raise ValueError(
                f"cost_model has no {precision}_lengths samples"
            ) from error

    def expected_suffix(
        self, precision: str, frontier: int, live: int, median_prompt_tokens: float
    ) -> tuple[float, float] | None:
        """(rollout seconds, sampled tokens) to decode from ``frontier`` to cap."""
        prompt_bucket = self.prompt_bucket(median_prompt_tokens)
        key = (precision, int(frontier), int(live), prompt_bucket)
        if key in self._suffix_cache:
            return self._suffix_cache[key]
        prompt = float(prompt_bucket)
        conditional = self._conditional(self._lengths_for(precision), frontier)
        if len(conditional) < self.min_conditional_samples:
            self._suffix_cache[key] = None
            return None
        cap = self.response_cap
        chunk = self.chunk_tokens
        remaining_tokens = (
            live * sum(length - frontier for length in conditional) / len(conditional)
        )
        modeled_ms = 0.0
        for position in range(frontier, cap, chunk):
            alive = sum(length > position for length in conditional) / len(conditional)
            modeled_ms += self.expected_tpot_ms(
                live, alive, position + prompt, precision
            ) * min(chunk, cap - position)
        result = (modeled_ms / 1000.0 * self._speed_scale[precision], remaining_tokens)
        self._suffix_cache[key] = result
        return result

    def expected_bf16_prefix(
        self, start: int, end: int, live: int, median_prompt_tokens: float
    ) -> tuple[float, float, int] | None:
        """(seconds, tokens, surviving live) for BF16 from ``start`` to ``end``."""
        prompt_bucket = self.prompt_bucket(median_prompt_tokens)
        key = (int(start), int(end), int(live), prompt_bucket)
        if key in self._prefix_cache:
            return self._prefix_cache[key]
        prompt = float(prompt_bucket)
        conditional = self._conditional(self._lengths_for("bf16"), start)
        if not conditional:
            self._prefix_cache[key] = None
            return None
        survival = sum(length > end for length in conditional) / len(conditional)
        future_live = max(1, round(live * survival))
        chunk = self.chunk_tokens
        modeled_ms = 0.0
        for position in range(start, end, chunk):
            alive = sum(length > position for length in conditional) / len(conditional)
            modeled_ms += self.expected_tpot_ms(
                live, alive, position + prompt, "bf16"
            ) * min(chunk, end - position)
        tokens = (
            live
            * sum(min(length, end) - start for length in conditional)
            / len(conditional)
        )
        seconds = modeled_ms / 1000.0 * self._speed_scale["bf16"]
        result = (seconds, tokens, future_live)
        self._prefix_cache[key] = result
        return result

    # -- archived online predictor (golden oracle) ------------------------

    def predict_receding_horizon(
        self, current_frontier: int, live: int, median_prompt_tokens: float
    ) -> Prediction | None:
        """Full remaining RL-step cost of staying BF16 versus the best plan.

        Every future frontier on the scan grid is costed as BF16 prefix plus
        W4 suffix (plus switch overhead); a future whose surviving batch
        exceeds ``capture_max_batch`` is skipped.  Candidates are compared
        with strict ``<`` while iterating from the current frontier upward,
        so ties resolve to the earliest frontier ("now").  ``switch_now`` is
        set only when the best plan is the current frontier and the gain
        exceeds ``required_gain_seconds``.
        """
        bf16 = self.expected_suffix(
            "bf16", current_frontier, live, median_prompt_tokens
        )
        if bf16 is None:
            return None
        slope = self.downstream_seconds_per_token
        bf16_rollout, bf16_tokens = bf16
        bf16_cost = bf16_rollout + slope * bf16_tokens
        best: tuple[float, int, int, float, float] | None = None
        for future in range(
            current_frontier, self.response_cap, self.scan_interval_tokens
        ):
            prefix = self.expected_bf16_prefix(
                current_frontier, future, live, median_prompt_tokens
            )
            if prefix is None:
                continue
            prefix_seconds, prefix_tokens, future_live = prefix
            if future_live > self.capture_max_batch:
                continue
            w4 = self.expected_suffix("w4", future, future_live, median_prompt_tokens)
            if w4 is None:
                continue
            w4_seconds, w4_tokens = w4
            w4_seconds *= self.tail_correction(future_live, "time_correction")
            w4_tokens *= self.tail_correction(future_live, "token_correction")
            plan_rollout = prefix_seconds + w4_seconds + self.switch_overhead_seconds
            plan_tokens = prefix_tokens + w4_tokens
            plan_cost = plan_rollout + slope * plan_tokens
            candidate = (plan_cost, future, future_live, plan_rollout, plan_tokens)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is None:
            return None
        plan_cost, planned_frontier, planned_live, plan_rollout, plan_tokens = best
        gain = bf16_cost - plan_cost
        return Prediction(
            switch_now=planned_frontier == current_frontier
            and gain > self.required_gain_seconds,
            current_frontier=current_frontier,
            planned_frontier=planned_frontier,
            planned_live_batch=planned_live,
            bf16_remaining_rollout_seconds=bf16_rollout,
            bf16_remaining_tokens=bf16_tokens,
            bf16_remaining_cost_seconds=bf16_cost,
            planned_remaining_rollout_seconds=plan_rollout,
            planned_remaining_tokens=plan_tokens,
            planned_remaining_cost_seconds=plan_cost,
            predicted_gain_seconds=gain,
        )

    # -- hazard-table builder primitives ----------------------------------

    @property
    def frontier_count(self) -> int:
        """Number of frontier bins ``scan, 2*scan, ..., < response_cap``."""
        return max(
            0,
            (self.response_cap - self.scan_interval_tokens)
            // self.scan_interval_tokens,
        )

    def frontier_tokens(self, frontier_index: int) -> int:
        return self.scan_interval_tokens * (frontier_index + 1)

    def prompt_tokens(self, prompt_bucket_index: int) -> int:
        return self.prompt_bucket_tokens * prompt_bucket_index

    def predict(
        self,
        precision: str,
        frontier_index: int,
        prompt_bucket_index: int,
        live: int,
        alive: Sequence[float],
    ) -> float:
        """Expected cost (seconds) of decoding bins ``frontier_index..`` under
        ``precision`` when ``alive[k]`` is the per-request probability of
        still running at the start of bin ``frontier_index + k``.

        Rollout seconds use ``E[TPOT | non-empty batch] * P(non-empty)`` per
        bin; the downstream term charges ``downstream_seconds_per_token`` for
        every expected sampled token.  No speed scale or tail correction is
        applied: this is the raw heatmap definition used by table builders.
        """
        if live <= 0:
            return 0.0
        cap = self.response_cap
        step = self.scan_interval_tokens
        prompt = float(self.prompt_tokens(prompt_bucket_index))
        rollout_seconds = 0.0
        downstream_tokens = 0.0
        for offset, alive_probability in enumerate(alive):
            frontier = self.frontier_tokens(frontier_index + offset)
            if frontier >= cap:
                break
            tokens = min(step, cap - frontier)
            alive_probability = float(alive_probability)
            downstream_tokens += alive_probability * tokens
            if alive_probability <= 0.0:
                continue
            nonempty = 1.0 - (1.0 - alive_probability) ** live
            if nonempty <= 0.0:
                continue
            effective = max(1, round(live * alive_probability / nonempty))
            if self.live_batch_max is not None:
                effective = min(effective, self.live_batch_max)
            rollout_seconds += (
                self.tpot_ms(effective, prompt + frontier, precision)
                * tokens
                / 1000.0
                * nonempty
            )
        return (
            rollout_seconds
            + self.downstream_seconds_per_token * live * downstream_tokens
        )

    def plan_switch(
        self,
        frontier_index: int,
        prompt_bucket_index: int,
        live: int,
        bf16_alive: Sequence[float],
        w4_alive_from: Callable[[int], Sequence[float]],
    ) -> tuple[int | None, float, float]:
        """Global search over later switch frontiers.

        ``bf16_alive`` is the BF16 survival from ``frontier_index`` (one entry
        per remaining bin, ``bf16_alive[0] == 1``); ``w4_alive_from(fj)``
        returns the W4 survival from bin ``fj``.  Returns
        ``(switch_frontier_tokens | None, best_plan_cost, stay_cost)``.  The
        candidate loop uses strict ``<`` from the current bin upward, so ties
        resolve to the earliest frontier; a switch is planned only when the
        best plan is strictly cheaper than staying BF16.
        """
        stay = self.predict(
            "bf16", frontier_index, prompt_bucket_index, live, bf16_alive
        )
        best = math.inf
        best_index: int | None = None
        for future_index in range(frontier_index, frontier_index + len(bf16_alive)):
            prefix_bins = future_index - frontier_index
            prefix = self.predict(
                "bf16",
                frontier_index,
                prompt_bucket_index,
                live,
                bf16_alive[:prefix_bins],
            )
            reach = float(bf16_alive[prefix_bins])
            w4_alive = [reach * float(value) for value in w4_alive_from(future_index)]
            candidate = prefix + self.predict(
                "w4", future_index, prompt_bucket_index, live, w4_alive
            )
            if candidate < best:
                best = candidate
                best_index = future_index
        if best_index is None or not best < stay:
            return None, best, stay
        return self.frontier_tokens(best_index), best, stay


# ---------------------------------------------------------------------------
# Policy object and loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrecisionPolicy:
    """One object for every policy kind (decision 6).

    ``table`` is the runtime decision source for every kind except
    ``uniform_w4`` (switched from the first token) and ``cost_model`` (offline
    oracle only, not schedulable).  ``fixed_switch_frontier`` is set for the
    ``fixed_frontier`` kind (inline spec or the JSON field of the same name).
    """

    kind: str
    scan_interval_tokens: int
    arm_min_requests: int
    capture_max_batch: int
    commitment_enabled: bool
    receding_horizon_lookup: bool
    initial_rollout_batch: int | None
    max_switch_live_batch: int | None
    policy_revision: int
    table: LookupTable | None
    fixed_switch_frontier: int | None = None
    cost_model: CostModel | None = field(default=None, compare=False, repr=False)
    schema_version: int | None = None
    description: str = ""
    source: str = ""

    @property
    def commitment_mode(self) -> str:
        return (
            COMMITMENT_RECEDING if self.receding_horizon_lookup else COMMITMENT_MONOTONE
        )

    @property
    def switch_live_cap(self) -> int:
        """Largest *actual* live batch at which a switch may be applied.

        ``max_switch_live_batch`` when set, else the INT4 capture ceiling:
        switching above the ceiling would run eager INT4 steps.
        """
        if self.max_switch_live_batch is not None:
            return self.max_switch_live_batch
        return self.capture_max_batch

    @property
    def schedulable(self) -> bool:
        return self.kind != KIND_COST_MODEL

    def to_json(self) -> dict[str, Any]:
        """Schema-6 JSON of this policy (degenerate kinds included)."""
        payload: dict[str, Any] = {
            "schema_version": 6,
            "description": self.description or f"{self.kind} policy",
            "scan_interval_tokens": self.scan_interval_tokens,
            "arm_min_requests": self.arm_min_requests,
            "capture_max_batch": self.capture_max_batch,
            "commitment_enabled": self.commitment_enabled,
            "receding_horizon_lookup": self.receding_horizon_lookup,
            "initial_rollout_batch": self.initial_rollout_batch,
            "max_switch_live_batch": self.max_switch_live_batch,
            "calibration": {"kind": self.kind, "policy_revision": self.policy_revision},
        }
        if self.fixed_switch_frontier is not None:
            # The table is derived from the frontier; emit the frontier and
            # the span the loader needs to rebuild an identical table.
            payload["fixed_switch_frontier"] = self.fixed_switch_frontier
            if self.table is not None:
                payload["offline_cost_model"] = {
                    "response_cap": self.table.frontier_start
                    + self.table.frontier_step * self.table.frontier_count
                }
        elif self.table is not None:
            payload["lookup_table"] = self.table.to_json()
        if self.cost_model is not None:
            payload["cost_model"] = self.cost_model.raw
        return payload


def _read_policy_file(path: str) -> dict[str, Any]:
    policy_path = Path(path)
    try:
        raw = json.loads(policy_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cannot load precision policy {policy_path}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise ValueError(f"Precision policy {policy_path} must be a JSON object")
    return raw


def _optional_positive_int(raw: dict[str, Any], key: str) -> int | None:
    value = raw.get(key)
    if value is None:
        return None
    try:
        value = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{key} must be an integer") from error
    if value <= 0:
        raise ValueError(f"{key} must be positive when provided")
    return value


def policy_from_json(raw: dict[str, Any], source: str = "") -> PrecisionPolicy:
    """Validate a policy JSON object (schema 2-6) and build a policy.

    Accepts the lookup-table schemas written by the calibration tooling
    (4/5/6) and the cost-model schemas (2/3, offline oracle).  Rejects the
    dropped ``switch_thresholds`` / ``lookup_hierarchy`` modes and any
    malformed table.
    """
    if not isinstance(raw, dict):
        raise ValueError("Precision policy must be a JSON object")
    label = f"Precision policy {source}" if source else "Precision policy"
    for key, reason in DROPPED_POLICY_KEYS.items():
        if key in raw:
            raise ValueError(f"{label}: {reason}")

    schema_version = raw.get("schema_version")
    if schema_version is None:
        raise ValueError(f"{label}: schema_version is required")
    try:
        schema_version = int(schema_version)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: schema_version must be an integer") from error
    if schema_version not in ACCEPTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"{label}: schema_version {schema_version} is not supported "
            f"(accepted: {ACCEPTED_SCHEMA_VERSIONS})"
        )

    try:
        interval = int(raw.get("scan_interval_tokens", DEFAULT_SCAN_INTERVAL_TOKENS))
        arm_min = int(raw.get("arm_min_requests", 1))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label}: {error}") from error
    if interval <= 0 or arm_min <= 0:
        raise ValueError(
            f"{label}: scan_interval_tokens and arm_min_requests must be positive"
        )

    commitment_enabled = bool(raw.get("commitment_enabled", False))
    receding = bool(raw.get("receding_horizon_lookup", False))
    try:
        initial_rollout_batch = _optional_positive_int(raw, "initial_rollout_batch")
        max_switch_live_batch = _optional_positive_int(raw, "max_switch_live_batch")
        fixed_switch_frontier = _optional_positive_int(raw, "fixed_switch_frontier")
        capture_max = int(raw.get("capture_max_batch", DEFAULT_CAPTURE_MAX_BATCH))
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from error
    if capture_max <= 0:
        raise ValueError(f"{label}: capture_max_batch must be positive")
    if commitment_enabled and initial_rollout_batch is None:
        raise ValueError(
            f"{label}: a committed policy requires a positive initial_rollout_batch"
        )
    if max_switch_live_batch is not None and max_switch_live_batch > capture_max:
        raise ValueError(
            f"{label}: max_switch_live_batch ({max_switch_live_batch}) exceeds "
            f"capture_max_batch ({capture_max}); switching above the INT4 "
            f"capture ceiling would run eager steps"
        )

    lookup_raw = raw.get("lookup_table")
    cost_raw = raw.get("cost_model")
    if lookup_raw is None and cost_raw is None and fixed_switch_frontier is None:
        raise ValueError(
            f"{label}: requires one of lookup_table, cost_model, or "
            "fixed_switch_frontier"
        )
    if lookup_raw is not None and fixed_switch_frontier is not None:
        raise ValueError(
            f"{label}: lookup_table and fixed_switch_frontier are mutually exclusive"
        )

    table: LookupTable | None = None
    if lookup_raw is not None:
        try:
            table = LookupTable.from_json(lookup_raw)
        except ValueError as error:
            raise ValueError(f"{label}: {error}") from error
        if table.frontier_step != interval:
            raise ValueError(
                f"{label}: lookup_table frontier_step ({table.frontier_step}) "
                f"must equal scan_interval_tokens ({interval})"
            )
        kind = KIND_LOOKUP
    elif fixed_switch_frontier is not None:
        cap = _response_cap_hint(raw, fixed_switch_frontier)
        table = LookupTable.constant(
            fixed_switch_frontier, scan_interval_tokens=interval, response_cap=cap
        )
        kind = KIND_FIXED_FRONTIER
        receding = False
    else:
        kind = KIND_COST_MODEL

    cost_model: CostModel | None = None
    if cost_raw is not None:
        try:
            cost_model = CostModel(
                cost_raw,
                scan_interval_tokens=interval,
                capture_max_batch=capture_max,
                live_batch_max=initial_rollout_batch,
            )
        except ValueError as error:
            raise ValueError(f"{label}: {error}") from error

    calibration = raw.get("calibration") or {}
    if not isinstance(calibration, dict):
        raise ValueError(f"{label}: calibration must be an object")
    try:
        revision = int(calibration.get("policy_revision", 0))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{label}: calibration.policy_revision must be an integer"
        ) from error

    return PrecisionPolicy(
        kind=kind,
        scan_interval_tokens=interval,
        arm_min_requests=arm_min,
        capture_max_batch=capture_max,
        commitment_enabled=commitment_enabled,
        receding_horizon_lookup=receding,
        initial_rollout_batch=initial_rollout_batch,
        max_switch_live_batch=max_switch_live_batch,
        policy_revision=revision,
        table=table,
        fixed_switch_frontier=fixed_switch_frontier,
        cost_model=cost_model,
        schema_version=schema_version,
        description=str(raw.get("description", "")),
        source=source,
    )


def _response_cap_hint(raw: dict[str, Any], minimum: int) -> int:
    offline = raw.get("offline_cost_model") or {}
    cost = raw.get("cost_model") or {}
    for container in (offline, cost):
        if isinstance(container, dict) and container.get("response_cap"):
            return max(int(container["response_cap"]), minimum + 1)
    return DEFAULT_SPEC_RESPONSE_CAP


def _policy_from_spec(spec: PolicySpec) -> PrecisionPolicy:
    interval = DEFAULT_SCAN_INTERVAL_TOKENS
    common: dict[str, Any] = {
        "scan_interval_tokens": interval,
        "arm_min_requests": 1,
        # Inline specs know no rollout cohort: cohort-free arming (the
        # scheduler arms on the first unfinished request), no commitment
        # bookkeeping beyond the degenerate table.
        "commitment_enabled": False,
        "receding_horizon_lookup": False,
        "initial_rollout_batch": None,
        "policy_revision": 0,
        "schema_version": None,
        "source": str(spec),
    }
    if spec.kind == KIND_UNIFORM_W4:
        return PrecisionPolicy(
            kind=KIND_UNIFORM_W4,
            capture_max_batch=UNBOUNDED_CAPTURE_MAX_BATCH,
            max_switch_live_batch=None,
            table=None,
            description="INT4 base precision from the first token",
            **common,
        )
    assert spec.value is not None
    if spec.kind == KIND_FIXED_FRONTIER:
        return PrecisionPolicy(
            kind=KIND_FIXED_FRONTIER,
            capture_max_batch=DEFAULT_CAPTURE_MAX_BATCH,
            max_switch_live_batch=None,
            table=LookupTable.constant(
                spec.value,
                scan_interval_tokens=interval,
                response_cap=DEFAULT_SPEC_RESPONSE_CAP,
            ),
            fixed_switch_frontier=spec.value,
            description=f"switch every request at {spec.value} response tokens",
            **common,
        )
    if spec.kind == KIND_FIXED_THRESHOLD:
        return PrecisionPolicy(
            kind=KIND_FIXED_THRESHOLD,
            capture_max_batch=max(DEFAULT_CAPTURE_MAX_BATCH, spec.value),
            max_switch_live_batch=spec.value,
            table=LookupTable.constant(
                interval,
                scan_interval_tokens=interval,
                response_cap=DEFAULT_SPEC_RESPONSE_CAP,
            ),
            description=f"switch when the live batch drains to {spec.value}",
            **common,
        )
    raise ValueError(f"Unsupported precision policy spec kind {spec.kind!r}")


def load_precision_policy(spec: str | PolicySpec) -> PrecisionPolicy:
    """Load any policy kind from the decision-6 spec string or a JSON path."""
    parsed = spec if isinstance(spec, PolicySpec) else PolicySpec.parse(spec)
    if not parsed.enabled:
        raise ValueError("Precision policy spec is empty (policy disabled)")
    if parsed.path is not None:
        return policy_from_json(_read_policy_file(parsed.path), source=parsed.path)
    return _policy_from_spec(parsed)


# ---------------------------------------------------------------------------
# Runtime decider
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decision:
    """Outcome of one ``PolicyDecider.observe`` call.

    ``switch_now`` is True exactly on the call that flips the rollout to
    INT4.  ``committed_frontier`` is the current planned switch frontier
    (``None`` when no switch is planned).  ``reason`` is one of the
    ``REASON_*`` constants.  ``frontier`` is the observed scan-grid frontier
    when this call was a new observation (``observed``), else ``None``;
    ``candidate_frontier`` / ``previous_frontier`` describe the table lookup
    of that observation.
    """

    committed_frontier: int | None
    switch_now: bool
    reason: str
    observed: bool = False
    frontier: int | None = None
    candidate_frontier: int | None = None
    previous_frontier: int | None = None


class PolicyDecider:
    """Per-rollout switching state machine driven by scheduler observations.

    The scheduler owns cohort bookkeeping (arrival sets, rollout index, the
    response watermark, the median prompt, decision-live inflation for
    not-yet-arrived cohort members) and calls ``observe`` on every
    ``schedule()`` once the cohort is armed.  Non-crossing calls are O(1):
    only the first call at each new scan-grid frontier performs a lookup.

    Commitment modes:

    * ``monotone`` (``receding_horizon_lookup=False``): a candidate can only
      move the committed frontier earlier (``min``); the switch predicate is
      evaluated on every call, so a guard-blocked switch fires as soon as
      the live batch drains.
    * ``receding`` (``receding_horizon_lookup=True``): the candidate replaces
      the commitment at every observed frontier (``None`` clears it); the
      switch predicate is evaluated only at observation boundaries against
      the freshly looked-up candidate.

    Switch predicate: ``max_response_tokens >= committed`` and
    ``actual_live <= policy.switch_live_cap``.  The table is indexed with
    ``decision_live`` (padded), the guard uses ``actual_live``.
    """

    def __init__(self, policy: PrecisionPolicy) -> None:
        if not policy.schedulable:
            raise ValueError(
                "cost_model-only policies are offline oracles and cannot drive "
                "the scheduler (decision 4 dropped the online cost loop)"
            )
        self.policy = policy
        self.last_frontier = 0
        self.committed_frontier: int | None = None
        self.switched = False
        self.commitment_armed = False
        self.observations = 0
        self.reset()

    def reset(self) -> None:
        """Start a new rollout."""
        self.last_frontier = 0
        self.committed_frontier = None
        self.switched = self.policy.kind == KIND_UNIFORM_W4
        self.commitment_armed = False
        self.observations = 0

    @property
    def base_precision(self) -> str:
        return BASE_PRECISION_INT4 if self.switched else BASE_PRECISION_BF16

    @property
    def receding(self) -> bool:
        return self.policy.receding_horizon_lookup

    def guard_allows(self, actual_live: int) -> bool:
        return actual_live <= self.policy.switch_live_cap

    def quantize_frontier(self, tokens: int) -> int:
        step = self.policy.scan_interval_tokens
        return (int(tokens) // step) * step

    def _switch_reached(self, max_response_tokens: int) -> bool:
        return (
            self.committed_frontier is not None
            and max_response_tokens >= self.committed_frontier
        )

    def observe(
        self,
        frontier_tokens: int,
        prompt_tokens_median: float,
        decision_live: int,
        actual_live: int,
        max_response_tokens: int,
    ) -> Decision:
        if self.switched:
            return Decision(self.committed_frontier, False, REASON_ALREADY_SWITCHED)
        table = self.policy.table
        assert table is not None

        # Monotone mode evaluates the switch on every call, before the
        # observation gate, so a guard-blocked switch fires on drain.
        if (
            not self.receding
            and self._switch_reached(max_response_tokens)
            and self.guard_allows(actual_live)
        ):
            self.switched = True
            return Decision(self.committed_frontier, True, REASON_SWITCH)

        frontier = self.quantize_frontier(frontier_tokens)
        if frontier <= self.last_frontier:
            reason = REASON_NOT_OBSERVED
            if not self.receding and self._switch_reached(max_response_tokens):
                reason = REASON_GUARD_BLOCKED
            return Decision(self.committed_frontier, False, reason)
        self.last_frontier = frontier
        self.observations += 1
        self.commitment_armed = True

        previous = self.committed_frontier
        candidate = table.committed_frontier(
            frontier, prompt_tokens_median, decision_live
        )
        if self.receding:
            self.committed_frontier = candidate
        elif candidate is not None:
            self.committed_frontier = (
                candidate if previous is None else min(previous, candidate)
            )
        committed = self.committed_frontier

        def decision(switch_now: bool, reason: str) -> Decision:
            return Decision(
                committed_frontier=committed,
                switch_now=switch_now,
                reason=reason,
                observed=True,
                frontier=frontier,
                candidate_frontier=candidate,
                previous_frontier=previous,
            )

        if committed is None:
            return decision(False, REASON_NO_COMMITMENT)
        if max_response_tokens >= committed:
            if self.guard_allows(actual_live):
                self.switched = True
                return decision(True, REASON_SWITCH)
            return decision(False, REASON_GUARD_BLOCKED)
        return decision(False, REASON_COMMITTED)


# ---------------------------------------------------------------------------
# Revision-checked policy store (online calibration reload)
# ---------------------------------------------------------------------------


class PolicyStore:
    """Load a policy once and reload it between rollouts with a revision check.

    The online calibrator rewrites the policy JSON atomically and bumps
    ``calibration.policy_revision``.  ``reload`` re-reads the file and:

    * raises ``PolicyRevisionError`` (keeping the previous policy installed)
      when the file is unreadable/invalid or its revision went backwards;
    * with ``require_advance=True`` also raises when the revision did not
      advance (fail closed: a stale table must not silently drive the next
      rollout);
    * otherwise installs the new policy (or keeps the identical one).

    Inline specs have no file; ``reload`` returns the same policy.
    """

    def __init__(
        self, spec: str | PolicySpec, *, require_advance: bool = False
    ) -> None:
        self.spec = spec if isinstance(spec, PolicySpec) else PolicySpec.parse(spec)
        self.require_advance = require_advance
        self._policy: PrecisionPolicy | None = None
        self.reload_count = 0
        self.last_reload_advanced = False

    @property
    def policy(self) -> PrecisionPolicy:
        if self._policy is None:
            return self.load()
        return self._policy

    @property
    def revision(self) -> int:
        return self.policy.policy_revision

    def load(self) -> PrecisionPolicy:
        self._policy = load_precision_policy(self.spec)
        return self._policy

    def reload(self) -> PrecisionPolicy:
        current = self.policy
        if not self.spec.is_file:
            self.last_reload_advanced = False
            return current
        try:
            candidate = load_precision_policy(self.spec)
        except ValueError as error:
            raise PolicyRevisionError(
                f"policy reload failed; keeping revision "
                f"{current.policy_revision}: {error}"
            ) from error
        if candidate.policy_revision < current.policy_revision:
            raise PolicyRevisionError(
                f"policy revision went backwards: {current.policy_revision} -> "
                f"{candidate.policy_revision} ({self.spec})"
            )
        advanced = candidate.policy_revision > current.policy_revision
        if not advanced and self.require_advance:
            raise PolicyRevisionError(
                f"policy revision did not advance from "
                f"{current.policy_revision} ({self.spec})"
            )
        self.reload_count += 1
        self.last_reload_advanced = advanced
        if advanced:
            self._policy = candidate
        return current if not advanced else candidate

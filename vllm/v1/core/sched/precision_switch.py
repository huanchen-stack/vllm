# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side runtime that turns a switching policy into a per-step signal.

``RolloutPrecisionSwitcher`` is a pure state machine (no scheduler imports,
no environment reads) that the scheduler drives with three hooks and one
``tick`` per ``schedule()``:

* ``on_new_request(request_id)`` from ``add_request`` for genuinely new
  request ids (re-admitted resumable-streaming continuations carry the id of
  a request that already belongs to the rollout and are ignored);
* ``on_request_output(request_id, output_tokens)`` from output processing,
  so the per-rollout response watermark advances even when a request crosses
  a frontier on its final token and is freed before the next ``schedule()``;
* ``on_request_finished(request_id, output_tokens)`` when a request is freed;
* ``tick(step)`` once per ``schedule()`` with a snapshot of the live requests;
  returns ``"bf16"`` or ``"int4"`` for this step.

Rollouts are tracked as cohorts.  With a policy that carries
``initial_rollout_batch`` (every calibrated table) the cohort is exactly that
many request ids: the first genuinely new id after a complete cohort is the
boundary of the next rollout, the policy is reloaded there (once), and
not-yet-arrived cohort members inflate the live count used to index the table
(``decision_live``) so asynchronous admission cannot make a B64 rollout look
like B8 at the first observation.  Inline specs (``fixed_frontier:K``,
``fixed_threshold:t``, ``uniform_w4``) carry no cohort: the rollout is armed
at the first ``tick`` that sees at least ``arm_min_requests`` unfinished
requests, ends when the scheduler drains, and ``decision_live`` is the actual
live count.

The decision itself (table lookup, commitment mode, live-batch guard, one-way
latch) lives in ``precision_policy.PolicyDecider``; this module owns cohort
bookkeeping, the watermark, the median prompt, the switch-cohort JSONL, the
reload-at-boundary contract and the profiler override seam.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from vllm.logger import init_logger
from vllm.v1.core.sched.precision_policy import (
    BASE_PRECISION_BF16,
    BASE_PRECISION_INT4,
    Decision,
    PolicyDecider,
    PolicyRevisionError,
    PolicyStore,
    PrecisionPolicy,
)

logger = init_logger(__name__)

__all__ = [
    "BASE_PRECISION_BF16",
    "BASE_PRECISION_INT4",
    "COHORT_EVENT",
    "CohortEntry",
    "LiveRequest",
    "RolloutPrecisionSwitcher",
    "SchedulerStep",
    "SwitchEvent",
    "SwitchLogWriter",
    "parse_cohort_line",
]

#: ``event`` value of every switch-cohort JSONL record.
COHORT_EVENT = "switch_cohort"

VALID_PRECISIONS = (BASE_PRECISION_BF16, BASE_PRECISION_INT4)


@dataclass(frozen=True)
class LiveRequest:
    """One unfinished request as seen by the scheduler at a tick."""

    request_id: str
    prompt_tokens: int
    #: Cumulative response length (``Request.num_cumulative_output_tokens``).
    output_tokens: int


@dataclass(frozen=True)
class SchedulerStep:
    """Snapshot handed to ``tick``.

    ``live`` holds every request that is unfinished and not waiting for its
    next streaming chunk; ``unfinished`` is the scheduler's own count
    (``get_num_unfinished_requests``), which can be zero while ``live`` is
    empty only transiently (all requests between streaming chunks).
    """

    live: Sequence[LiveRequest]
    unfinished: int


@dataclass(frozen=True)
class CohortEntry:
    request_id: str
    entry_output_tokens: int
    prompt_tokens: int


@dataclass(frozen=True)
class SwitchEvent:
    """The state at the tick that flipped a rollout to INT4."""

    rollout_index: int
    committed_frontier: int
    applied_response_tokens: int
    applied_live_requests: int
    decision_live_requests: int
    median_prompt_tokens: float
    reason: str
    policy_kind: str
    policy_revision: int
    cohort: tuple[CohortEntry, ...]
    #: The reload before this rollout saw an unchanged revision (calibrator
    #: lagged a boundary); the switch ran on the previous table.
    policy_reload_lagged: bool = False

    def to_record(self) -> dict[str, Any]:
        """The switch-cohort JSONL record (schema below, additive over the
        archived ``{event, rollout_index, requests[{request_id,
        entry_output_tokens}]}`` layout that the online calibrator reads)."""
        return {
            "event": COHORT_EVENT,
            "rollout_index": self.rollout_index,
            "policy_kind": self.policy_kind,
            "policy_revision": self.policy_revision,
            "policy_reload_lagged": self.policy_reload_lagged,
            "trigger": {
                "committed_frontier": self.committed_frontier,
                "applied_response_tokens": self.applied_response_tokens,
                "applied_live_requests": self.applied_live_requests,
                "decision_live_requests": self.decision_live_requests,
                "median_prompt_tokens": self.median_prompt_tokens,
                "reason": self.reason,
            },
            "requests": [
                {
                    "request_id": entry.request_id,
                    "entry_output_tokens": entry.entry_output_tokens,
                    "prompt_tokens": entry.prompt_tokens,
                }
                for entry in self.cohort
            ],
        }


def parse_cohort_line(line: str) -> dict[str, Any] | None:
    """Parse one JSONL line; ``None`` for blank lines or other events."""
    text = line.strip()
    if not text:
        return None
    record = json.loads(text)
    if record.get("event") != COHORT_EVENT:
        return None
    return record


class SwitchLogWriter:
    """Append switch-cohort records to a JSONL file (one object per line,
    ``sort_keys=True``, directory created on first write)."""

    def __init__(self, path: str) -> None:
        if not path:
            raise ValueError("SwitchLogWriter needs a non-empty path")
        self.path = path
        self.records_written = 0

    def write(self, event: SwitchEvent) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps(event.to_record(), sort_keys=True) + "\n")
        self.records_written += 1


@dataclass
class _RolloutState:
    """Everything that resets at a rollout boundary."""

    armed: bool = False
    arrival_ids: set[str] = field(default_factory=set)
    rollout_ids: set[str] = field(default_factory=set)
    watermark: int = 0
    switch: SwitchEvent | None = None
    observations: int = 0
    reload_lagged: bool = False


class RolloutPrecisionSwitcher:
    """Per-rollout precision state machine.  See the module docstring."""

    def __init__(
        self,
        store: PolicyStore,
        *,
        reload_each_rollout: bool = False,
        observations_path: str | None = None,
        writer: SwitchLogWriter | None = None,
        log: Callable[..., None] | None = None,
    ) -> None:
        self.store = store
        self.reload_each_rollout = reload_each_rollout
        if writer is None and observations_path:
            writer = SwitchLogWriter(observations_path)
        self.writer = writer
        self._log = log if log is not None else logger.info
        self.decider = PolicyDecider(store.policy)
        self.rollout_index = 0
        self.forced_precision: str | None = None
        self.ticks = 0
        self.reloads: list[tuple[int, int]] = []
        #: Boundaries (rollout >= 2) whose reload saw an unchanged revision.
        self.policy_reload_lag_count = 0
        self.switches: list[SwitchEvent] = []
        self._state = _RolloutState()

    @classmethod
    def from_settings(
        cls,
        policy_spec: str,
        *,
        reload_each_rollout: bool = False,
        require_advance: bool = False,
        observations_path: str = "",
        log: Callable[..., None] | None = None,
    ) -> RolloutPrecisionSwitcher:
        """Build from the decision-6 spec string and the runtime knobs (the
        scheduler passes ``envs.*`` values through; nothing here reads the
        environment).  ``require_advance`` is the strict fallback: fail
        closed on an unchanged revision instead of logging a lag."""
        store = PolicyStore(policy_spec, require_advance=require_advance)
        store.load()
        return cls(
            store,
            reload_each_rollout=reload_each_rollout,
            observations_path=observations_path or None,
            log=log,
        )

    # ----------------------------------------------------------------- policy

    @property
    def policy(self) -> PrecisionPolicy:
        return self.decider.policy

    @property
    def cohort_size(self) -> int | None:
        return self.policy.initial_rollout_batch

    @property
    def armed(self) -> bool:
        return self._state.armed

    @property
    def switched(self) -> bool:
        return self.decider.switched

    @property
    def committed_frontier(self) -> int | None:
        return self.decider.committed_frontier

    @property
    def watermark(self) -> int:
        return self._state.watermark

    @property
    def arrival_ids(self) -> frozenset[str]:
        return frozenset(self._state.arrival_ids)

    @property
    def rollout_ids(self) -> frozenset[str]:
        return frozenset(self._state.rollout_ids)

    @property
    def last_switch(self) -> SwitchEvent | None:
        return self._state.switch

    @property
    def base_precision(self) -> str:
        if self.forced_precision is not None:
            return self.forced_precision
        return self.decider.base_precision

    def set_forced_precision(self, precision: str | None) -> None:
        """Profiler seam: pin the reported precision regardless of policy
        state (``None`` restores policy control).  The state machine keeps
        running so cohort logs stay meaningful."""
        if precision is not None and precision not in VALID_PRECISIONS:
            raise ValueError(
                f"forced precision must be one of {VALID_PRECISIONS} or None, "
                f"got {precision!r}"
            )
        self.forced_precision = precision

    # ------------------------------------------------------------- boundaries

    def start_rollout(self) -> None:
        """Begin the next rollout: reload the policy once (when enabled),
        reset every per-rollout state, arm.

        The reload before rollout 1 re-reads the file just loaded (the
        archived runs log "Reloaded ... before rollout 1: revision=0"); the
        store exempts that first reload from the advance check, and every
        later boundary must see a higher revision or fail closed.
        """
        lagged = False
        if self.reload_each_rollout and self.store.spec.is_file:
            lagged = self._reload_policy(self.rollout_index + 1)
        self.end_rollout()
        self.rollout_index += 1
        self._state.armed = True
        self._state.reload_lagged = lagged
        self._log(
            "Precision rollout armed: rollout_index=%d, policy=%s, "
            "cohort_size=%s, revision=%d",
            self.rollout_index,
            self.policy.kind,
            self.cohort_size,
            self.policy.policy_revision,
        )

    def end_rollout(self) -> None:
        """Disarm and clear per-rollout state without touching the policy."""
        self._state = _RolloutState()
        self.decider.reset()
        policy = self.policy
        if (
            policy.fixed_switch_frontier is not None
            and not policy.receding_horizon_lookup
        ):
            # A fixed frontier needs no table observation to be known: seed
            # the monotone commitment so ``fixed_frontier:K`` switches at K
            # even when K is below the first scan-grid frontier (the table
            # would otherwise only commit at the 250-token observation, i.e.
            # switch at max(K, 250)). For K >= 250 the seeded value equals
            # the first lookup, so archived behaviour is unchanged.
            self.decider.committed_frontier = policy.fixed_switch_frontier

    def _reload_policy(self, next_rollout_index: int) -> bool:
        """Reload once; returns True when the revision did not advance at a
        boundary from rollout 2 on (calibrator lag).  With
        ``store.require_advance`` that case raises instead (strict fallback);
        an invalid file or a revision going backwards always raises."""
        try:
            self.store.reload()
        except PolicyRevisionError as error:
            raise PolicyRevisionError(
                f"policy reload before rollout {next_rollout_index} failed: {error}"
            ) from error
        if self.store.policy is not self.decider.policy:
            self.decider = PolicyDecider(self.store.policy)
        revision = self.policy.policy_revision
        self.reloads.append((next_rollout_index, revision))
        lagged = next_rollout_index >= 2 and not self.store.last_reload_advanced
        if lagged:
            self.policy_reload_lag_count += 1
            self._log(
                "Precision policy reload lagged before rollout %d: revision=%d "
                "unchanged (calibrator did not finish); running on the previous "
                "table (lag_count=%d)",
                next_rollout_index,
                revision,
                self.policy_reload_lag_count,
            )
        self._log(
            "Reloaded dynamic precision policy before rollout %d: revision=%d",
            next_rollout_index,
            revision,
        )
        return lagged

    # ------------------------------------------------------------------ hooks

    def on_new_request(self, request_id: str) -> None:
        state = self._state
        if request_id in state.rollout_ids:
            # A resumable streaming request can be freed at a chunk boundary
            # and re-admitted with the same id; it stays in this rollout.
            return
        cohort = self.cohort_size
        if cohort is None:
            # Cohort-free arming happens in tick(); nothing to record here.
            return
        if len(state.arrival_ids) >= cohort or not state.armed:
            # First id after a complete cohort (or the very first id) is the
            # boundary of the next rollout.  Waiting for the whole cohort
            # would let early arrivals decode past a frontier without a
            # commitment.
            if state.arrival_ids and len(state.arrival_ids) < cohort:
                raise RuntimeError(
                    "precision switcher: rollout disarmed with a partial cohort"
                )
            self.start_rollout()
            state = self._state
        state.arrival_ids.add(request_id)
        state.rollout_ids.add(request_id)
        if len(state.arrival_ids) == cohort:
            self._log(
                "Precision rollout cohort complete: rollout_index=%d, requests=%d",
                self.rollout_index,
                cohort,
            )

    def on_request_output(self, request_id: str, output_tokens: int) -> None:
        if self._state.armed and output_tokens > self._state.watermark:
            self._state.watermark = output_tokens

    def on_request_finished(self, request_id: str, output_tokens: int = 0) -> None:
        self.on_request_output(request_id, output_tokens)

    # ------------------------------------------------------------------- tick

    def tick(self, step: SchedulerStep) -> str:
        self.ticks += 1
        self._advance(step)
        return self.base_precision

    def _advance(self, step: SchedulerStep) -> None:
        state = self._state
        live = step.live
        live_ids = {request.request_id for request in live}
        cohort = self.cohort_size

        if step.unfinished == 0 and not live:
            if cohort is not None and state.arrival_ids:
                # A temporarily empty scheduler does not end a cohort rollout:
                # the cohort may still be arriving, or every resumable request
                # is between streaming chunks.  Keep commitment and watermark.
                return
            if state.armed:
                self.end_rollout()
            return

        if cohort is None:
            if (
                state.armed
                and state.rollout_ids
                and live_ids
                and state.rollout_ids.isdisjoint(live_ids)
            ):
                # The next batch arrived before the scheduler ever looked
                # empty (back-to-back rollouts): boundary.
                self.start_rollout()
                state = self._state
            if not state.armed and step.unfinished >= self.policy.arm_min_requests:
                self.start_rollout()
                state = self._state
                state.rollout_ids.update(live_ids)
            elif state.armed:
                state.rollout_ids.update(live_ids)

        if not state.armed or self.decider.switched:
            return

        decision_live = len(live)
        if cohort is not None:
            arrived = len(state.arrival_ids)
            if arrived < cohort:
                decision_live += cohort - arrived
        max_response = state.watermark
        for request in live:
            if request.output_tokens > max_response:
                max_response = request.output_tokens
        state.watermark = max_response
        prompts = sorted(request.prompt_tokens for request in live)
        median_prompt = float(prompts[len(prompts) // 2]) if prompts else 0.0

        decision = self.decider.observe(
            frontier_tokens=max_response,
            prompt_tokens_median=median_prompt,
            decision_live=decision_live,
            actual_live=len(live),
            max_response_tokens=max_response,
        )
        if decision.observed:
            state.observations += 1
            self._log_observation(decision, decision_live, median_prompt)
        if decision.switch_now:
            assert decision.committed_frontier is not None
            self._record_switch(
                decision, live, max_response, decision_live, median_prompt
            )

    def _log_observation(
        self, decision: Decision, decision_live: int, median_prompt: float
    ) -> None:
        if decision.candidate_frontier == decision.previous_frontier:
            return
        if self.policy.receding_horizon_lookup:
            self._log(
                "Dynamic precision receding lookup updated: rollout_index=%d, "
                "observation_frontier=%d, live_requests=%d, "
                "median_prompt_tokens=%.1f, candidate_frontier=%s, "
                "previous_frontier=%s",
                self.rollout_index,
                decision.frontier,
                decision_live,
                median_prompt,
                decision.candidate_frontier,
                decision.previous_frontier,
            )
        elif decision.committed_frontier != decision.previous_frontier:
            self._log(
                "Dynamic precision lookup commitment updated: rollout_index=%d, "
                "observation_frontier=%d, live_requests=%d, "
                "median_prompt_tokens=%.1f, candidate_frontier=%s, "
                "previous_frontier=%s, committed_frontier=%s",
                self.rollout_index,
                decision.frontier,
                decision_live,
                median_prompt,
                decision.candidate_frontier,
                decision.previous_frontier,
                decision.committed_frontier,
            )

    def _record_switch(
        self,
        decision: Decision,
        live: Sequence[LiveRequest],
        max_response: int,
        decision_live: int,
        median_prompt: float,
    ) -> None:
        assert decision.committed_frontier is not None
        event = SwitchEvent(
            rollout_index=self.rollout_index,
            committed_frontier=decision.committed_frontier,
            applied_response_tokens=max_response,
            applied_live_requests=len(live),
            decision_live_requests=decision_live,
            median_prompt_tokens=median_prompt,
            reason=decision.reason,
            policy_kind=self.policy.kind,
            policy_revision=self.policy.policy_revision,
            policy_reload_lagged=self._state.reload_lagged,
            cohort=tuple(
                CohortEntry(
                    request_id=request.request_id,
                    entry_output_tokens=request.output_tokens,
                    prompt_tokens=request.prompt_tokens,
                )
                for request in live
            ),
        )
        self._state.switch = event
        self.switches.append(event)
        if self.writer is not None:
            self.writer.write(event)
        # Log-line contract kept verbatim: the archived audits parse it.
        self._log(
            "Lookup dynamic full-cost switch: rollout_index=%d, "
            "committed_frontier=%d, applied_response_tokens=%d, "
            "applied_live_requests=%d",
            event.rollout_index,
            event.committed_frontier,
            event.applied_response_tokens,
            event.applied_live_requests,
        )

    # ------------------------------------------------------------ inspection

    def describe(self) -> dict[str, Any]:
        state = self._state
        return {
            "rollout_index": self.rollout_index,
            "armed": state.armed,
            "switched": self.decider.switched,
            "committed_frontier": self.decider.committed_frontier,
            "watermark": state.watermark,
            "arrived": len(state.arrival_ids),
            "cohort_size": self.cohort_size,
            "policy_kind": self.policy.kind,
            "policy_revision": self.policy.policy_revision,
            "policy_reload_lagged": state.reload_lagged,
            "policy_reload_lag_count": self.policy_reload_lag_count,
            "forced_precision": self.forced_precision,
            "base_precision": self.base_precision,
        }

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side precision switching runtime (component C4).

Three tiers:

* unit tests of the pure ``RolloutPrecisionSwitcher`` (fixed latch, uniform,
  drain re-arm, env-threshold vs lookup-emulation equivalence, cohorts under
  asynchronous admission, watermark, receding/monotone commitment, guard,
  cohort JSONL round trip, reload once per boundary, forced precision);
* scheduler integration through the public
  ``SchedulerOutput.dual_precision_base_precision`` field (the seven archived
  scenarios re-expressed on fixed-frontier / tiny lookup policies) plus the
  vanilla regression with the policy flag empty;
* golden replays of archived rollout lifetimes against the archived switch
  lines and cohorts (trimmed fixtures under ``golden/precision_switch``).
"""

from __future__ import annotations

import json
import random
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import pytest

import vllm.envs as envs
from vllm.v1.core.sched.precision_policy import (
    KIND_FIXED_FRONTIER,
    PolicyRevisionError,
    PolicyStore,
)
from vllm.v1.core.sched.precision_switch import (
    BASE_PRECISION_BF16,
    BASE_PRECISION_INT4,
    COHORT_EVENT,
    LiveRequest,
    RolloutPrecisionSwitcher,
    SchedulerStep,
    SwitchLogWriter,
    parse_cohort_line,
)
from vllm.v1.request import RequestStatus, StreamingUpdate

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

GOLDEN = Path(__file__).resolve().parent / "golden" / "precision_switch"

BF16 = BASE_PRECISION_BF16
INT4 = BASE_PRECISION_INT4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _table(
    cells: list[int],
    *,
    frontier_start=250,
    frontier_step=250,
    prompt_count=1,
    live_count=1,
):
    frontier_count = len(cells) // (prompt_count * live_count)
    assert frontier_count * prompt_count * live_count == len(cells)
    return {
        "layout": "frontier_major,prompt_bucket,live_batch",
        "frontier_start": frontier_start,
        "frontier_step": frontier_step,
        "frontier_count": frontier_count,
        "prompt_bucket_start": 0,
        "prompt_bucket_step": 128,
        "prompt_bucket_count": prompt_count,
        "live_batch_start": 1,
        "live_batch_count": live_count,
        "committed_frontiers": cells,
    }


def _policy_json(table: dict, **overrides) -> dict:
    raw = {
        "schema_version": 6,
        "description": "unit test policy",
        "scan_interval_tokens": table["frontier_step"],
        "arm_min_requests": 1,
        "capture_max_batch": 32,
        "commitment_enabled": True,
        "receding_horizon_lookup": False,
        "initial_rollout_batch": None,
        "max_switch_live_batch": None,
        "calibration": {"kind": "unit", "policy_revision": 0},
        "lookup_table": table,
    }
    raw.update(overrides)
    return raw


def _write_policy(tmp_path: Path, raw: dict, name: str = "policy.json") -> str:
    path = tmp_path / name
    path.write_text(json.dumps(raw))
    return str(path)


def _fixed_frontier_json(
    frontier: int, *, batch: int, cap: int = 4000, **overrides
) -> dict:
    """The archived build_fixed_frontier_policy.py shape: every cell at or
    before ``frontier`` is ``frontier``, later cells 0."""
    frontiers = list(range(250, cap, 250))
    cells = [frontier if f <= frontier else 0 for f in frontiers]
    return _policy_json(
        _table(cells),
        arm_min_requests=batch,
        initial_rollout_batch=batch,
        **overrides,
    )


def _switcher(spec: str, **kwargs) -> RolloutPrecisionSwitcher:
    return RolloutPrecisionSwitcher.from_settings(
        spec, log=lambda *a, **k: None, **kwargs
    )


def _live(*items: tuple[str, int, int]) -> list[LiveRequest]:
    return [LiveRequest(rid, prompt, out) for rid, prompt, out in items]


def _step(live: list[LiveRequest], unfinished: int | None = None) -> SchedulerStep:
    return SchedulerStep(
        live=live, unfinished=len(live) if unfinished is None else unfinished
    )


@dataclass
class Lifetime:
    request_id: str
    prompt_tokens: int
    generation_tokens: int


def lockstep_replay(
    switcher: RolloutPrecisionSwitcher,
    rollouts: list[list[Lifetime]],
    *,
    arrive_all_first: bool = True,
) -> list[list[str]]:
    """Drive the switcher in lockstep decode order.

    Every live request produces one token per tick; at tick ``t`` a request
    with ``generation_tokens > t`` is live with ``t`` response tokens, and
    output processing after the tick advances the watermark to ``t + 1``.
    Returns the per-tick precision of every rollout.
    """
    history: list[list[str]] = []
    for rollout in rollouts:
        if arrive_all_first:
            for item in rollout:
                switcher.on_new_request(item.request_id)
        precisions: list[str] = []
        longest = max(item.generation_tokens for item in rollout)
        for tick in range(longest):
            live = [
                LiveRequest(item.request_id, item.prompt_tokens, tick)
                for item in rollout
                if item.generation_tokens > tick
            ]
            precisions.append(switcher.tick(_step(live)))
            for request in live:
                switcher.on_request_output(request.request_id, tick + 1)
        history.append(precisions)
    return history


# ---------------------------------------------------------------------------
# Unit: inline kinds (fixed latch, uniform, drain re-arm)
# ---------------------------------------------------------------------------


def test_fixed_frontier_latches_one_way_and_rearms_after_drain():
    switcher = _switcher("fixed_frontier:1000")
    assert switcher.policy.kind == KIND_FIXED_FRONTIER
    assert switcher.tick(_step([])) == BF16  # nothing live: no rollout
    assert not switcher.armed
    live = _live(("a", 50, 0), ("b", 60, 0))
    assert switcher.tick(_step(live)) == BF16
    assert switcher.armed and switcher.rollout_index == 1
    assert switcher.tick(_step(_live(("a", 50, 999), ("b", 60, 999)))) == BF16
    assert switcher.tick(_step(_live(("a", 50, 1000), ("b", 60, 999)))) == INT4
    event = switcher.last_switch
    assert event is not None
    assert (
        event.committed_frontier,
        event.applied_response_tokens,
        event.applied_live_requests,
    ) == (1000, 1000, 2)
    # One-way for the rest of the rollout, even after admission changes.
    assert switcher.tick(_step(_live(("a", 50, 1001), ("c", 10, 0)))) == INT4
    assert switcher.tick(_step(_live(("c", 10, 5)))) == INT4
    # Drain: back to BF16 and the next batch re-arms and can switch again.
    assert switcher.tick(_step([], unfinished=0)) == BF16
    assert not switcher.armed
    assert switcher.tick(_step(_live(("d", 10, 0)))) == BF16
    assert switcher.rollout_index == 2
    assert switcher.tick(_step(_live(("d", 10, 1000)))) == INT4
    assert [event.rollout_index for event in switcher.switches] == [1, 2]


def test_fixed_frontier_below_the_scan_grid_switches_at_the_frontier(tmp_path):
    """``fixed_frontier:32``: the commitment is seeded from the spec, so the
    switch does not wait for the first 250-token table observation."""
    switcher = _switcher("fixed_frontier:32")
    live = _live(("a", 10, 0), ("b", 10, 0))
    assert switcher.tick(_step(live)) == BF16
    assert switcher.committed_frontier == 32
    assert switcher.tick(_step(_live(("a", 10, 31), ("b", 10, 31)))) == BF16
    assert switcher.tick(_step(_live(("a", 10, 32), ("b", 10, 31)))) == INT4
    assert switcher.last_switch is not None
    assert switcher.last_switch.applied_response_tokens == 32
    # A JSON policy with fixed_switch_frontier behaves the same way.
    raw = {
        "schema_version": 6,
        "scan_interval_tokens": 250,
        "capture_max_batch": 32,
        "fixed_switch_frontier": 100,
        "offline_cost_model": {"response_cap": 2000},
    }
    switcher = _switcher(_write_policy(tmp_path, raw))
    switcher.tick(_step(live))
    assert switcher.committed_frontier == 100
    assert switcher.tick(_step(_live(("a", 10, 100)))) == INT4


def test_uniform_w4_is_int4_from_the_first_tick_and_never_reports_a_switch():
    switcher = _switcher("uniform_w4")
    assert switcher.tick(_step([], unfinished=0)) == INT4
    assert switcher.tick(_step(_live(("a", 10, 0)))) == INT4
    assert switcher.tick(_step(_live(("a", 10, 5000)))) == INT4
    assert switcher.switches == []
    assert switcher.tick(_step([], unfinished=0)) == INT4


class FixedThresholdLatch:
    """Reference: the experimental env-threshold latch (scheduler-owned)."""

    def __init__(self, threshold: int, max_num_seqs: int) -> None:
        self.threshold = threshold
        self.uniform = threshold >= max_num_seqs
        self.armed = False
        self.switched = False

    def tick(self, live: int) -> str:
        if live == 0:
            self.armed = self.switched = False
            return BF16
        if self.switched:
            return INT4
        if live > self.threshold:
            self.armed = True
        if self.uniform or (self.armed and live <= self.threshold):
            self.switched = True
            return INT4
        return BF16


def _synthetic_rollouts(
    seed: int, batch: int, rollouts: int, low: int, high: int
) -> list[list[Lifetime]]:
    rng = random.Random(seed)
    result = []
    for index in range(rollouts):
        result.append(
            [
                Lifetime(f"r{index}-{i}", rng.randint(20, 200), rng.randint(low, high))
                for i in range(batch)
            ]
        )
    return result


@pytest.mark.parametrize("threshold", [2, 4, 8])
def test_fixed_threshold_matches_the_env_latch_on_synthetic_lifetimes(threshold):
    """``fixed_threshold:t`` through the lookup emulation (constant-250 table
    plus the live-batch guard, as the full-RL matrix emulated it) switches at
    the same tick as the experimental env latch whenever the live batch
    exceeded ``t`` before the first 250-token frontier, i.e. in every
    archived configuration (all responses > 250 tokens)."""
    rollouts = _synthetic_rollouts(
        seed=threshold, batch=64, rollouts=3, low=300, high=3000
    )
    switcher = _switcher(f"fixed_threshold:{threshold}")
    history = lockstep_replay(switcher, rollouts)
    for rollout, precisions in zip(rollouts, history, strict=True):
        latch = FixedThresholdLatch(threshold, max_num_seqs=64)
        reference = []
        longest = max(item.generation_tokens for item in rollout)
        for tick in range(longest):
            live = sum(1 for item in rollout if item.generation_tokens > tick)
            reference.append(latch.tick(live))
        latch.tick(0)  # drain
        assert precisions == reference
        assert INT4 in precisions
        switcher.tick(_step([], unfinished=0))  # drain between rollouts
    assert len(switcher.switches) == 3
    for event in switcher.switches:
        assert event.applied_live_requests <= threshold


def test_fixed_threshold_archived_scenario_switches_when_live_drains_to_t():
    """smollm3_3b_gsm8k_tail_w4_t8.log: live_requests=8, threshold=8,
    observed_max_response_tokens=1114 -> the switch fires at the first tick
    where the live batch is <= 8 after having exceeded it."""
    switcher = _switcher("fixed_threshold:8")
    live = [LiveRequest(f"q{i}", 40, 1113) for i in range(16)]
    assert switcher.tick(_step(live)) == BF16
    assert switcher.tick(_step(live[:9])) == BF16
    assert (
        switcher.tick(_step([LiveRequest(r.request_id, 40, 1114) for r in live[:8]]))
        == INT4
    )
    event = switcher.last_switch
    assert event is not None
    assert event.applied_live_requests == 8
    assert event.applied_response_tokens == 1114
    assert len(event.cohort) == 8


# ---------------------------------------------------------------------------
# Unit: cohorts, watermark, commitment, guard
# ---------------------------------------------------------------------------


def test_cohort_arms_on_first_arrival_and_pads_decision_live(tmp_path):
    # At frontier 1000: live 1 -> no switch planned, live 4 -> commit 2000.
    cells = [0, 0, 0, 0] * 3 + [0, 0, 0, 2000] + [0, 0, 0, 0] * 4
    raw = _policy_json(
        _table(cells, live_count=4), arm_min_requests=4, initial_rollout_batch=4
    )
    switcher = _switcher(_write_policy(tmp_path, raw))
    switcher.on_new_request("0")
    assert switcher.armed and switcher.rollout_index == 1
    # Only one request arrived, but the runtime knows this is a B4 rollout:
    # the lookup indexes live=4 rather than live=1.
    assert switcher.tick(_step(_live(("0", 30, 1000)))) == BF16
    assert switcher.committed_frontier == 2000
    for rid in ("1", "2", "3"):
        switcher.on_new_request(rid)
    assert switcher.arrival_ids == frozenset("0123")
    live = _live(("0", 30, 1000), ("1", 30, 1000), ("2", 30, 1000), ("3", 30, 1000))
    assert switcher.tick(_step(live)) == BF16
    live = _live(("1", 30, 2000), ("2", 30, 2000), ("3", 30, 2000))
    assert switcher.tick(_step(live)) == INT4
    assert switcher.last_switch is not None
    assert switcher.last_switch.decision_live_requests == 3


def test_cohort_survives_an_empty_scheduler_and_ignores_readmitted_ids(tmp_path):
    raw = _fixed_frontier_json(1000, batch=4)
    switcher = _switcher(_write_policy(tmp_path, raw))
    switcher.on_new_request("0")
    # The first session finished before the others were admitted: an empty
    # scheduler must not end the rollout or its commitment.
    assert switcher.tick(_step(_live(("0", 10, 900)))) == BF16
    assert switcher.committed_frontier == 1000
    assert switcher.tick(_step([], unfinished=0)) == BF16
    assert switcher.armed and switcher.rollout_index == 1
    assert switcher.arrival_ids == frozenset({"0"})
    for rid in ("1", "2", "3"):
        switcher.on_new_request(rid)
    # Re-admission of a resumable continuation with a known id is not a new
    # cohort member.
    switcher.on_new_request("1")
    assert switcher.rollout_index == 1
    assert len(switcher.arrival_ids) == 4
    assert switcher.tick(_step(_live(("1", 10, 1000), ("2", 10, 10)))) == INT4


def test_cohort_boundary_is_the_first_new_id_after_a_complete_cohort(tmp_path):
    raw = _fixed_frontier_json(1000, batch=2)
    switcher = _switcher(_write_policy(tmp_path, raw))
    switcher.on_new_request("a")
    switcher.on_new_request("b")
    assert switcher.tick(_step(_live(("a", 10, 1000), ("b", 10, 1000)))) == INT4
    # Next cohort arrives while the old requests linger (overlapping cleanup):
    # its first id both returns to BF16 and arms rollout 2.
    switcher.on_new_request("c")
    assert switcher.rollout_index == 2
    assert not switcher.switched and switcher.armed
    assert switcher.tick(_step(_live(("c", 10, 0)), unfinished=3)) == BF16
    switcher.on_new_request("d")
    assert switcher.rollout_index == 2
    assert switcher.tick(_step(_live(("c", 10, 1000), ("d", 10, 999)))) == INT4
    assert [e.rollout_index for e in switcher.switches] == [1, 2]
    with pytest.raises(RuntimeError):
        # A third id inside the second cohort would exceed the batch.
        switcher._state.arrival_ids.discard("d")
        switcher._state.armed = False
        switcher.on_new_request("e")


def test_watermark_keeps_a_frontier_crossed_by_a_request_freed_between_ticks(tmp_path):
    raw = _fixed_frontier_json(8000, batch=2, cap=12000)
    switcher = _switcher(_write_policy(tmp_path, raw))
    switcher.on_new_request("0")
    switcher.on_new_request("1")
    assert switcher.tick(_step(_live(("0", 10, 7999), ("1", 10, 7999)))) == BF16
    # Request 0 crosses 8K in output processing and is freed before the next
    # schedule(); the surviving request is still at 7999.
    switcher.on_request_output("0", 8000)
    switcher.on_request_finished("0", 8000)
    assert switcher.watermark == 8000
    assert switcher.tick(_step(_live(("1", 10, 7999)))) == INT4
    assert switcher.last_switch is not None
    assert switcher.last_switch.applied_response_tokens == 8000


def test_receding_lookup_can_postpone_a_committed_frontier(tmp_path):
    cells = []
    for candidate in (1000, 1250, 1500, 1500, 1500, 1500):
        cells.extend([0, 0, 0, candidate])
    raw = _policy_json(
        _table(cells, live_count=4),
        arm_min_requests=4,
        initial_rollout_batch=4,
        receding_horizon_lookup=True,
    )
    switcher = _switcher(_write_policy(tmp_path, raw))
    for rid in "0123":
        switcher.on_new_request(rid)
    live = [LiveRequest(rid, 10, 250) for rid in "0123"]
    assert switcher.tick(_step(live)) == BF16
    assert switcher.committed_frontier == 1000
    live = [LiveRequest(rid, 10, 1000) for rid in "0123"]
    assert switcher.tick(_step(live)) == BF16
    assert switcher.committed_frontier == 1500
    live = [LiveRequest(rid, 10, 1500) for rid in "0123"]
    assert switcher.tick(_step(live)) == INT4


def test_lookup_requires_deadline_and_live_batch_cap(tmp_path):
    cells = [0, 0, 0, 2000, 0, 0, 0, 2000]
    raw = _policy_json(
        _table(cells, frontier_start=1000, frontier_step=1000, live_count=4),
        arm_min_requests=4,
        initial_rollout_batch=4,
        max_switch_live_batch=2,
    )
    switcher = _switcher(_write_policy(tmp_path, raw))
    for rid in "0123":
        switcher.on_new_request(rid)
    live = [LiveRequest(rid, 10, 2000) for rid in "0123"]
    assert switcher.tick(_step(live)) == BF16
    assert switcher.committed_frontier == 2000
    assert switcher.tick(_step(live[2:])) == INT4
    assert switcher.last_switch is not None
    assert switcher.last_switch.applied_live_requests == 2


def test_forced_precision_overrides_the_policy_but_keeps_state_running():
    switcher = _switcher("fixed_frontier:1000")
    switcher.set_forced_precision(INT4)
    assert switcher.tick(_step(_live(("a", 10, 0)))) == INT4
    assert not switcher.switched
    assert switcher.tick(_step(_live(("a", 10, 1000)))) == INT4
    assert switcher.switched  # the state machine still switched underneath
    switcher.set_forced_precision(BF16)
    assert switcher.tick(_step(_live(("a", 10, 1001)))) == BF16
    switcher.set_forced_precision(None)
    assert switcher.tick(_step(_live(("a", 10, 1002)))) == INT4
    with pytest.raises(ValueError):
        switcher.set_forced_precision("fp8")


# ---------------------------------------------------------------------------
# Unit: cohort JSONL round trip and reload contract
# ---------------------------------------------------------------------------

ENGINE_SUFFIX_LENGTH = 8


def _watcher_read_cohorts(path: Path) -> list[dict]:
    """The online calibrator's reader (verl precision_scheduler.traces):
    one JSON object per line, ``event == "switch_cohort"``."""
    rows = []
    with path.open() as stream:
        for line in stream:
            record = parse_cohort_line(line)
            if record is not None:
                rows.append(record)
    return rows


def _watcher_resolve(request_id: str, finishes: dict[str, int]) -> str | None:
    if request_id in finishes:
        return request_id
    client_id, separator, suffix = request_id.rpartition("-")
    if separator and len(suffix) == ENGINE_SUFFIX_LENGTH and client_id in finishes:
        return client_id
    return None


def test_switch_cohort_jsonl_round_trips_through_the_watcher_parser(tmp_path):
    path = tmp_path / "obs" / "online_switch_cohorts.jsonl"
    switcher = _switcher("fixed_frontier:1000", observations_path=str(path))
    live = _live(("aaaa-0badc0de", 50, 1000), ("bbbb-0badc0de", 70, 998))
    assert switcher.tick(_step(live)) == INT4
    assert path.exists()
    rows = _watcher_read_cohorts(path)
    assert len(rows) == 1
    record = rows[0]
    assert record["event"] == COHORT_EVENT
    assert record["rollout_index"] == 1
    assert record["policy_revision"] == 0
    assert record["policy_kind"] == KIND_FIXED_FRONTIER
    assert record["trigger"] == {
        "committed_frontier": 1000,
        "applied_response_tokens": 1000,
        "applied_live_requests": 2,
        "decision_live_requests": 2,
        "median_prompt_tokens": 70.0,
        "reason": "switch",
    }
    assert record["requests"] == [
        {
            "entry_output_tokens": 1000,
            "prompt_tokens": 50,
            "request_id": "aaaa-0badc0de",
        },
        {
            "entry_output_tokens": 998,
            "prompt_tokens": 70,
            "request_id": "bbbb-0badc0de",
        },
    ]
    finishes = {"aaaa": 1200, "bbbb": 1500}
    ids = [_watcher_resolve(r["request_id"], finishes) for r in record["requests"]]
    assert ids == ["aaaa", "bbbb"]
    entries = [int(r["entry_output_tokens"]) for r in record["requests"]]
    finals = [finishes[rid] for rid in ids]
    assert all(f >= e for e, f in zip(entries, finals, strict=True))
    # sort_keys, one object per line, append mode.
    text = path.read_text()
    assert text.count("\n") == 1
    assert json.loads(text) == json.loads(json.dumps(record, sort_keys=True))
    switcher.tick(_step([], unfinished=0))
    switcher.tick(_step(_live(("cccc-0badc0de", 50, 1000))))
    assert [r["rollout_index"] for r in _watcher_read_cohorts(path)] == [1, 2]
    assert isinstance(switcher.writer, SwitchLogWriter)
    assert switcher.writer.records_written == 2


def test_reload_once_per_boundary_and_fail_closed_on_stale_revision(tmp_path):
    """Strict mode (VLLM_DUAL_PRECISION_REQUIRE_POLICY_ADVANCE=1)."""
    raw = _fixed_frontier_json(1000, batch=2, calibration={"policy_revision": 0})
    path = _write_policy(tmp_path, raw)
    switcher = _switcher(path, reload_each_rollout=True, require_advance=True)
    switcher.on_new_request("a")
    switcher.on_new_request("b")
    # Before rollout 1 the file is re-read as is (the calibrator has not run).
    assert switcher.reloads == [(1, 0)]
    # Watcher bumps the revision between cohorts (atomic replace).
    updated = dict(raw, calibration={"policy_revision": 1})
    _write_policy(tmp_path, updated)
    switcher.on_new_request("c")
    switcher.on_new_request("d")
    assert switcher.reloads == [(1, 0), (2, 1)]
    assert switcher.policy.policy_revision == 1
    assert switcher.rollout_index == 2
    # Empty scheduler at the boundary (rollout-only back-to-back calls) must
    # not reload twice: the old runtime logged 59 reloads for 29 boundaries.
    _write_policy(tmp_path, dict(raw, calibration={"policy_revision": 2}))
    switcher.tick(_step([], unfinished=0))
    switcher.on_new_request("e")
    switcher.tick(_step([], unfinished=0))
    switcher.on_new_request("f")
    assert switcher.reloads == [(1, 0), (2, 1), (3, 2)]
    # Stale revision (watcher lag or death): fail closed at the boundary.
    with pytest.raises(PolicyRevisionError):
        switcher.on_new_request("g")
    # A stale revision at the FIRST boundary after rollout 1 fails too: only
    # the reload before rollout 1 is exempt.
    stale_path = _write_policy(tmp_path, raw, "stale.json")
    stale = _switcher(stale_path, reload_each_rollout=True, require_advance=True)
    stale.on_new_request("a")
    stale.on_new_request("b")
    assert stale.reloads == [(1, 0)]
    with pytest.raises(PolicyRevisionError):
        stale.on_new_request("c")
    # Inline specs never reload.
    inline = _switcher("fixed_frontier:1000", reload_each_rollout=True)
    inline.tick(_step(_live(("a", 10, 0))))
    inline.tick(_step([], unfinished=0))
    inline.tick(_step(_live(("b", 10, 0))))
    assert inline.rollout_index == 2 and inline.reloads == []


def test_reload_lag_is_logged_and_recorded_when_not_strict(tmp_path):
    """Default mode: an unchanged revision at a boundary (rollout >= 2) is a
    logged lag, counted on the switcher and flagged in the cohort record
    (the archived a000 run lagged before rollout 6; ema_pair128_a010 before
    rollouts 6 and 22). Backwards revisions and invalid files still raise."""
    raw = _fixed_frontier_json(1000, batch=1, calibration={"policy_revision": 0})
    path = _write_policy(tmp_path, raw)
    cohorts = tmp_path / "cohorts.jsonl"
    logged: list[str] = []
    switcher = RolloutPrecisionSwitcher.from_settings(
        path,
        reload_each_rollout=True,
        require_advance=False,
        observations_path=str(cohorts),
        log=lambda fmt, *args: logged.append(fmt % args),
    )
    switcher.on_new_request("a")  # rollout 1: reload sees 0 == 0, not a lag
    assert switcher.policy_reload_lag_count == 0
    _write_policy(tmp_path, dict(raw, calibration={"policy_revision": 1}))
    switcher.on_new_request("b")  # rollout 2: advanced
    assert switcher.policy_reload_lag_count == 0
    switcher.on_new_request("c")  # rollout 3: 1 == 1 -> lag
    assert switcher.policy_reload_lag_count == 1
    assert switcher.reloads == [(1, 0), (2, 1), (3, 1)]
    assert sum("reload lagged before rollout 3" in line for line in logged) == 1
    assert switcher.tick(_step(_live(("c", 10, 1000)))) == INT4
    record = _watcher_read_cohorts(cohorts)[-1]
    assert record["rollout_index"] == 3
    assert record["policy_reload_lagged"] is True
    assert record["policy_revision"] == 1
    _write_policy(tmp_path, dict(raw, calibration={"policy_revision": 2}))
    switcher.on_new_request("d")  # rollout 4: advanced again, flag clears
    assert switcher.policy_reload_lag_count == 1
    assert switcher.tick(_step(_live(("d", 10, 1000)))) == INT4
    assert _watcher_read_cohorts(cohorts)[-1]["policy_reload_lagged"] is False
    # Backwards revision always fails closed.
    _write_policy(tmp_path, dict(raw, calibration={"policy_revision": 0}))
    with pytest.raises(PolicyRevisionError):
        switcher.on_new_request("e")
    assert switcher.describe()["policy_reload_lag_count"] == 1


def test_reload_disabled_keeps_the_loaded_policy(tmp_path):
    raw = _fixed_frontier_json(1000, batch=1, calibration={"policy_revision": 0})
    path = _write_policy(tmp_path, raw)
    switcher = _switcher(path, reload_each_rollout=False)
    switcher.on_new_request("a")
    _write_policy(tmp_path, dict(raw, calibration={"policy_revision": 5}))
    switcher.on_new_request("b")
    assert switcher.rollout_index == 2
    assert switcher.policy.policy_revision == 0
    assert switcher.reloads == []
    assert isinstance(switcher.store, PolicyStore)


# ---------------------------------------------------------------------------
# Scheduler integration through the public output field
# ---------------------------------------------------------------------------


@pytest.fixture
def policy_env(monkeypatch, tmp_path):
    def configure(spec_or_raw, **knobs):
        if isinstance(spec_or_raw, dict):
            spec = _write_policy(tmp_path, spec_or_raw)
        else:
            spec = spec_or_raw
        monkeypatch.setattr(envs, "VLLM_DUAL_PRECISION_POLICY", spec)
        for key, value in knobs.items():
            monkeypatch.setattr(envs, key, value)
        return spec

    return configure


def _precision(scheduler) -> str | None:
    output = scheduler.schedule()
    assert output.num_unfinished_requests == scheduler.get_num_unfinished_requests()
    return output.dual_precision_base_precision


def _grow(request, tokens: int) -> None:
    request.append_output_token_ids([123] * tokens)


def test_vanilla_scheduler_reports_none_with_the_policy_flag_empty(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_DUAL_PRECISION_POLICY", "")
    scheduler = create_scheduler(max_num_seqs=4)
    assert scheduler.precision_switcher is None
    requests = create_requests(num_requests=4, max_tokens=64)
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert output.dual_precision_base_precision is None
    assert output.num_unfinished_requests == 4
    for request in requests:
        _grow(request, 30)
    output = scheduler.schedule()
    assert output.dual_precision_base_precision is None
    assert requests[0].num_cumulative_output_tokens == requests[0].num_output_tokens


def test_switch_is_frontier_driven_and_one_way(policy_env):
    policy_env("fixed_frontier:2000")
    scheduler = create_scheduler(max_num_seqs=4)
    requests = create_requests(num_requests=4, max_tokens=4000)
    for request in requests:
        scheduler.add_request(request)
        _grow(request, 1000)
    assert _precision(scheduler) == BF16
    assert scheduler.precision_switcher.armed
    scheduler.finish_requests("0", RequestStatus.FINISHED_ABORTED)
    for request in requests[1:]:
        _grow(request, 1000)
    assert _precision(scheduler) == INT4
    # Permanent for this rollout even if admission changes.
    scheduler.add_request(
        create_requests(num_requests=1, max_tokens=4000, req_ids=["late"])[0]
    )
    assert _precision(scheduler) == INT4


def test_resets_when_next_rollout_overlaps_cleanup(policy_env):
    policy_env(_fixed_frontier_json(1000, batch=4))
    scheduler = create_scheduler(max_num_seqs=8)
    first = create_requests(num_requests=4, max_tokens=2000)
    scheduler.add_request(first[0])
    switcher = scheduler.precision_switcher
    assert switcher.armed and switcher.rollout_index == 1
    assert _precision(scheduler) == BF16
    for request in first[1:]:
        scheduler.add_request(request)
    assert len(switcher.arrival_ids) == 4
    for request in first:
        _grow(request, 1000)
    assert _precision(scheduler) == INT4
    # Next cohort submitted before the scheduler observes an empty iteration;
    # completed streaming requests may linger as WAITING_FOR_STREAMING_REQ.
    for request in first:
        request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    second = create_requests(
        num_requests=4, max_tokens=2000, req_ids=[f"next-{i}" for i in range(4)]
    )
    scheduler.add_request(second[0])
    assert switcher.rollout_index == 2 and switcher.armed and not switcher.switched
    assert _precision(scheduler) == BF16
    for request in second[1:]:
        scheduler.add_request(request)
    assert switcher.rollout_index == 2
    assert _precision(scheduler) == BF16
    assert switcher.rollout_ids == frozenset(r.request_id for r in second)
    for request in second:
        _grow(request, 1000)
    assert _precision(scheduler) == INT4


def test_preserves_staggered_arrival_cohort(policy_env):
    policy_env(_fixed_frontier_json(1000, batch=4))
    scheduler = create_scheduler(max_num_seqs=4)
    requests = create_requests(num_requests=4, max_tokens=2000)
    scheduler.add_request(requests[0])
    switcher = scheduler.precision_switcher
    assert switcher.armed and switcher.rollout_index == 1
    # An early session finishes before later sessions are admitted; an empty
    # scheduler must not erase the rollout or its commitment.
    scheduler.finish_requests("0", RequestStatus.FINISHED_STOPPED)
    assert _precision(scheduler) == BF16
    assert switcher.armed and switcher.rollout_index == 1
    assert switcher.arrival_ids == frozenset({"0"})
    for request in requests[1:]:
        scheduler.add_request(request)
    assert switcher.rollout_index == 1 and len(switcher.arrival_ids) == 4


def test_frontier_is_cumulative_across_streaming_chunks(policy_env):
    policy_env(_fixed_frontier_json(8000, batch=1, cap=12000))
    scheduler = create_scheduler(max_num_seqs=1, max_num_batched_tokens=16384)
    request = create_requests(num_requests=1, max_tokens=12000)[0]
    request.resumable = True
    request.streaming_queue = deque()
    scheduler.add_request(request)
    _grow(request, 6000)
    request.num_computed_tokens = request.num_prompt_tokens + 6000
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    scheduler.num_waiting_for_streaming_input = 1
    scheduler._update_request_as_session(
        request,
        StreamingUpdate(
            mm_features=None,
            prompt_token_ids=[],
            max_tokens=12000,
            arrival_time=1.0,
            sampling_params=request.sampling_params,
        ),
    )
    _grow(request, 2000)
    assert request.num_output_tokens == 2000
    assert request.num_cumulative_output_tokens == 8000
    assert _precision(scheduler) == INT4


def test_preserves_complete_cohort_during_streaming_gap(policy_env):
    policy_env(_fixed_frontier_json(8000, batch=1, cap=12000))
    scheduler = create_scheduler(max_num_seqs=1, max_num_batched_tokens=16384)
    request = create_requests(num_requests=1, max_tokens=12000)[0]
    request.resumable = True
    request.streaming_queue = deque()
    scheduler.add_request(request)
    _grow(request, 6000)
    request.num_computed_tokens = request.num_prompt_tokens + 6000
    request.status = RequestStatus.WAITING_FOR_STREAMING_REQ
    scheduler.num_waiting_for_streaming_input = 1
    switcher = scheduler.precision_switcher
    # A synchronized streaming boundary makes the runnable count zero even
    # though the rollout is not finished: neither the tick nor admission of
    # the continuation may erase the 8K commitment.
    assert _precision(scheduler) == BF16
    assert switcher.armed and switcher.rollout_index == 1
    continuation = create_requests(num_requests=1, max_tokens=12000)[0]
    continuation.request_id = request.request_id
    continuation.resumable = True
    scheduler.add_request(continuation)
    assert switcher.armed and switcher.rollout_index == 1
    _grow(request, 2000)
    assert request.num_cumulative_output_tokens == 8000
    assert _precision(scheduler) == INT4


def test_keeps_frontier_from_request_finished_between_schedules(policy_env):
    policy_env(_fixed_frontier_json(8000, batch=2, cap=12000))
    scheduler = create_scheduler(max_num_seqs=2, max_num_batched_tokens=16384)
    requests = create_requests(num_requests=2, max_tokens=9000)
    for request in requests:
        scheduler.add_request(request)
        _grow(request, 7999)
    assert _precision(scheduler) == BF16
    # The first request crosses 8K in output processing and is removed before
    # schedule() runs again.
    _, stopped = scheduler._update_request_with_output(requests[0], [123])
    assert stopped
    assert scheduler.precision_switcher.watermark == 8000
    scheduler.running.remove(requests[0])
    scheduler._free_request(requests[0])
    assert "0" not in scheduler.requests
    assert requests[1].num_cumulative_output_tokens == 7999
    assert _precision(scheduler) == INT4


def test_preserves_re_admitted_streaming_request_after_free(policy_env):
    policy_env(_fixed_frontier_json(8000, batch=4, cap=20000))
    scheduler = create_scheduler(max_num_seqs=4, max_num_batched_tokens=32768)
    requests = create_requests(num_requests=4, max_tokens=16000)
    for request in requests:
        request.resumable = True
        scheduler.add_request(request)
        _grow(request, 250)
    switcher = scheduler.precision_switcher
    assert _precision(scheduler) == BF16
    assert switcher.committed_frontier == 8000
    assert switcher.rollout_index == 1 and len(switcher.arrival_ids) == 4
    # The frontend re-admits a continuation after the old scheduler-side
    # object was freed; its stable id is the rollout-membership signal.
    original = requests[0]
    scheduler.waiting.remove_requests({original})
    del scheduler.requests[original.request_id]
    continuation = create_requests(num_requests=1, max_tokens=16000)[0]
    continuation.request_id = original.request_id
    continuation.resumable = True
    continuation.streaming_output_token_offset = 7000
    scheduler.add_request(continuation)
    assert switcher.rollout_index == 1 and switcher.armed
    assert switcher.committed_frontier == 8000
    assert len(switcher.arrival_ids) == 4
    _grow(continuation, 1000)
    assert _precision(scheduler) == INT4


def test_scheduler_writes_the_cohort_file_and_reads_knobs_from_envs(
    policy_env, tmp_path
):
    path = tmp_path / "cohorts.jsonl"
    policy_env(
        "fixed_frontier:500",
        VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS=str(path),
        VLLM_DUAL_PRECISION_RELOAD_POLICY_EACH_ROLLOUT=False,
    )
    scheduler = create_scheduler(max_num_seqs=2)
    requests = create_requests(num_requests=2, max_tokens=1000)
    for request in requests:
        scheduler.add_request(request)
        _grow(request, 500)
    assert _precision(scheduler) == INT4
    rows = _watcher_read_cohorts(path)
    assert [r["request_id"] for r in rows[0]["requests"]] == ["0", "1"]
    assert rows[0]["requests"][0]["prompt_tokens"] == requests[0].num_prompt_tokens


# ---------------------------------------------------------------------------
# Golden replays of archived rollouts
# ---------------------------------------------------------------------------

SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$")


def _strip_engine_suffix(request_id: str) -> str:
    return SUFFIX_RE.sub("", request_id)


def _load_golden(name: str):
    root = GOLDEN / name
    if not root.exists():
        pytest.skip(f"golden fixture {name} missing")
    meta = json.loads((root / "meta.json").read_text())
    lifetimes = [
        Lifetime(
            row["request_id"], int(row["prompt_tokens"]), int(row["generation_tokens"])
        )
        for row in map(json.loads, (root / "lifetimes.jsonl").read_text().splitlines())
    ]
    batch = int(meta["batch"])
    rollouts = [lifetimes[i : i + batch] for i in range(0, len(lifetimes), batch)]
    switches = {
        int(row["rollout_index"]): row
        for row in map(json.loads, (root / "switches.jsonl").read_text().splitlines())
    }
    return str(root / "policy.json"), rollouts, switches, meta


def _replay_golden(name: str, entry_tolerance: int):
    policy_path, rollouts, switches, meta = _load_golden(name)
    switcher = _switcher(policy_path)
    lockstep_replay(switcher, rollouts)
    assert switcher.rollout_index == len(rollouts) == meta["rollouts"]
    replayed = {event.rollout_index: event for event in switcher.switches}
    assert sorted(replayed) == sorted(switches), "switched rollouts differ"
    assert len(replayed) == meta["switches"]
    for index, expected in switches.items():
        event = replayed[index]
        assert (
            event.committed_frontier,
            event.applied_response_tokens,
            event.applied_live_requests,
        ) == (
            expected["committed_frontier"],
            expected["applied_response_tokens"],
            expected["applied_live_requests"],
        ), f"rollout {index}"
        archived = {
            _strip_engine_suffix(row["request_id"]): int(row["entry_output_tokens"])
            for row in expected["cohort"]
        }
        replayed_cohort = {
            entry.request_id: entry.entry_output_tokens for entry in event.cohort
        }
        assert set(replayed_cohort) == set(archived), f"rollout {index} cohort"
        for request_id, entry in archived.items():
            # Lockstep replay cannot reproduce the few-token admission skew of
            # the archived run; the frontier and membership are exact.
            replayed_entry = replayed_cohort[request_id]
            assert abs(replayed_entry - entry) <= entry_tolerance, (
                f"rollout {index} request {request_id}: {replayed_entry} vs {entry}"
            )


def test_golden_replay_ema_alpha0_b32_cap16384_30step():
    """29 switches and cohorts of ``b32_cap16384_ema_alpha_a000_30step``
    (receding EMA lookup whose table is revision-invariant at alpha=0; the
    only archived run produced by the final experimental scheduler with a
    constant table)."""
    _replay_golden("b32_cap16384_ema_alpha_a000_30step", entry_tolerance=4)


def test_golden_replay_fixed_frontier8000_b32_cap16384_30step():
    """28 switches of ``b32_cap16384_fixed_frontier8000_30step`` with cohorts
    parsed from the ``exact switch request states`` log lines (the run never
    set the observations path)."""
    _replay_golden("b32_cap16384_fixed_frontier8000_30step", entry_tolerance=4)


def test_golden_inline_fixed_frontier8000_matches_the_file_policy():
    """The inline ``fixed_frontier:8000`` spec (cohort-free arming, seeded
    commitment) flips at the same tick as the archived B32 file policy in
    all 30 rollouts and reproduces the same 28 switches."""
    policy_path, rollouts, switches, _ = _load_golden(
        "b32_cap16384_fixed_frontier8000_30step"
    )
    file_switcher = _switcher(policy_path)
    file_history = lockstep_replay(file_switcher, rollouts)
    inline = _switcher("fixed_frontier:8000")
    inline_history = []
    for rollout in rollouts:
        inline_history.extend(lockstep_replay(inline, [rollout]))
        inline.tick(_step([], unfinished=0))  # drain between cohort-free rollouts

    def first_int4(history: list[str]) -> int | None:
        return history.index(INT4) if INT4 in history else None

    assert [first_int4(h) for h in inline_history] == [
        first_int4(h) for h in file_history
    ]
    assert len(inline.switches) == len(file_switcher.switches) == len(switches)
    for mine, theirs in zip(inline.switches, file_switcher.switches, strict=True):
        assert (mine.rollout_index, mine.applied_response_tokens) == (
            theirs.rollout_index,
            theirs.applied_response_tokens,
        )
        assert {e.request_id for e in mine.cohort} == {
            e.request_id for e in theirs.cohort
        }

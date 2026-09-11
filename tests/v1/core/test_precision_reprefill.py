# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Re-prefill after the rollout precision switch (component C7).

Default-off ablation: with ``VLLM_DUAL_PRECISION_REPREFILL=1`` the scheduler
preempts every surviving request at the step on which the precision switcher
(C4) reports the BF16 -> INT4 switch, so the survivors' KV is recomputed under
the INT4 base.  The four archived scheduler tests are ported to the
switch-driven trigger; re-arm on drain, idempotence and the async placeholder
discard (vanilla ``reset_prefix_cache`` semantics) are new.
"""

from __future__ import annotations

import pytest

import vllm.envs as envs
from vllm.v1.core.sched.precision_switch import (
    BASE_PRECISION_BF16,
    BASE_PRECISION_INT4,
)
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test

BF16 = BASE_PRECISION_BF16
INT4 = BASE_PRECISION_INT4

# ``fixed_threshold:t`` (C5 emulation) commits at the first 250-token frontier
# and switches once the live batch is <= t; the archived tests crossed the
# threshold by aborting one of 17 requests with t = 16.
THRESHOLD = 16
FRONTIER = 250


@pytest.fixture
def reprefill_env(monkeypatch):
    def configure(policy: str = f"fixed_threshold:{THRESHOLD}", enabled: bool = True):
        monkeypatch.setattr(envs, "VLLM_DUAL_PRECISION_POLICY", policy)
        monkeypatch.setattr(envs, "VLLM_DUAL_PRECISION_REPREFILL", enabled)

    return configure


def _grow(requests, tokens: int) -> None:
    for request in requests:
        request.append_output_token_ids([123] * tokens)


def _ids(requests) -> set[str]:
    return {request.request_id for request in requests}


def _scheduled_ids(output) -> set[str]:
    ids = {req.req_id for req in output.scheduled_new_reqs}
    ids.update(output.scheduled_cached_reqs.req_ids)
    return ids


def _cross_threshold(scheduler, requests):
    """Schedule 17 fresh requests once, grow them past the first frontier and
    abort request "0" so the next ``schedule()`` crosses ``THRESHOLD``."""
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert len(output.scheduled_new_reqs) == len(requests)
    assert output.preempted_req_ids == set()
    assert output.dual_precision_base_precision == BF16
    assert not scheduler.precision_reprefill_triggered
    _grow(requests, FRONTIER)
    scheduler.finish_requests(requests[0].request_id, RequestStatus.FINISHED_ABORTED)
    return requests[1:]


# ---------------------------------------------------------------------------
# Ported: threshold crossing preempts every survivor on an idle step
# ---------------------------------------------------------------------------


def test_reprefill_on_switch_preempts_every_survivor(reprefill_env):
    reprefill_env()
    scheduler = create_scheduler(max_num_seqs=17)
    assert scheduler.precision_reprefill_enabled
    requests = create_requests(num_requests=17, num_tokens=10, max_tokens=1000)
    survivors = _cross_threshold(scheduler, requests)

    output = scheduler.schedule()
    # The switch and the re-prefill happen on the same step ...
    assert output.dual_precision_base_precision == INT4
    assert scheduler.precision_switcher.switched
    assert scheduler.precision_reprefill_triggered
    # ... which is an idle engine step: every survivor is preempted and
    # nothing is scheduled (the waiting loop is skipped after a preemption).
    assert output.preempted_req_ids == _ids(survivors)
    assert output.total_num_scheduled_tokens == 0
    assert len(scheduler.running) == 0
    assert len(scheduler.waiting) == 16
    assert scheduler.prev_step_scheduled_req_ids == set()
    # FCFS order is preserved in the waiting queue.
    assert [r.request_id for r in scheduler.waiting] == [
        r.request_id for r in survivors
    ]

    for request in survivors:
        assert request.precision_reprefill_done
        assert request.precision_reprefill_output_offset == FRONTIER
        assert request.status == RequestStatus.PREEMPTED
        assert list(request.output_token_ids) == [123] * FRONTIER
        assert request.num_computed_tokens == 0
        assert request.num_preemptions == 1

    # Next step resumes all survivors (prefill under INT4) with no preemption.
    output = scheduler.schedule()
    assert output.dual_precision_base_precision == INT4
    assert _scheduled_ids(output) == _ids(survivors)
    assert output.preempted_req_ids == set()
    assert len(scheduler.running) == 16
    assert len(scheduler.waiting) == 0
    resumed = output.scheduled_cached_reqs
    assert set(resumed.req_ids) == _ids(survivors)
    assert resumed.resumed_req_ids == _ids(survivors)
    assert output.total_num_scheduled_tokens == 16 * (10 + FRONTIER)

    # Idempotent: the switch already happened for this rollout.
    output = scheduler.schedule()
    assert output.preempted_req_ids == set()
    assert scheduler.precision_reprefill_triggered


def test_reprefill_is_a_no_op_when_the_flag_is_off(reprefill_env):
    reprefill_env(enabled=False)
    scheduler = create_scheduler(max_num_seqs=17)
    assert not scheduler.precision_reprefill_enabled
    requests = create_requests(num_requests=17, num_tokens=10, max_tokens=1000)
    survivors = _cross_threshold(scheduler, requests)
    output = scheduler.schedule()
    assert output.dual_precision_base_precision == INT4
    assert output.preempted_req_ids == set()
    assert len(scheduler.running) == 16
    assert all(not r.precision_reprefill_done for r in survivors)
    assert all(r.num_preemptions == 0 for r in survivors)


def test_reprefill_is_only_honoured_with_a_policy(reprefill_env):
    # The flag alone does nothing: without a switcher there is no switch.
    reprefill_env(policy="", enabled=True)
    scheduler = create_scheduler(max_num_seqs=4)
    assert scheduler.precision_switcher is None
    assert not scheduler.precision_reprefill_enabled
    requests = create_requests(num_requests=4, max_tokens=1000)
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    assert output.dual_precision_base_precision is None
    _grow(requests, FRONTIER)
    output = scheduler.schedule()
    assert output.preempted_req_ids == set()


def test_uniform_w4_never_switches_and_never_reprefills(reprefill_env):
    reprefill_env(policy="uniform_w4")
    scheduler = create_scheduler(max_num_seqs=4)
    assert scheduler.precision_reprefill_enabled
    requests = create_requests(num_requests=4, max_tokens=1000)
    for request in requests:
        scheduler.add_request(request)
    assert scheduler.schedule().dual_precision_base_precision == INT4
    _grow(requests, FRONTIER)
    output = scheduler.schedule()
    assert output.dual_precision_base_precision == INT4
    assert output.preempted_req_ids == set()
    assert not scheduler.precision_reprefill_triggered


# ---------------------------------------------------------------------------
# Ported: construction-time guards
# ---------------------------------------------------------------------------


def test_reprefill_rejects_prefix_cache(reprefill_env):
    reprefill_env()
    with pytest.raises(ValueError, match="requires prefix caching to be disabled"):
        create_scheduler(enable_prefix_caching=True)
    # Same configuration without the flag constructs fine (vanilla path).
    reprefill_env(enabled=False)
    create_scheduler(enable_prefix_caching=True)


def test_reprefill_rejects_kv_connector(reprefill_env):
    reprefill_env()
    with pytest.raises(ValueError, match="does not support KV connectors"):
        create_scheduler(use_kv_connector=True)


def test_reprefill_rejects_ec_connector(reprefill_env):
    reprefill_env()
    with pytest.raises(ValueError, match="does not support EC connectors"):
        create_scheduler(use_ec_connector=True, ec_role="ec_both")


# ---------------------------------------------------------------------------
# Ported: the generation cap survives the re-prefill boundary
# ---------------------------------------------------------------------------


def test_generation_cap_counts_tokens_generated_before_the_reprefill(reprefill_env):
    """Output tokens survive preemption, so ``num_output_tokens`` keeps
    counting from the pre-switch length and ``max_tokens`` is not reset.  (The
    experimental ``num_visible_output_tokens`` fold branch and the
    ``check_stop`` change were dropped: no code path folds output into the
    prompt after a preemption.)"""
    reprefill_env()
    scheduler = create_scheduler(max_num_seqs=17)
    requests = create_requests(num_requests=17, num_tokens=10, max_tokens=FRONTIER + 2)
    survivors = _cross_threshold(scheduler, requests)
    scheduler.schedule()  # the switch + re-prefill step
    scheduler.schedule()  # survivors resumed
    request = survivors[0]
    assert request.precision_reprefill_output_offset == FRONTIER
    assert request.num_output_tokens == FRONTIER
    assert not check_stop(request, max_model_len=100_000)
    request.append_output_token_ids(124)
    assert not check_stop(request, max_model_len=100_000)
    request.append_output_token_ids(125)
    assert check_stop(request, max_model_len=100_000)
    assert request.status == RequestStatus.FINISHED_LENGTH_CAPPED
    assert request.num_output_tokens == FRONTIER + 2


# ---------------------------------------------------------------------------
# New: re-arm on drain and idempotence
# ---------------------------------------------------------------------------


def _run_batch(scheduler, prefix: str):
    requests = create_requests(
        num_requests=17,
        num_tokens=10,
        max_tokens=1000,
        req_ids=[f"{prefix}-{i}" for i in range(17)],
    )
    survivors = _cross_threshold(scheduler, requests)
    output = scheduler.schedule()
    assert output.dual_precision_base_precision == INT4
    assert output.preempted_req_ids == _ids(survivors)
    assert scheduler.precision_reprefill_triggered
    return survivors


def test_reprefill_rearms_after_drain_and_fires_once_per_batch(reprefill_env):
    reprefill_env()
    scheduler = create_scheduler(max_num_seqs=17)

    first = _run_batch(scheduler, "a")
    # Resume, then a later abort inside the same rollout must not re-trigger.
    scheduler.schedule()
    scheduler.finish_requests(first[0].request_id, RequestStatus.FINISHED_ABORTED)
    _grow(first[1:], 10)
    output = scheduler.schedule()
    assert output.preempted_req_ids == set()
    assert all(r.num_preemptions == 1 for r in first[1:])

    # Drain: every request finishes, the scheduler observes an empty step.
    scheduler.finish_requests(
        [r.request_id for r in first[1:]], RequestStatus.FINISHED_ABORTED
    )
    output = scheduler.schedule()
    assert output.num_unfinished_requests == 0
    assert output.preempted_req_ids == set()
    assert not scheduler.precision_reprefill_triggered

    # The next batch is a new rollout: BF16 again, one switch, one re-prefill.
    second = _run_batch(scheduler, "b")
    assert all(r.num_preemptions == 1 for r in second)
    assert all(r.num_preemptions == 1 for r in first[1:])
    assert len(scheduler.precision_switcher.switches) == 2
    output = scheduler.schedule()
    assert output.preempted_req_ids == set()
    assert _scheduled_ids(output) == _ids(second)


def test_reprefill_skips_requests_already_done_and_marks_waiting_ones(
    reprefill_env,
):
    reprefill_env()
    scheduler = create_scheduler(max_num_seqs=16, max_num_batched_tokens=8192)
    requests = create_requests(num_requests=17, num_tokens=10, max_tokens=1000)
    for request in requests:
        scheduler.add_request(request)
    output = scheduler.schedule()
    # max_num_seqs == 16: one request stays in the waiting queue.
    assert len(output.scheduled_new_reqs) == 16
    assert len(scheduler.waiting) == 1
    _grow(requests[:16], FRONTIER)
    scheduler.finish_requests("0", RequestStatus.FINISHED_ABORTED)
    output = scheduler.schedule()
    # 15 running survivors + 1 waiting = 16 <= threshold: switch.
    assert output.dual_precision_base_precision == INT4
    assert output.preempted_req_ids == _ids(requests[1:16])
    # The request that never ran is marked done without being preempted.
    waiting_request = requests[16]
    assert waiting_request.precision_reprefill_done
    assert waiting_request.num_preemptions == 0
    assert waiting_request.precision_reprefill_output_offset == 0
    # Resume: 15 preempted (prepended, FCFS) then the never-run request.
    output = scheduler.schedule()
    assert output.preempted_req_ids == set()
    assert _scheduled_ids(output) == _ids(requests[1:17])
    assert [r.request_id for r in scheduler.running] == [
        r.request_id for r in requests[1:17]
    ]


# ---------------------------------------------------------------------------
# New: async placeholder discard matches vanilla reset_prefix_cache
# ---------------------------------------------------------------------------


def _snapshot(scheduler):
    return {
        "waiting": [r.request_id for r in scheduler.waiting],
        "running": [r.request_id for r in scheduler.running],
        "prev_step_scheduled_req_ids": set(scheduler.prev_step_scheduled_req_ids),
        "requests": {
            r.request_id: (
                r.status,
                r.num_computed_tokens,
                r.num_preemptions,
                r.num_output_placeholders,
                r.async_tokens_to_discard,
                list(r.output_token_ids),
            )
            for r in scheduler.requests.values()
        },
    }


def _async_batch_at_the_crossing(scheduler):
    """Async scheduling: schedule 17 requests, run one more async step so every
    running request carries one in-flight output placeholder, grow past the
    frontier and abort one so the next ``schedule()`` crosses the threshold."""
    requests = create_requests(num_requests=17, num_tokens=10, max_tokens=1000)
    for request in requests:
        scheduler.add_request(request)
    scheduler.schedule()
    for request in requests:
        assert request.num_output_placeholders == 1
    _grow(requests, FRONTIER)
    scheduler.finish_requests("0", RequestStatus.FINISHED_ABORTED)
    return requests[1:]


def test_async_placeholder_discard_matches_vanilla_reset_prefix_cache(
    reprefill_env,
):
    reprefill_env()
    ours = create_scheduler(max_num_seqs=17, async_scheduling=True)
    survivors = _async_batch_at_the_crossing(ours)
    output = ours.schedule()
    assert output.preempted_req_ids == _ids(survivors)
    for request in survivors:
        assert request.async_tokens_to_discard == 1
        assert request.num_output_placeholders == 0

    # Reference: the vanilla preempt-all loop on an identical scheduler.
    reprefill_env(enabled=False)
    vanilla = create_scheduler(max_num_seqs=17, async_scheduling=True)
    _async_batch_at_the_crossing(vanilla)
    assert vanilla.reset_prefix_cache(reset_running_requests=True)

    assert _snapshot(ours) == _snapshot(vanilla)

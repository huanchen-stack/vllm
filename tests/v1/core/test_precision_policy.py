# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the torch-free switching policy module (component C5).

Unit tests cover the loader, the lookup table, the commitment modes and
guard, the offline cost model, the policy store and the replay tool.  Golden
tests replay archived scheduler logs and cost predictions; they run on the
trimmed fixtures committed under ``fixtures/precision_policy`` and, when the
archive is present, on the full archived files.  Set
``PRECISION_POLICY_FULL_GOLDEN=1`` to sweep every archived state instead of a
sample (the pure-Python cost model costs ~0.3 s per prediction).
"""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from vllm.v1.core.sched import precision_policy as pp
from vllm.v1.core.sched.precision_policy import (
    KIND_COST_MODEL,
    KIND_FIXED_FRONTIER,
    KIND_FIXED_THRESHOLD,
    KIND_LOOKUP,
    KIND_UNIFORM_W4,
    REASON_ALREADY_SWITCHED,
    REASON_COMMITTED,
    REASON_GUARD_BLOCKED,
    REASON_NO_COMMITMENT,
    REASON_NOT_OBSERVED,
    REASON_SWITCH,
    CostModel,
    LookupTable,
    PolicyDecider,
    PolicyRevisionError,
    PolicySpec,
    PolicyStore,
    PrecisionPolicy,
    load_precision_policy,
    policy_from_json,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = REPO_ROOT / "vllm" / "v1" / "core" / "sched" / "precision_policy.py"
TOOL_PATH = REPO_ROOT / "tools" / "precision_policy" / "replay_policy_log.py"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "precision_policy"

ARCHIVE = Path("/data/huanchen/verl/.codex-report/new-storyline-experiments")
TAIL8K = ARCHIVE / "dynamic_tail8k_heatmap_20260823"
FULL_GOLDEN = os.environ.get("PRECISION_POLICY_FULL_GOLDEN", "0") == "1"


def _load_tool():
    spec = importlib.util.spec_from_file_location("replay_policy_log", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["replay_policy_log"] = module
    spec.loader.exec_module(module)
    return module


replay_tool = _load_tool()


def _archive(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"archive not available: {path}")
    return path


def _single_log(run_dir: Path) -> Path:
    logs = sorted(_archive(run_dir / "logs").glob("*.log"))
    if not logs:
        pytest.skip(f"no log under {run_dir}")
    return logs[0]


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------


def _table_json(
    cells: list[int],
    *,
    frontier_count: int,
    prompt_count: int,
    live_count: int,
    **overrides,
):
    table = {
        "layout": pp.LOOKUP_LAYOUT,
        "frontier_start": 250,
        "frontier_step": 250,
        "frontier_count": frontier_count,
        "prompt_bucket_start": 0,
        "prompt_bucket_step": 128,
        "prompt_bucket_count": prompt_count,
        "live_batch_start": 1,
        "live_batch_count": live_count,
        "committed_frontiers": cells,
    }
    table.update(overrides)
    return table


def _policy_json(table: dict | None = None, **overrides) -> dict:
    raw = {
        "schema_version": 6,
        "description": "unit test policy",
        "scan_interval_tokens": 250,
        "arm_min_requests": 4,
        "capture_max_batch": 32,
        "commitment_enabled": True,
        "receding_horizon_lookup": False,
        "initial_rollout_batch": 4,
        "max_switch_live_batch": None,
        "calibration": {"kind": "unit", "policy_revision": 0},
    }
    if table is not None:
        raw["lookup_table"] = table
    raw.update(overrides)
    return raw


def _policy_with_cells(cells_by_frontier: list[int], **overrides) -> PrecisionPolicy:
    """One prompt bucket, one live column: ``cells_by_frontier[i]`` at 250*(i+1)."""
    table = _table_json(
        list(cells_by_frontier),
        frontier_count=len(cells_by_frontier),
        prompt_count=1,
        live_count=1,
    )
    return policy_from_json(_policy_json(table, **overrides), source="unit")


# ---------------------------------------------------------------------------
# Module hygiene
# ---------------------------------------------------------------------------


def test_module_imports_only_the_standard_library():
    tree = ast.parse(MODULE_PATH.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names)
    assert imported <= stdlib, imported - stdlib
    assert "torch" not in imported and "numpy" not in imported


def test_module_loads_standalone_without_torch():
    script = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location("
        f"'pp_standalone', {str(MODULE_PATH)!r})\n"
        "m = importlib.util.module_from_spec(spec); sys.modules['pp_standalone'] = m\n"
        "spec.loader.exec_module(m)\n"
        "p = m.load_precision_policy('fixed_frontier:8000')\n"
        "assert p.table.committed_frontier(250, 0, 4) == 8000\n"
        "for name in ('torch', 'numpy', 'vllm'):\n"
        "    assert name not in sys.modules, name\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


# ---------------------------------------------------------------------------
# PolicySpec (decision 6)
# ---------------------------------------------------------------------------


def test_policy_spec_parses_every_kind(tmp_path):
    assert not PolicySpec.parse("").enabled
    assert not PolicySpec.parse(None).enabled
    assert not PolicySpec.parse("   ").enabled
    assert PolicySpec.parse("fixed_threshold:8") == PolicySpec(
        KIND_FIXED_THRESHOLD, value=8
    )
    assert PolicySpec.parse(" fixed_frontier:8000 ") == PolicySpec(
        KIND_FIXED_FRONTIER, value=8000
    )
    assert PolicySpec.parse("uniform_w4") == PolicySpec(KIND_UNIFORM_W4)
    path = tmp_path / "policy.json"
    spec = PolicySpec.parse(str(path))
    assert spec.kind == KIND_LOOKUP and spec.path == str(path) and spec.is_file
    assert str(PolicySpec.parse("fixed_threshold:8")) == "fixed_threshold:8"
    assert str(spec) == str(path)


@pytest.mark.parametrize(
    "spec",
    [
        "fixed_threshold",
        "fixed_threshold:",
        "fixed_threshold:x",
        "fixed_threshold:0",
        "fixed_frontier:-5",
        "uniform",
        "bogus",
    ],
)
def test_policy_spec_rejects_malformed(spec):
    with pytest.raises(ValueError):
        PolicySpec.parse(spec)


def test_spec_kinds_become_degenerate_lookup_policies():
    fixed_frontier = load_precision_policy("fixed_frontier:8000")
    assert fixed_frontier.kind == KIND_FIXED_FRONTIER
    assert fixed_frontier.fixed_switch_frontier == 8000
    assert fixed_frontier.commitment_mode == "monotone"
    assert fixed_frontier.max_switch_live_batch is None
    assert fixed_frontier.switch_live_cap == fixed_frontier.capture_max_batch == 32
    for frontier in (250, 5000, 100_000):
        for live in (1, 64, 999):
            assert (
                fixed_frontier.table.committed_frontier(frontier, 1000.0, live) == 8000
            )

    fixed_threshold = load_precision_policy("fixed_threshold:8")
    assert fixed_threshold.kind == KIND_FIXED_THRESHOLD
    assert fixed_threshold.max_switch_live_batch == 8
    assert fixed_threshold.switch_live_cap == 8
    assert fixed_threshold.fixed_switch_frontier is None
    assert fixed_threshold.table.committed_frontier(250, 0.0, 64) == 250
    assert fixed_threshold.table.committed_frontier(9000, 0.0, 1) == 250
    # A threshold above the default capture ceiling lifts the ceiling.
    assert load_precision_policy("fixed_threshold:64").capture_max_batch == 64

    uniform = load_precision_policy("uniform_w4")
    assert uniform.kind == KIND_UNIFORM_W4 and uniform.table is None
    assert uniform.capture_max_batch == pp.UNBOUNDED_CAPTURE_MAX_BATCH

    with pytest.raises(ValueError):
        load_precision_policy("")


def test_degenerate_policy_round_trips_through_schema6_json(tmp_path):
    policy = load_precision_policy("fixed_threshold:4")
    path = tmp_path / "fixed_t4.json"
    path.write_text(json.dumps(policy.to_json()))
    loaded = load_precision_policy(str(path))
    assert loaded.kind == KIND_LOOKUP  # a JSON table is a plain lookup policy
    assert loaded.table == policy.table
    assert loaded.max_switch_live_batch == 4
    assert loaded.commitment_enabled is False  # no initial_rollout_batch given


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


SCHEMA_FIXTURES = {
    "schema2_cost_model": (
        "cost/b64_cap24576_online_full_cost.schema2.json",
        2,
        KIND_COST_MODEL,
    ),
    "schema4_static": (
        "schema/b64_cap16384_tail8k_lookup250.schema4.trim.json",
        4,
        KIND_LOOKUP,
    ),
    "schema4_no_calibration": (
        "schema/b128_cap24576_forced_tail8k_calibration.schema4.trim.json",
        4,
        KIND_LOOKUP,
    ),
    "schema5_receding": (
        "schema/b64_cap16384_online_hazard_warm5.schema5.trim.json",
        5,
        KIND_LOOKUP,
    ),
    "schema6_guard": ("schema/fixed_t8.schema6_guard.trim.json", 6, KIND_LOOKUP),
    "schema6_receding_null_guard": (
        "schema/b32_cap16384_ema_pair128_a010.schema6_receding.trim.json",
        6,
        KIND_LOOKUP,
    ),
    "schema6_fixed_frontier": (
        "schema/b64_cap16384_fixed_frontier8000.schema6.trim.json",
        6,
        KIND_LOOKUP,
    ),
}


@pytest.mark.parametrize("name", sorted(SCHEMA_FIXTURES))
def test_loader_accepts_every_archived_schema(name):
    relative, schema_version, kind = SCHEMA_FIXTURES[name]
    policy = load_precision_policy(str(FIXTURES / relative))
    assert policy.schema_version == schema_version
    assert policy.kind == kind
    assert policy.capture_max_batch == 32
    if kind == KIND_LOOKUP:
        assert policy.table is not None
        assert policy.table.frontier_step == policy.scan_interval_tokens == 250
        assert policy.commitment_enabled and policy.initial_rollout_batch in (
            32,
            64,
            128,
        )
        assert policy.cost_model is None
    else:
        assert policy.table is None and policy.cost_model is not None
        assert policy.scan_interval_tokens == 1000
    if name == "schema6_guard":
        assert policy.max_switch_live_batch == 8 and policy.switch_live_cap == 8
        assert not policy.receding_horizon_lookup
        assert all(cell == 250 for cell in policy.table.committed_frontiers)
    if name == "schema6_receding_null_guard":
        assert policy.receding_horizon_lookup
        assert policy.max_switch_live_batch is None and policy.switch_live_cap == 32
        assert policy.policy_revision == 28
    if name == "schema4_no_calibration":
        assert policy.policy_revision == 0
        assert all(cell == 8000 for cell in policy.table.committed_frontiers)
    if name == "schema5_receding":
        assert policy.receding_horizon_lookup and policy.policy_revision == 0
    if name == "schema6_fixed_frontier":
        assert all(cell == 8000 for cell in policy.table.committed_frontiers)


def test_loader_reads_revision_from_calibration_and_defaults_to_zero():
    table = _table_json([0], frontier_count=1, prompt_count=1, live_count=1)
    assert policy_from_json(_policy_json(table)).policy_revision == 0
    assert (
        policy_from_json(
            _policy_json(table, calibration={"policy_revision": 7})
        ).policy_revision
        == 7
    )
    assert policy_from_json(_policy_json(table, calibration=None)).policy_revision == 0


def test_receding_lookup_flag_is_explicitly_opt_in():
    table = _table_json([0], frontier_count=1, prompt_count=1, live_count=1)
    raw = _policy_json(table)
    del raw["receding_horizon_lookup"]
    assert not policy_from_json(raw).receding_horizon_lookup
    assert policy_from_json(
        _policy_json(table, receding_horizon_lookup=True)
    ).receding_horizon_lookup


def test_loader_accepts_fixed_switch_frontier_without_a_table():
    raw = _policy_json(
        fixed_switch_frontier=1000, initial_rollout_batch=None, commitment_enabled=False
    )
    raw["offline_cost_model"] = {"response_cap": 4096}
    policy = policy_from_json(raw)
    assert policy.kind == KIND_FIXED_FRONTIER
    assert policy.fixed_switch_frontier == 1000
    assert (
        policy.table is not None and policy.table.frontier_count == (4096 - 250) // 250
    )
    assert policy.table.committed_frontier(250, 0, 1) == 1000
    assert policy.table.committed_frontier(3750, 512, 64) == 1000
    assert policy.initial_rollout_batch is None  # cohort-free arming (profiler)
    assert not policy.receding_horizon_lookup


def _good_table():
    return _table_json(
        [0, 1000, 0, 1250], frontier_count=2, prompt_count=1, live_count=2
    )


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda raw: raw.update(switch_thresholds={"1000": 8}), "decision 4"),
        (lambda raw: raw.update(lookup_hierarchy=[]), "decision 4"),
        (
            lambda raw: raw["lookup_table"].update(
                layout="live_batch,prompt_bucket,frontier_major"
            ),
            "layout",
        ),
        (
            lambda raw: raw["lookup_table"].update(committed_frontiers=[0, 1000, 0]),
            "entries",
        ),
        (lambda raw: raw["lookup_table"].update(live_batch_count=0), "positive"),
        (lambda raw: raw["lookup_table"].update(frontier_step=0), "positive"),
        (lambda raw: raw["lookup_table"].update(frontier_step=500), "frontier_step"),
        (lambda raw: raw["lookup_table"].pop("frontier_count"), "missing"),
        (
            lambda raw: raw["lookup_table"].update(committed_frontiers="0,1000,0,1250"),
            "list",
        ),
        (lambda raw: raw.update(initial_rollout_batch=None), "initial_rollout_batch"),
        (lambda raw: raw.update(max_switch_live_batch=0), "max_switch_live_batch"),
        (lambda raw: raw.update(max_switch_live_batch=64), "capture_max_batch"),
        (lambda raw: raw.update(capture_max_batch=0), "capture_max_batch"),
        (lambda raw: raw.update(scan_interval_tokens=0), "scan_interval_tokens"),
        (lambda raw: raw.pop("schema_version"), "schema_version"),
        (lambda raw: raw.update(schema_version=1), "schema_version"),
        (lambda raw: raw.update(schema_version=7), "schema_version"),
        (lambda raw: raw.update(fixed_switch_frontier=1000), "mutually exclusive"),
        (lambda raw: raw.update(lookup_table="table"), "object"),
    ],
)
def test_loader_rejects_malformed_policies(mutate, message):
    raw = _policy_json(_good_table())
    mutate(raw)
    with pytest.raises(ValueError, match=message):
        policy_from_json(raw, source="unit")


def test_loader_requires_a_decision_source_and_reports_file_errors(tmp_path):
    with pytest.raises(ValueError, match="requires one of"):
        policy_from_json(
            _policy_json(initial_rollout_batch=None, commitment_enabled=False)
        )
    missing = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="Cannot load"):
        load_precision_policy(str(missing))
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    with pytest.raises(ValueError, match="Cannot load"):
        load_precision_policy(str(broken))
    not_object = tmp_path / "list.json"
    not_object.write_text("[]")
    with pytest.raises(ValueError, match="JSON object"):
        load_precision_policy(str(not_object))


def test_cost_model_only_policy_cannot_drive_the_scheduler():
    policy = load_precision_policy(
        str(FIXTURES / SCHEMA_FIXTURES["schema2_cost_model"][0])
    )
    assert not policy.schedulable
    with pytest.raises(ValueError, match="offline"):
        PolicyDecider(policy)


# ---------------------------------------------------------------------------
# Lookup table indexing and clamping
# ---------------------------------------------------------------------------


def test_dense_lookup_uses_constant_time_flat_indexing_and_clamps_buckets():
    # Ported from tests/model_executor/test_dual_precision.py.
    table = LookupTable.from_json(
        _table_json(
            [
                0,
                0,
                0,
                1000,
                0,
                0,
                0,
                1250,
                0,
                0,
                0,
                1500,
                0,
                0,
                0,
                1750,
            ],
            frontier_count=2,
            prompt_count=2,
            live_count=4,
        )
    )
    assert table.committed_frontier(250, 0, 4) == 1000
    assert table.committed_frontier(250, 10_000, 999) == 1250
    assert table.committed_frontier(500, 0, 4) == 1500
    assert table.committed_frontier(500, 128, 4) == 1750
    assert table.committed_frontier(0, 0, 4) is None


def test_lookup_axis_semantics():
    cells = []
    for frontier_index in range(3):
        for prompt_index in range(3):
            for live_index in range(4):
                cells.append(
                    10_000 + frontier_index * 1000 + prompt_index * 100 + live_index
                )
    table = LookupTable.from_json(
        _table_json(cells, frontier_count=3, prompt_count=3, live_count=4)
    )
    # Frontier axis: not clamped, exact grid or floor within a cell.
    assert table.committed_frontier(249, 0, 1) is None
    assert table.committed_frontier(250, 0, 1) == 10_000
    assert table.committed_frontier(499, 0, 1) == 10_000
    assert table.committed_frontier(750, 0, 1) == 12_000
    assert table.committed_frontier(1000, 0, 1) is None
    assert table.committed_frontier(-250, 0, 1) is None
    # Prompt axis: round half to even, then clamp.
    assert table.committed_frontier(250, 64, 1) == 10_000  # 0.5 -> 0
    assert table.committed_frontier(250, 65, 1) == 10_100
    assert table.committed_frontier(250, 192, 1) == 10_200  # 1.5 -> 2
    assert table.committed_frontier(250, 5000, 1) == 10_200
    assert table.committed_frontier(250, -500, 1) == 10_000
    # Live axis: clamp both ends.
    assert table.committed_frontier(250, 0, 0) == 10_000
    assert table.committed_frontier(250, 0, 4) == 10_003
    assert table.committed_frontier(250, 0, 999) == 10_003
    # Zero cell -> no planned switch.
    zero = LookupTable.from_json(
        _table_json([0, 7], frontier_count=1, prompt_count=1, live_count=2)
    )
    assert zero.committed_frontier(250, 0, 1) is None
    assert zero.committed_frontier(250, 0, 2) == 7
    assert table.frontier_tokens == (250, 500, 750)
    assert table.prompt_bucket_tokens == (0, 128, 256)


def test_lookup_table_json_round_trip():
    table = LookupTable.from_json(_good_table())
    assert LookupTable.from_json(table.to_json()) == table
    with pytest.raises(ValueError, match=">= 0"):
        LookupTable.from_json(
            _table_json([0, -1], frontier_count=1, prompt_count=1, live_count=2)
        )


# ---------------------------------------------------------------------------
# Commitment modes and guard (PolicyDecider)
# ---------------------------------------------------------------------------


def _observe(
    decider: PolicyDecider,
    frontier: int,
    live: int,
    *,
    actual=None,
    prompt=0.0,
    tokens=None,
):
    return decider.observe(
        frontier_tokens=frontier,
        prompt_tokens_median=prompt,
        decision_live=live,
        actual_live=live if actual is None else actual,
        max_response_tokens=frontier if tokens is None else tokens,
    )


def test_committed_frontier_can_only_move_earlier_in_monotone_mode():
    # Ported from test_committed_frontier_can_only_move_earlier: candidates
    # 9000 -> 11000 (ignored) -> 8500 (accepted) -> None (kept).
    policy = _policy_with_cells([9000, 11000, 8500, 0])
    decider = PolicyDecider(policy)
    assert decider.base_precision == "bf16"
    first = _observe(decider, 250, 4)
    assert first == pp.Decision(9000, False, REASON_COMMITTED, True, 250, 9000, None)
    assert _observe(decider, 500, 4).committed_frontier == 9000
    assert _observe(decider, 750, 4).committed_frontier == 8500
    fourth = _observe(decider, 1000, 4)
    assert fourth.committed_frontier == 8500 and fourth.candidate_frontier is None
    assert fourth.reason == REASON_COMMITTED


def test_monotone_switch_is_evaluated_on_every_call_once_the_frontier_is_reached():
    policy = _policy_with_cells([1000, 1000, 1000, 1000])
    decider = PolicyDecider(policy)
    assert _observe(decider, 250, 4).committed_frontier == 1000
    # Same frontier again: no new observation, no switch.
    again = _observe(decider, 250, 4, tokens=300)
    assert again.reason == REASON_NOT_OBSERVED and not again.observed
    # Reaching the frontier between grid crossings still switches (pre-check).
    reached = _observe(decider, 1000, 4, tokens=1000)
    assert reached.switch_now and reached.reason == REASON_SWITCH
    assert decider.switched and decider.base_precision == "int4"
    after = _observe(decider, 1250, 4)
    assert after == pp.Decision(1000, False, REASON_ALREADY_SWITCHED)
    decider.reset()
    assert (
        not decider.switched
        and decider.committed_frontier is None
        and decider.last_frontier == 0
    )


def test_receding_candidate_replaces_commitment_and_switch_is_observation_gated():
    # Candidate sequence 9000 -> 8750 -> 9500 (moves later) -> None (clears)
    # -> 1500 at frontier 1250 -> 1500 at 1500 (switch).
    policy = _policy_with_cells(
        [9000, 8750, 9500, 0, 1500, 1500], receding_horizon_lookup=True
    )
    decider = PolicyDecider(policy)
    assert policy.commitment_mode == "receding"
    assert _observe(decider, 250, 4).committed_frontier == 9000
    assert _observe(decider, 500, 4).committed_frontier == 8750
    later = _observe(decider, 750, 4)
    assert later.committed_frontier == 9500 and later.previous_frontier == 8750
    cleared = _observe(decider, 1000, 4)
    assert cleared.committed_frontier is None and cleared.reason == REASON_NO_COMMITMENT
    committed = _observe(decider, 1250, 4)
    assert committed.committed_frontier == 1500 and not committed.switch_now
    # The watermark passes the candidate between observations: no switch
    # until the next grid crossing re-looks-up the candidate.
    between = _observe(decider, 1250, 4, tokens=1500)
    assert between.reason == REASON_NOT_OBSERVED and not between.switch_now
    assert not decider.switched
    crossing = _observe(decider, 1500, 4, tokens=1500)
    assert (
        crossing.switch_now and crossing.reason == REASON_SWITCH and crossing.observed
    )


def test_guard_uses_actual_live_and_table_uses_decision_live():
    # Two live columns: live 1 plans 500, live >= 2 plans 1000.
    cells = []
    for _frontier in range(4):
        cells.extend([500, 1000])
    table = _table_json(cells, frontier_count=4, prompt_count=1, live_count=2)
    policy = policy_from_json(_policy_json(table, max_switch_live_batch=2))
    decider = PolicyDecider(policy)
    # decision_live is padded with not-yet-arrived cohort members (2 + 2),
    # actual live is 2: the table sees 4, the guard sees 2.
    first = _observe(decider, 250, 4, actual=2)
    assert first.committed_frontier == 1000
    # Frontier reached while the actual batch is above the guard: blocked.
    blocked = _observe(decider, 1000, 3, actual=3)
    assert (
        not blocked.switch_now
        and blocked.reason == REASON_GUARD_BLOCKED
        and blocked.observed
    )
    still_blocked = _observe(decider, 1000, 3, actual=3, tokens=1100)
    assert still_blocked.reason == REASON_GUARD_BLOCKED and not still_blocked.observed
    # Drain to the guard without a new frontier: monotone switches now.
    drained = _observe(decider, 1000, 2, actual=2, tokens=1100)
    assert drained.switch_now and decider.switched


def test_receding_guard_blocked_switch_retries_at_the_next_boundary():
    policy = _policy_with_cells(
        [1000, 1000, 1000, 1000, 1000],
        receding_horizon_lookup=True,
        max_switch_live_batch=2,
    )
    decider = PolicyDecider(policy)
    _observe(decider, 250, 4)
    blocked = _observe(decider, 1000, 4, actual=4)
    assert blocked.reason == REASON_GUARD_BLOCKED
    # Drained between boundaries: receding waits for the next observation.
    waiting = _observe(decider, 1000, 2, actual=2, tokens=1100)
    assert waiting.reason == REASON_NOT_OBSERVED and not decider.switched
    retry = _observe(decider, 1250, 2, actual=2)
    assert retry.switch_now


def test_guard_defaults_to_the_capture_ceiling():
    policy = _policy_with_cells([250, 250], capture_max_batch=2)
    decider = PolicyDecider(policy)
    assert policy.max_switch_live_batch is None and policy.switch_live_cap == 2
    assert _observe(decider, 250, 3, actual=3).reason == REASON_GUARD_BLOCKED
    assert _observe(decider, 250, 2, actual=2).switch_now


def test_decider_quantizes_frontiers_to_the_scan_grid():
    policy = _policy_with_cells([9000, 8000])
    decider = PolicyDecider(policy)
    assert _observe(decider, 499, 4).frontier == 250
    assert _observe(decider, 500, 4).frontier == 500
    assert decider.observations == 2


def test_uniform_w4_is_switched_from_the_first_token():
    decider = PolicyDecider(load_precision_policy("uniform_w4"))
    assert decider.switched and decider.base_precision == "int4"
    assert _observe(decider, 250, 4).reason == REASON_ALREADY_SWITCHED
    decider.reset()
    assert decider.switched


def test_fixed_threshold_spec_switches_on_drain():
    decider = PolicyDecider(load_precision_policy("fixed_threshold:2"))
    assert _observe(decider, 250, 8, actual=8).reason == REASON_GUARD_BLOCKED
    assert (
        _observe(decider, 250, 3, actual=3, tokens=400).reason == REASON_GUARD_BLOCKED
    )
    assert _observe(decider, 250, 2, actual=2, tokens=400).switch_now


def test_fixed_frontier_spec_switches_at_the_frontier_for_any_batch():
    decider = PolicyDecider(load_precision_policy("fixed_frontier:1000"))
    assert _observe(decider, 250, 32).committed_frontier == 1000
    assert not _observe(decider, 750, 32, tokens=999).switch_now
    assert _observe(decider, 750, 32, tokens=1000).switch_now


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


def _synthetic_cost_model(**overrides) -> dict:
    raw = {
        "response_cap": 2000,
        "integration_chunk_tokens": 250,
        "min_conditional_samples": 1,
        "downstream_seconds_per_token": 0.0,
        "switch_overhead_seconds": 0.0,
        "required_gain_seconds": 0.0,
        "bf16_speed_scale": 1.0,
        "w4_speed_scale": 1.0,
        "bf16_lengths": [1500, 1500, 1500, 1500],
        "w4_lengths": [1500, 1500, 1500, 1500],
        "tpot_batches": [1, 2, 4],
        "tpot_contexts": [256, 1024, 4096],
        "bf16_tpot_ms": [[10.0, 10.0, 10.0], [12.0, 12.0, 12.0], [16.0, 16.0, 16.0]],
        "w4_tpot_ms": [[8.0, 8.0, 8.0], [9.0, 9.0, 9.0], [11.0, 11.0, 11.0]],
    }
    raw.update(overrides)
    return raw


def test_interp_log_clamps_and_interpolates_in_log2_space():
    xs, ys = [1.0, 2.0, 8.0], [0.0, 1.0, 3.0]
    assert pp._interp_log(0.5, xs, ys) == 0.0
    assert pp._interp_log(16.0, xs, ys) == 3.0
    assert pp._interp_log(2.0, xs, ys) == 1.0
    assert pp._interp_log(4.0, xs, ys) == pytest.approx(2.0)  # midpoint in log2
    assert pp._interp_log(math.sqrt(2.0), xs, ys) == pytest.approx(0.5)


def test_tpot_skips_non_finite_cells_per_batch_row():
    raw = _synthetic_cost_model(
        bf16_tpot_ms=[
            [10.0, float("nan"), 30.0],
            [float("inf"), float("nan"), float("nan")],
            [16.0, 16.0, 16.0],
        ],
    )
    model = CostModel(raw, scan_interval_tokens=250, capture_max_batch=4)
    # Row 2 (batch 2) has no finite cell and is dropped; row 1 interpolates
    # between contexts 256 and 4096 in log2 space (1024 is the midpoint).
    assert model.tpot_ms(1, 1024, "bf16") == pytest.approx(20.0)
    assert model.tpot_ms(2, 1024, "bf16") == pytest.approx(
        18.0
    )  # midpoint of 20 and 16
    assert model.tpot_ms(4, 1024, "bf16") == pytest.approx(16.0)
    with pytest.raises(ValueError, match="finite"):
        CostModel(
            _synthetic_cost_model(w4_tpot_ms=[[float("nan")] * 3] * 3),
            scan_interval_tokens=250,
            capture_max_batch=4,
        ).tpot_ms(1, 256, "w4")


def test_expected_tpot_matches_binomial_limits_and_stays_within_bounds():
    model = CostModel(
        _synthetic_cost_model(), scan_interval_tokens=250, capture_max_batch=4
    )
    assert model.expected_tpot_ms(4, 1.0, 512, "bf16") == model.tpot_ms(4, 512, "bf16")
    assert model.expected_tpot_ms(4, 0.0, 512, "bf16") == 0.0
    assert model.expected_tpot_ms(0, 0.5, 512, "bf16") == 0.0
    for live in (1, 2, 3, 4):
        for probability in (0.1, 0.5, 0.9):
            exact = sum(
                math.comb(live, k)
                * probability**k
                * (1 - probability) ** (live - k)
                * model.tpot_ms(k, 512, "bf16")
                for k in range(1, live + 1)
            )
            approx = model.expected_tpot_ms(live, probability, 512, "bf16")
            # TPOT is concave in the batch size here, so the plug-in
            # approximation sits within a few percent of the exact value.
            assert 0.9 <= approx / exact <= 1.1, (live, probability, approx, exact)


def test_tail_correction_defaults_to_one_and_interpolates_anchors():
    model = CostModel(
        _synthetic_cost_model(), scan_interval_tokens=250, capture_max_batch=4
    )
    assert model.tail_correction(8, "time_correction") == 1.0
    anchors = [
        {"live_batch": 2, "time_correction": 0.5, "token_correction": 0.7},
        {"live_batch": 8, "time_correction": 1.0, "token_correction": 1.0},
    ]
    model = CostModel(
        _synthetic_cost_model(tail_correction_anchors=anchors),
        scan_interval_tokens=250,
        capture_max_batch=4,
    )
    assert model.tail_correction(1, "time_correction") == 0.5
    assert model.tail_correction(4, "time_correction") == pytest.approx(0.75)
    assert model.tail_correction(16, "token_correction") == 1.0


def test_receding_horizon_ties_keep_now_and_capture_cap_skips_futures():
    # Equal BF16/W4 TPOT everywhere: every plan costs the same as staying,
    # so gain is 0 and the strict '<' rule keeps the current frontier.
    raw = _synthetic_cost_model(w4_tpot_ms=[[10.0] * 3, [12.0] * 3, [16.0] * 3])
    model = CostModel(raw, scan_interval_tokens=250, capture_max_batch=4)
    prediction = model.predict_receding_horizon(250, 4, 0.0)
    assert prediction is not None
    assert prediction.planned_frontier == 250
    assert prediction.predicted_gain_seconds == pytest.approx(0.0)
    assert not prediction.switch_now  # gain must exceed the margin strictly
    # W4 strictly cheaper: switch now with a positive gain.
    model = CostModel(
        _synthetic_cost_model(), scan_interval_tokens=250, capture_max_batch=4
    )
    prediction = model.predict_receding_horizon(250, 4, 0.0)
    assert prediction.switch_now and prediction.planned_frontier == 250
    assert prediction.predicted_gain_seconds > 0
    assert (
        prediction.planned_remaining_cost_seconds
        < prediction.bf16_remaining_cost_seconds
    )
    # A capture ceiling below the surviving batch skips those futures.
    capped = CostModel(
        _synthetic_cost_model(), scan_interval_tokens=250, capture_max_batch=1
    )
    assert capped.predict_receding_horizon(250, 4, 0.0) is None
    # Too few conditional samples -> no prediction.
    sparse = CostModel(
        _synthetic_cost_model(min_conditional_samples=5),
        scan_interval_tokens=250,
        capture_max_batch=4,
    )
    assert sparse.predict_receding_horizon(250, 4, 0.0) is None


def test_receding_horizon_plans_a_later_switch_when_early_w4_is_slower():
    # W4 is slower than BF16 at batch >= 2 and faster at batch 1; requests
    # finish at 1000, so the optimal plan waits for the batch to drain.
    raw = _synthetic_cost_model(
        bf16_lengths=[600, 900, 1200, 1500],
        w4_lengths=[600, 900, 1200, 1500],
        bf16_tpot_ms=[[10.0] * 3, [12.0] * 3, [16.0] * 3],
        w4_tpot_ms=[[6.0] * 3, [14.0] * 3, [20.0] * 3],
    )
    model = CostModel(raw, scan_interval_tokens=250, capture_max_batch=4)
    prediction = model.predict_receding_horizon(250, 4, 0.0)
    assert prediction is not None
    assert prediction.planned_frontier > 250 and not prediction.switch_now
    assert prediction.planned_live_batch < 4


def test_cost_model_predict_is_monotone_in_live_and_plan_search_uses_strict_less_than():
    model = CostModel(
        _synthetic_cost_model(downstream_seconds_per_token=0.001),
        scan_interval_tokens=250,
        capture_max_batch=4,
        live_batch_max=4,
    )
    alive = [1.0, 0.75, 0.5, 0.25]
    costs = [model.predict("bf16", 0, 0, live, alive) for live in range(1, 5)]
    assert all(later > earlier for earlier, later in zip(costs, costs[1:]))
    assert model.predict("bf16", 0, 0, 0, alive) == 0.0
    assert model.predict("bf16", 0, 0, 2, []) == 0.0
    # Downstream term charges every expected sampled token.
    no_slope = CostModel(
        _synthetic_cost_model(), scan_interval_tokens=250, capture_max_batch=4
    )
    rollout_only = no_slope.predict("bf16", 0, 0, 2, alive)
    assert model.predict("bf16", 0, 0, 2, alive) == pytest.approx(
        rollout_only + 0.001 * 2 * sum(alive) * 250
    )
    # Plan-vs-plan comparison: with identical TPOT grids every plan ties the
    # stay cost, so no switch is planned (strict '<' against staying).
    tie = CostModel(
        _synthetic_cost_model(w4_tpot_ms=[[10.0] * 3, [12.0] * 3, [16.0] * 3]),
        scan_interval_tokens=250,
        capture_max_batch=4,
    )
    survival = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    planned, best, stay = tie.plan_switch(0, 0, 4, survival, lambda fj: survival[fj:])
    assert planned is None and best == pytest.approx(stay)
    # W4 strictly cheaper everywhere: the earliest frontier (now) wins.
    cheaper = CostModel(
        _synthetic_cost_model(), scan_interval_tokens=250, capture_max_batch=4
    )
    planned, best, stay = cheaper.plan_switch(
        0, 0, 4, survival, lambda fj: survival[fj:]
    )
    assert planned == 250 and best < stay
    # Equal-cost futures tie to the earliest one: make W4 equal to BF16 for
    # the first two bins and cheaper afterwards; the plan switches at bin 0
    # only if strictly cheaper, so it ties at bins 0 and 1 and both plans
    # cost the same -> the earliest (bin 0) is chosen.
    grid_bf16 = [[10.0, 10.0, 10.0]] * 3
    grid_w4 = [[10.0, 10.0, 10.0]] * 3
    equal = CostModel(
        _synthetic_cost_model(bf16_tpot_ms=grid_bf16, w4_tpot_ms=grid_w4),
        scan_interval_tokens=250,
        capture_max_batch=4,
    )
    planned, best, stay = equal.plan_switch(0, 0, 4, survival, lambda fj: survival[fj:])
    assert planned is None and best == pytest.approx(stay)


def test_cost_model_memoizes_per_instance():
    model_a = CostModel(
        _synthetic_cost_model(), scan_interval_tokens=250, capture_max_batch=4
    )
    model_b = CostModel(
        _synthetic_cost_model(w4_tpot_ms=[[1.0] * 3] * 3),
        scan_interval_tokens=250,
        capture_max_batch=4,
    )
    a = model_a.expected_suffix("w4", 250, 4, 0.0)
    b = model_b.expected_suffix("w4", 250, 4, 0.0)
    assert a is not None and b is not None and a[0] > b[0]
    assert model_a.expected_suffix("w4", 250, 4, 0.0) is a  # cached tuple identity


# ---------------------------------------------------------------------------
# Policy store (reload with revision check)
# ---------------------------------------------------------------------------


def _write_policy(path: Path, revision: int, cells: list[int] | None = None) -> None:
    table = _table_json(
        cells or [9000, 9000], frontier_count=2, prompt_count=1, live_count=1
    )
    path.write_text(
        json.dumps(_policy_json(table, calibration={"policy_revision": revision}))
    )


def test_policy_store_reload_installs_only_advanced_revisions(tmp_path):
    path = tmp_path / "policy.json"
    _write_policy(path, 0)
    store = PolicyStore(str(path))
    first = store.load()
    assert store.revision == 0 and store.policy is first
    # Same revision, changed content: keep the installed policy.
    _write_policy(path, 0, [8000, 8000])
    assert store.reload() is first and not store.last_reload_advanced
    # Advanced revision: install.
    _write_policy(path, 1, [8000, 8000])
    second = store.reload()
    assert second is not first and store.revision == 1 and store.last_reload_advanced
    assert second.table.committed_frontier(250, 0, 1) == 8000
    # Backwards revision: fail closed, keep the installed policy.
    _write_policy(path, 0)
    with pytest.raises(PolicyRevisionError, match="backwards"):
        store.reload()
    assert store.policy is second
    # Corrupt file: fail closed, keep the installed policy.
    path.write_text("{")
    with pytest.raises(PolicyRevisionError, match="reload failed"):
        store.reload()
    assert store.policy is second and store.reload_count == 2


def test_policy_store_fails_closed_when_advance_is_required(tmp_path):
    path = tmp_path / "policy.json"
    _write_policy(path, 3)
    store = PolicyStore(str(path), require_advance=True)
    store.load()
    with pytest.raises(PolicyRevisionError, match="did not advance"):
        store.reload()
    _write_policy(path, 4)
    assert store.reload().policy_revision == 4


def test_policy_store_reload_is_a_no_op_for_inline_specs():
    store = PolicyStore("fixed_frontier:8000")
    policy = store.load()
    assert store.reload() is policy and not store.last_reload_advanced


# ---------------------------------------------------------------------------
# Replay tool
# ---------------------------------------------------------------------------


STATIC_RUNS = {
    "b64_cap16384_tail8k_dynamic": ("b64_cap16384_tail8k_lookup250", 53, 14),
    "b32_cap24576_tail8k_dynamic": ("b32_cap24576_tail8k_lookup250", 49, 13),
    "b128_cap16384_tail8k_dynamic": ("b128_cap16384_tail8k_lookup250", 54, 15),
}
RECEDING_RUNS = {
    "b64_cap16384_online_hazard_warm5_receding_gpuval": (
        "b64_cap16384_online_hazard_warm5_receding_lookup250",
        291,
        15,
    ),
    "b64_cap24576_online_hazard_warm5_receding_gpuval": (
        "b64_cap24576_online_hazard_warm5_receding_lookup250",
        389,
        15,
    ),
    "b128_cap16384_online_hazard_warm5_receding_gpuval": (
        "b128_cap16384_online_hazard_warm5_receding_lookup250",
        375,
        15,
    ),
    "b128_cap24576_online_hazard_warm5_receding_gpuval": (
        "b128_cap24576_online_hazard_warm5_receding_lookup250",
        258,
        15,
    ),
}


def test_replay_tool_parses_both_update_line_formats():
    prefix = "WARNING [scheduler.py:1] "
    lines = [
        prefix + "Dynamic precision lookup commitment armed: rollout_index=1, "
        "observed_peak_batch=64",
        prefix + "Dynamic precision lookup commitment updated: rollout_index=1, "
        "observation_frontier=500, live_requests=64, median_prompt_tokens=59.0, "
        "lookup_level=single_lookup, candidate_frontier=9250, "
        "previous_frontier=None, committed_frontier=9250",
        prefix + "Dynamic precision receding lookup updated: rollout_index=2, "
        "observation_frontier=750, live_requests=63, median_prompt_tokens=59.0, "
        "candidate_frontier=None, previous_frontier=8750",
        prefix + "Lookup dynamic full-cost switch: rollout_index=2, "
        "committed_frontier=8750, applied_response_tokens=8750, "
        "applied_live_requests=7",
        prefix + "Dynamic precision exact switch request states: rollout_index=2, "
        "request_count=2, format=request_id:response_tokens:prompt_tokens, "
        "states=a-1:8750:40;b-2:8746:56"
        "\x1b[36m(TransferQueueController pid=1)\x1b[0m trailing noise",
        prefix + "Reloaded dynamic precision policy before rollout 3: revision=2",
    ]
    events = list(replay_tool.parse_log_lines(lines))
    assert [event.kind for event in events] == [
        "armed",
        "commitment",
        "receding",
        "switch",
        "states",
        "reload",
    ]
    assert (
        events[1].fields["previous"] is None and events[1].fields["committed"] == 9250
    )
    assert (
        events[2].fields["candidate"] is None and events[2].fields["previous"] == 8750
    )
    assert events[3].fields == {"committed": 8750, "tokens": 8750, "live": 7}
    assert events[4].fields["states"] == [("a-1", 8750, 40), ("b-2", 8746, 56)]
    assert events[5].fields["revision"] == 2


def test_replay_tool_cli_smoke(tmp_path):
    policy = FIXTURES / "static" / "b64_cap16384_tail8k_lookup250.trim.json"
    full_log = FIXTURES / "static" / "b64_cap16384_tail8k_dynamic.policy_lines.txt"
    excerpt = tmp_path / "excerpt.log"
    excerpt.write_text("".join(full_log.read_text().splitlines(keepends=True)[:20]))
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--policy",
            str(policy),
            "--log",
            str(excerpt),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert re.search(
        r"updates=(\d+)/\1 switches=(\d+)/\2 mismatches=0", result.stdout
    ), result.stdout
    # Inject a mismatch: change one logged candidate.
    corrupted = tmp_path / "corrupted.log"
    corrupted.write_text(
        excerpt.read_text().replace(
            "candidate_frontier=9250", "candidate_frontier=9500", 1
        )
    )
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--policy",
            str(policy),
            "--log",
            str(corrupted),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "mismatches=1" in result.stdout and "MISMATCH" in result.stdout
    # Extract mode prints the policy lines only.
    result = subprocess.run(
        [sys.executable, str(TOOL_PATH), "extract", "--log", str(excerpt)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0 and len(result.stdout.splitlines()) == 20


def test_trace_replay_applies_gain_gating_and_monotone_commitment():
    class ScriptedModel:
        required_gain_seconds = 30.0
        response_cap = 2000

        def predict_receding_horizon(self, frontier, live, median_prompt):
            planned, gain = {
                250: (1000, 40.0),
                500: (1250, 50.0),
                750: (750, 20.0),
            }.get(frontier, (None, 0.0))
            if planned is None:
                return None
            return pp.Prediction(
                False, frontier, planned, live, 0, 0, 0, 0, 0, 100.0, gain
            )

    requests = [
        {"prompt_tokens": 60, "generation_tokens": 1300},
        {"prompt_tokens": 70, "generation_tokens": 1100},
        {"prompt_tokens": 80, "generation_tokens": 600},
    ]
    row = replay_tool.replay_trace_step(
        ScriptedModel(), requests, observation_interval=250, response_cap=2000
    )
    # 1000 committed at 250; 1250 ignored (later); 750 ignored (gain <= 30);
    # switch when the observed frontier reaches the commitment (1000, live 2).
    assert [entry["committed_frontier"] for entry in row["history"]] == [
        1000,
        1000,
        1000,
        1000,
    ]
    assert row["switch_frontier"] == 1000 and row["switch_live_batch"] == 2
    assert row["initial_planned_frontier"] == 1000


# ---------------------------------------------------------------------------
# Golden replays
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("run", sorted(STATIC_RUNS))
def test_lookup_replay_matches_archived_static_runs_on_fixtures(run):
    policy_name, updates, switches = STATIC_RUNS[run]
    report = replay_tool.replay_log(
        FIXTURES / "static" / f"{policy_name}.trim.json",
        [FIXTURES / "static" / f"{run}.policy_lines.txt"],
    )
    assert report.ok, report.mismatches[:5]
    assert (report.updates, report.update_matches) == (updates, updates)
    assert (report.switches, report.switch_matches) == (switches, switches)
    assert report.rollouts == 15


@pytest.mark.parametrize("run", sorted(STATIC_RUNS))
def test_lookup_replay_matches_archived_static_runs(run):
    policy_name, updates, switches = STATIC_RUNS[run]
    policy = _archive(TAIL8K / "policies" / f"{policy_name}.json")
    report = replay_tool.replay_log(policy, [_single_log(TAIL8K / "runs" / run)])
    assert report.ok, report.mismatches[:5]
    assert (report.update_matches, report.switch_matches) == (updates, switches)


@pytest.mark.parametrize("run", sorted(RECEDING_RUNS))
def test_receding_replay_matches_warm5_gpuval_runs_on_fixtures(run):
    policy_name, updates, switches = RECEDING_RUNS[run]
    report = replay_tool.replay_log(
        FIXTURES / "receding" / f"{policy_name}.trim.json",
        [FIXTURES / "receding" / f"{run}.policy_lines.txt"],
    )
    assert report.ok, report.mismatches[:5]
    assert (report.updates, report.update_matches) == (updates, updates)
    assert (report.switches, report.switch_matches) == (switches, switches)


@pytest.mark.parametrize("run", sorted(RECEDING_RUNS))
def test_receding_replay_matches_warm5_gpuval_runs(run):
    policy_name, updates, switches = RECEDING_RUNS[run]
    policy = _archive(TAIL8K / "policies" / f"{policy_name}.json")
    report = replay_tool.replay_log(policy, [_single_log(TAIL8K / "runs" / run)])
    assert report.ok, report.mismatches[:5]
    assert (report.update_matches, report.switch_matches) == (updates, switches)


_PREDICTION_LINE = re.compile(
    r"frontier=(?P<frontier>\d+), live_requests=(?P<live>\d+), .*?"
    r"planned_frontier=(?P<planned>\w+), planned_live=(?P<planned_live>\d+), "
    r"bf16_rollout_seconds=(?P<bf16_rollout>[\d.]+), "
    r"bf16_tokens=(?P<bf16_tokens>[\d.]+), "
    r"bf16_cost_seconds=(?P<bf16_cost>[\d.]+), "
    r"plan_rollout_seconds=(?P<plan_rollout>[\d.]+), "
    r"plan_tokens=(?P<plan_tokens>[\d.]+), plan_cost_seconds=(?P<plan_cost>[\d.]+), "
    r"gain_seconds=(?P<gain>[\d.-]+), switch_now=(?P<switch_now>True|False)"
)


def test_cost_model_reproduces_logged_sync_predictions():
    policy = load_precision_policy(
        str(FIXTURES / "cost" / "b64_cap24576_online_full_cost.schema2.json")
    )
    assert policy.cost_model is not None
    lines = (
        (FIXTURES / "cost" / "b64_cap24576_online_cost_dynamic.predictions.txt")
        .read_text()
        .splitlines()
    )
    assert len(lines) == 102
    sample = lines if FULL_GOLDEN else lines[::17]
    assert len(sample) >= 6
    for line in sample:
        match = _PREDICTION_LINE.search(line)
        assert match is not None, line
        logged = match.groupdict()
        # The sync log line does not carry the median prompt.  GSM8K prompts
        # in this run are 50-80 tokens, i.e. prompt bucket 0 or 128; exactly
        # one bucket reproduces the logged BF16 rollout seconds (16 lines at
        # bucket 0, 86 at bucket 128 over the full log).
        candidates = {}
        for bucket in (0.0, 128.0, 256.0):
            prediction = policy.cost_model.predict_receding_horizon(
                int(logged["frontier"]), int(logged["live"]), bucket
            )
            if prediction is not None and (
                prediction.bf16_remaining_rollout_seconds
                == pytest.approx(float(logged["bf16_rollout"]), abs=1e-5)
            ):
                candidates[bucket] = prediction
        assert len(candidates) == 1, (line, sorted(candidates))
        (prediction,) = candidates.values()
        assert prediction.planned_frontier == int(logged["planned"])
        assert prediction.planned_live_batch == int(logged["planned_live"])
        assert prediction.switch_now is (logged["switch_now"] == "True")
        for name, value in (
            ("bf16_remaining_rollout_seconds", logged["bf16_rollout"]),
            ("bf16_remaining_cost_seconds", logged["bf16_cost"]),
            ("planned_remaining_rollout_seconds", logged["plan_rollout"]),
            ("planned_remaining_cost_seconds", logged["plan_cost"]),
            ("predicted_gain_seconds", logged["gain"]),
        ):
            assert getattr(prediction, name) == pytest.approx(float(value), abs=1e-5), (
                name,
                line,
            )
        for name, value in (
            ("bf16_remaining_tokens", logged["bf16_tokens"]),
            ("planned_remaining_tokens", logged["plan_tokens"]),
        ):
            # Logged with three decimals.
            assert getattr(prediction, name) == pytest.approx(float(value), abs=1e-3), (
                name,
                line,
            )


def test_offline_table_equals_online_prediction_on_validation_states():
    policy = load_precision_policy(
        str(FIXTURES / "cost" / "runtime_full_cost_receding_1k.schema2.json")
    )
    assert policy.cost_model is not None and policy.scan_interval_tokens == 1000
    model = CostModel(
        policy.cost_model.raw,
        scan_interval_tokens=250,
        capture_max_batch=policy.capture_max_batch,
    )
    validation = json.loads(
        (
            FIXTURES / "cost" / "runtime_full_cost_lookup_committed_250_validation.json"
        ).read_text()
    )
    rows = validation["rows"]
    assert validation["sample_count"] == validation["match_count"] == len(rows) == 87
    sample = rows if FULL_GOLDEN else rows[::9]
    assert any(row["direct"] is None for row in sample) and any(
        row["direct"] is not None for row in sample
    )
    for row in sample:
        prediction = model.predict_receding_horizon(
            row["frontier"], row["live_batch"], float(row["prompt_bucket"])
        )
        planned = None if prediction is None else prediction.planned_frontier
        assert planned == row["direct"], row
    archived_table = (
        ARCHIVE
        / "dynamic_switch_rollout_20260822"
        / "policies"
        / "runtime_full_cost_lookup_committed_250.json"
    )
    if archived_table.exists():
        table_policy = load_precision_policy(str(archived_table))
        assert table_policy.table is not None
        for row in rows:
            assert (
                table_policy.table.committed_frontier(
                    row["frontier"], row["prompt_bucket"], row["live_batch"]
                )
                == row["lookup"]
            ), row


def test_commitment_replay_reproduces_archived_switch_frontiers():
    trace = _archive(
        ARCHIVE
        / "dynamic_switch_rollout_20260822"
        / "runs"
        / "bf16"
        / "traces"
        / "request_lifetimes_replica000_node000.jsonl"
    )
    expected = json.loads(
        (FIXTURES / "cost" / "commitment_policy_replay.summary.json").read_text()
    )
    by_key = {(row["action_grid_tokens"], row["step"]): row for row in expected["rows"]}
    cases = (
        [(250, 6)]
        if not FULL_GOLDEN
        else [(grid, step) for grid in (1000, 500, 250) for step in range(6, 16)]
    )
    for grid, step in cases:
        rows = replay_tool.replay_trace(
            FIXTURES / "cost" / "runtime_full_cost_receding_1k.schema2.json",
            trace,
            batch=128,
            steps=[step],
            action_grid=grid,
        )
        archived = by_key[(grid, step)]
        assert rows[0]["switch_frontier"] == archived["switch_frontier"], (grid, step)
        assert rows[0]["switch_live_batch"] == archived["switch_live_batch"], (
            grid,
            step,
        )
        assert (
            rows[0]["initial_planned_frontier"] == archived["initial_planned_frontier"]
        )
        assert rows[0]["initial_predicted_plan_cost_seconds"] == pytest.approx(
            archived["initial_predicted_plan_cost_seconds"], abs=1e-5
        )


def test_replace_keeps_policy_frozen_dataclass_usable():
    policy = load_precision_policy("fixed_frontier:8000")
    tweaked = replace(policy, max_switch_live_batch=4)
    assert tweaked.switch_live_cap == 4 and policy.switch_live_cap == 32

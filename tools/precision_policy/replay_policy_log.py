#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay a precision policy against archived scheduler logs or traces.

Torch-free audit tool for the switching policy (``vllm/v1/core/sched/
precision_policy.py``).  Two modes:

``log`` (default)
    Parse the scheduler's policy log lines from one or more run logs and drive
    a ``PolicyDecider`` with the observations they record.  Every
    ``lookup commitment updated`` / ``receding lookup updated`` line must be
    reproduced by the decider, and every ``Lookup dynamic full-cost switch``
    line must be consistent with the decider's committed frontier.

``trace``
    Replay the offline cost model over a request-lifetime trace (the archived
    ``commitment_policy_replay.py`` analysis): for each RL step, observe the
    survivors every 250 tokens, commit with the monotone rule, and report the
    switch frontier and live batch.

Exit status is 0 when every line/step matched, 1 otherwise.

Examples::

    replay_policy_log.py --policy policy.json --log run.log
    replay_policy_log.py trace --policy cost.json --trace lifetimes.jsonl \
        --batch 128 --steps 6-15 --action-grid 250
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import sys
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

_MODULE_RELATIVE = Path("vllm") / "v1" / "core" / "sched" / "precision_policy.py"


def load_policy_module() -> ModuleType:
    """Import the policy module; fall back to loading it from the repo path.

    The package import pulls in the full ``vllm`` package (and torch).  When
    that is unavailable, e.g. on a CPU-only analysis box, the module is loaded
    directly from its file, which needs only the standard library.
    """
    try:
        from vllm.v1.core.sched import precision_policy

        return precision_policy
    except ImportError:
        pass
    name = "vllm_precision_policy_standalone"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[2] / _MODULE_RELATIVE
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load precision policy module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

_LINE_PATTERNS: dict[str, re.Pattern[str]] = {
    "armed": re.compile(
        r"Dynamic precision lookup commitment armed: rollout_index=(?P<rollout>\d+), "
        r"observed_peak_batch=(?P<peak>\d+)"
    ),
    "commitment": re.compile(
        r"Dynamic precision lookup commitment updated: rollout_index=(?P<rollout>\d+), "
        r"observation_frontier=(?P<frontier>\d+), live_requests=(?P<live>\d+), "
        r"median_prompt_tokens=(?P<prompt>[\d.]+), lookup_level=(?P<level>\w+), "
        r"candidate_frontier=(?P<candidate>\d+), previous_frontier=(?P<previous>\w+), "
        r"committed_frontier=(?P<committed>\d+)"
    ),
    "receding": re.compile(
        r"Dynamic precision receding lookup updated: rollout_index=(?P<rollout>\d+), "
        r"observation_frontier=(?P<frontier>\d+), live_requests=(?P<live>\d+), "
        r"median_prompt_tokens=(?P<prompt>[\d.]+), "
        r"candidate_frontier=(?P<candidate>\w+), previous_frontier=(?P<previous>\w+)"
    ),
    "switch": re.compile(
        r"Lookup dynamic full-cost switch: rollout_index=(?P<rollout>\d+), "
        r"committed_frontier=(?P<committed>\d+), "
        r"applied_response_tokens=(?P<tokens>\d+), applied_live_requests=(?P<live>\d+)"
    ),
    "states": re.compile(
        r"Dynamic precision exact switch request states: "
        r"rollout_index=(?P<rollout>\d+), request_count=(?P<count>\d+), "
        r"format=request_id:response_tokens:prompt_tokens, "
        r"states=(?P<states>[0-9A-Za-z_:;-]*)"
    ),
    "reload": re.compile(
        r"Reloaded dynamic precision policy before rollout (?P<rollout>\d+): "
        r"revision=(?P<revision>\d+)"
    ),
}

#: Any line containing one of these markers is a policy log line.
LOG_MARKERS = (
    "Dynamic precision lookup commitment armed",
    "Dynamic precision lookup commitment updated",
    "Dynamic precision receding lookup updated",
    "Lookup dynamic full-cost switch",
    "Dynamic precision exact switch request states",
    "Reloaded dynamic precision policy",
)


@dataclass(frozen=True)
class LogEvent:
    kind: str
    rollout: int
    line_number: int
    fields: dict[str, Any] = field(default_factory=dict)


def _maybe_int(text: str) -> int | None:
    return None if text == "None" else int(text)


def parse_log_lines(lines: Iterable[str]) -> Iterator[LogEvent]:
    """Yield the policy events found in scheduler log lines, in order."""
    for line_number, line in enumerate(lines, 1):
        for kind, pattern in _LINE_PATTERNS.items():
            match = pattern.search(line)
            if match is None:
                continue
            groups = match.groupdict()
            rollout = int(groups.pop("rollout"))
            fields: dict[str, Any] = {}
            if kind == "armed":
                fields["peak"] = int(groups["peak"])
            elif kind == "commitment":
                fields.update(
                    frontier=int(groups["frontier"]),
                    live=int(groups["live"]),
                    prompt=float(groups["prompt"]),
                    candidate=int(groups["candidate"]),
                    previous=_maybe_int(groups["previous"]),
                    committed=int(groups["committed"]),
                )
            elif kind == "receding":
                fields.update(
                    frontier=int(groups["frontier"]),
                    live=int(groups["live"]),
                    prompt=float(groups["prompt"]),
                    candidate=_maybe_int(groups["candidate"]),
                    previous=_maybe_int(groups["previous"]),
                )
            elif kind == "switch":
                fields.update(
                    committed=int(groups["committed"]),
                    tokens=int(groups["tokens"]),
                    live=int(groups["live"]),
                )
            elif kind == "states":
                states = []
                for item in groups["states"].split(";"):
                    if not item:
                        continue
                    request_id, response, prompt = item.rsplit(":", 2)
                    states.append((request_id, int(response), int(prompt)))
                fields["states"] = states
            elif kind == "reload":
                fields["revision"] = int(groups["revision"])
            yield LogEvent(kind, rollout, line_number, fields)
            break


def parse_log(path: str | Path) -> list[LogEvent]:
    with Path(path).open(errors="replace") as handle:
        return list(parse_log_lines(handle))


def extract_policy_lines(path: str | Path) -> Iterator[str]:
    """Yield the policy log lines of a run log, stripped of the Ray prefix."""
    with Path(path).open(errors="replace") as handle:
        for line in handle:
            if not any(marker in line for marker in LOG_MARKERS):
                continue
            index = line.find("WARNING")
            yield (line[index:] if index >= 0 else line).rstrip("\n")


# ---------------------------------------------------------------------------
# Log replay
# ---------------------------------------------------------------------------


@dataclass
class ReplayReport:
    updates: int = 0
    update_matches: int = 0
    switches: int = 0
    switch_matches: int = 0
    rollouts: int = 0
    mismatches: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.mismatches

    def summary(self) -> str:
        return (
            f"rollouts={self.rollouts} updates={self.update_matches}/{self.updates} "
            f"switches={self.switch_matches}/{self.switches} "
            f"mismatches={len(self.mismatches)}"
        )


def _median_prompt(states: list[tuple[str, int, int]]) -> float:
    prompts = sorted(prompt for _, _, prompt in states)
    return float(prompts[len(prompts) // 2]) if prompts else 0.0


def replay_events(
    policy: Any, events: list[LogEvent], module: ModuleType | None = None
) -> ReplayReport:
    """Drive a ``PolicyDecider`` with logged observations and compare."""
    module = module or load_policy_module()
    report = ReplayReport()
    decider = module.PolicyDecider(policy)
    current_rollout: int | None = None
    last_prompt = 0.0
    last_decision = None

    def start_rollout(rollout: int) -> None:
        nonlocal current_rollout, last_prompt, last_decision
        if rollout != current_rollout:
            decider.reset()
            current_rollout = rollout
            last_prompt = 0.0
            last_decision = None
            report.rollouts += 1

    def check_switch(event: LogEvent, prompt: float) -> None:
        nonlocal last_decision
        expected = event.fields["committed"]
        tokens = event.fields["tokens"]
        live = event.fields["live"]
        report.switches += 1
        if decider.switched and last_decision is not None and last_decision.switch_now:
            ok = last_decision.committed_frontier == expected and tokens >= expected
        else:
            last_decision = decider.observe(
                frontier_tokens=tokens,
                prompt_tokens_median=prompt,
                decision_live=live,
                actual_live=live,
                max_response_tokens=tokens,
            )
            ok = (
                last_decision.switch_now
                and last_decision.committed_frontier == expected
            )
        if ok:
            report.switch_matches += 1
        else:
            report.mismatches.append(
                f"line {event.line_number} rollout {event.rollout}: switch expected "
                f"committed={expected} at tokens={tokens} live={live}; got "
                f"{last_decision}"
            )

    for index, event in enumerate(events):
        if event.kind in ("armed", "commitment", "receding", "switch"):
            start_rollout(event.rollout)
        if event.kind in ("commitment", "receding"):
            fields = event.fields
            last_prompt = fields["prompt"]
            last_decision = decider.observe(
                frontier_tokens=fields["frontier"],
                prompt_tokens_median=fields["prompt"],
                decision_live=fields["live"],
                actual_live=fields["live"],
                max_response_tokens=fields["frontier"],
            )
            report.updates += 1
            ok = (
                last_decision.observed
                and last_decision.candidate_frontier == fields["candidate"]
                and last_decision.previous_frontier == fields["previous"]
            )
            if event.kind == "commitment":
                ok = ok and last_decision.committed_frontier == fields["committed"]
            if ok:
                report.update_matches += 1
            else:
                report.mismatches.append(
                    f"line {event.line_number} rollout {event.rollout}: expected "
                    f"{fields}; got {last_decision}"
                )
        elif event.kind == "switch":
            # The exact request states (when logged) directly follow the
            # switch line and give the median prompt at the switch.
            prompt = last_prompt
            following = events[index + 1] if index + 1 < len(events) else None
            if (
                following is not None
                and following.kind == "states"
                and following.rollout == event.rollout
            ):
                prompt = _median_prompt(following.fields["states"])
            check_switch(event, prompt)
    return report


def replay_log(
    policy_path: str | Path, log_paths: Iterable[str | Path]
) -> ReplayReport:
    module = load_policy_module()
    policy = module.load_precision_policy(str(policy_path))
    combined = ReplayReport()
    for log_path in log_paths:
        report = replay_events(policy, parse_log(log_path), module)
        combined.updates += report.updates
        combined.update_matches += report.update_matches
        combined.switches += report.switches
        combined.switch_matches += report.switch_matches
        combined.rollouts += report.rollouts
        combined.mismatches.extend(f"{log_path}: {item}" for item in report.mismatches)
    return combined


# ---------------------------------------------------------------------------
# Trace replay (offline cost model)
# ---------------------------------------------------------------------------

_TRACE_PATTERNS = {
    "event": re.compile(r'"event"\s*:\s*"([^"]+)"'),
    "request_id": re.compile(r'"request_id"\s*:\s*"([^"]+)"'),
    "prompt_tokens": re.compile(r'"prompt_tokens"\s*:\s*(\d+)'),
    "generation_tokens": re.compile(r'"generation_tokens"\s*:\s*(\d+)'),
}


def _trace_field(line: str, name: str) -> str:
    found = _TRACE_PATTERNS[name].search(line)
    if found is None:
        raise ValueError(f"trace line is missing {name}")
    return found.group(1)


def load_trace_steps(path: str | Path, batch: int) -> dict[int, list[dict[str, int]]]:
    """Group a request-lifetime trace into RL steps of ``batch`` arrivals."""
    starts: list[dict[str, Any]] = []
    finishes: dict[str, int] = {}
    with Path(path).open(errors="replace") as handle:
        for line in handle:
            if '"event"' not in line:
                continue
            event = _trace_field(line, "event")
            request_id = _trace_field(line, "request_id")
            if event == "start":
                starts.append(
                    {
                        "request_id": request_id,
                        "prompt_tokens": int(_trace_field(line, "prompt_tokens")),
                    }
                )
            elif event == "finish":
                finishes[request_id] = int(_trace_field(line, "generation_tokens"))
    steps: dict[int, list[dict[str, int]]] = {}
    for index in range(0, len(starts) - batch + 1, batch):
        group = starts[index : index + batch]
        if any(request["request_id"] not in finishes for request in group):
            break
        steps[index // batch + 1] = [
            {
                "prompt_tokens": request["prompt_tokens"],
                "generation_tokens": finishes[request["request_id"]],
            }
            for request in group
        ]
    return steps


def replay_trace_step(
    cost_model: Any,
    requests: list[dict[str, int]],
    *,
    observation_interval: int,
    response_cap: int,
) -> dict[str, Any]:
    """Monotone commitment replay of one RL step (archived analysis rule)."""
    committed: int | None = None
    history: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    initial = None
    for frontier in range(observation_interval, response_cap, observation_interval):
        survivors = [
            request for request in requests if request["generation_tokens"] > frontier
        ]
        if not survivors:
            break
        live = len(survivors)
        median_prompt = float(
            statistics.median(request["prompt_tokens"] for request in survivors)
        )
        prediction = cost_model.predict_receding_horizon(frontier, live, median_prompt)
        if prediction is not None and initial is None:
            initial = prediction
        if (
            prediction is not None
            and prediction.planned_frontier is not None
            and prediction.predicted_gain_seconds > cost_model.required_gain_seconds
        ):
            candidate = prediction.planned_frontier
            committed = candidate if committed is None else min(committed, candidate)
        history.append(
            {
                "frontier": frontier,
                "live_batch": live,
                "planned_frontier": prediction.planned_frontier if prediction else None,
                "committed_frontier": committed,
            }
        )
        if committed is not None and frontier >= committed:
            selected = history[-1]
            break
    return {
        "initial_planned_frontier": initial.planned_frontier if initial else None,
        "initial_predicted_plan_cost_seconds": (
            initial.planned_remaining_cost_seconds if initial else None
        ),
        "switch_frontier": selected["frontier"] if selected else None,
        "switch_live_batch": selected["live_batch"] if selected else None,
        "history": history,
    }


def replay_trace(
    policy_path: str | Path,
    trace_path: str | Path,
    *,
    batch: int,
    steps: Iterable[int],
    action_grid: int,
    observation_interval: int = 250,
) -> list[dict[str, Any]]:
    module = load_policy_module()
    policy = module.load_precision_policy(str(policy_path))
    if policy.cost_model is None:
        raise ValueError(f"{policy_path} carries no cost_model")
    cost_model = module.CostModel(
        policy.cost_model.raw,
        scan_interval_tokens=action_grid,
        capture_max_batch=policy.capture_max_batch,
    )
    trace_steps = load_trace_steps(trace_path, batch)
    rows = []
    for step in steps:
        if step not in trace_steps:
            raise ValueError(f"trace has no complete step {step} (batch {batch})")
        row = replay_trace_step(
            cost_model,
            trace_steps[step],
            observation_interval=observation_interval,
            response_cap=cost_model.response_cap,
        )
        rows.append({"action_grid_tokens": action_grid, "step": step, **row})
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_steps(text: str) -> list[int]:
    steps: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if "-" in part:
            low, high = part.split("-", 1)
            steps.extend(range(int(low), int(high) + 1))
        elif part:
            steps.append(int(part))
    return steps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="mode")

    log_parser = subparsers.add_parser(
        "log", help="replay scheduler log lines (default)"
    )
    log_parser.add_argument(
        "--policy", required=True, help="policy JSON or inline spec"
    )
    log_parser.add_argument(
        "--log", action="append", required=True, help="run log (repeatable)"
    )
    log_parser.add_argument(
        "--json", action="store_true", help="print the report as JSON"
    )

    trace_parser = subparsers.add_parser(
        "trace", help="replay the cost model over a trace"
    )
    trace_parser.add_argument(
        "--policy", required=True, help="policy JSON with a cost_model"
    )
    trace_parser.add_argument("--trace", required=True, help="request_lifetimes jsonl")
    trace_parser.add_argument(
        "--batch", type=int, required=True, help="requests per RL step"
    )
    trace_parser.add_argument("--steps", default="1", help="steps, e.g. 6-15 or 1,3")
    trace_parser.add_argument("--action-grid", type=int, default=250)
    trace_parser.add_argument("--observation-interval", type=int, default=250)
    trace_parser.add_argument(
        "--expect", help="archived commitment_policy_replay.json to compare against"
    )

    extract_parser = subparsers.add_parser(
        "extract", help="print the policy lines of a run log"
    )
    extract_parser.add_argument("--log", required=True)

    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] not in ("log", "trace", "extract", "-h", "--help"):
        argv = ["log", *argv]
    args = parser.parse_args(argv)

    if args.mode == "extract":
        for line in extract_policy_lines(args.log):
            print(line)
        return 0

    if args.mode == "trace":
        rows = replay_trace(
            args.policy,
            args.trace,
            batch=args.batch,
            steps=_parse_steps(args.steps),
            action_grid=args.action_grid,
            observation_interval=args.observation_interval,
        )
        failures = 0
        expected_rows: dict[tuple[int, int], dict[str, Any]] = {}
        if args.expect:
            archived = json.loads(Path(args.expect).read_text())
            expected_rows = {
                (row["action_grid_tokens"], row["step"]): row
                for row in archived["rows"]
            }
        for row in rows:
            line = (
                f"grid={row['action_grid_tokens']} step={row['step']} "
                f"switch_frontier={row['switch_frontier']} "
                f"switch_live_batch={row['switch_live_batch']}"
            )
            expected = expected_rows.get((row["action_grid_tokens"], row["step"]))
            if expected is not None:
                same = (
                    expected["switch_frontier"] == row["switch_frontier"]
                    and expected["switch_live_batch"] == row["switch_live_batch"]
                )
                failures += not same
                line += (
                    " match"
                    if same
                    else (
                        f" MISMATCH expected {expected['switch_frontier']}/"
                        f"{expected['switch_live_batch']}"
                    )
                )
            print(line)
        return 1 if failures else 0

    report = replay_log(args.policy, args.log)
    if args.json:
        print(json.dumps(report.__dict__, indent=2))
    else:
        print(report.summary())
        for item in report.mismatches[:50]:
            print("MISMATCH", item)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())

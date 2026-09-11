#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trim an archived rollout run into a small golden fixture for the switcher.

Produces, under ``--out``:

* ``policy.json``   the run's policy with the prompt axis trimmed to the
                    buckets the run actually indexed (frontier axis whole);
* ``lifetimes.jsonl`` one row per request in start order:
                    ``{request_id, prompt_tokens, generation_tokens}``
                    (token ids and timestamps dropped);
* ``switches.jsonl`` the ``Lookup dynamic full-cost switch`` lines of
                    ``driver.log`` as ``{rollout_index, committed_frontier,
                    applied_response_tokens, applied_live_requests}`` plus the
                    cohort request ids/entries from ``online_switch_cohorts.jsonl``
                    when present, else from the ``exact switch request states``
                    log line;
* ``meta.json``     batch size, response cap, source paths.

    extract_switch_golden.py --run <run dir> --policy <policy.json> \
        --batch 32 --out tests/v1/core/golden/precision_switch/<name>
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SWITCH_RE = re.compile(
    r"Lookup dynamic full-cost switch: rollout_index=(\d+), "
    r"committed_frontier=(\d+), applied_response_tokens=(\d+), "
    r"applied_live_requests=(\d+)"
)
STATES_RE = re.compile(
    r"Dynamic precision exact switch request states: rollout_index=(\d+), "
    r"request_count=(\d+), format=request_id:response_tokens:prompt_tokens, "
    r"states=([0-9a-f\-]+:\d+:\d+(?:;[0-9a-f\-]+:\d+:\d+)*)"
)


def read_lifetimes(path: Path) -> list[dict[str, Any]]:
    starts: list[dict[str, Any]] = []
    finishes: dict[str, dict[str, Any]] = {}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row["event"] == "start":
                starts.append(row)
            elif row["event"] == "finish":
                finishes[row["request_id"]] = row
    starts.sort(key=lambda row: row["timestamp"])
    rows = []
    for row in starts:
        finish = finishes[row["request_id"]]
        rows.append(
            {
                "request_id": row["request_id"],
                "prompt_tokens": int(row["prompt_tokens"]),
                "generation_tokens": int(finish["generation_tokens"]),
            }
        )
    return rows


def read_switches(log_path: Path, cohorts_path: Path | None) -> list[dict[str, Any]]:
    text = log_path.read_text(errors="replace")
    switches: dict[int, dict[str, Any]] = {}
    for match in SWITCH_RE.finditer(text):
        index = int(match.group(1))
        switches[index] = {
            "rollout_index": index,
            "committed_frontier": int(match.group(2)),
            "applied_response_tokens": int(match.group(3)),
            "applied_live_requests": int(match.group(4)),
            "cohort": [],
        }
    if cohorts_path is not None and cohorts_path.exists():
        with cohorts_path.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get("event") != "switch_cohort":
                    continue
                index = int(row["rollout_index"])
                if index in switches:
                    switches[index]["cohort"] = [
                        {
                            "request_id": entry["request_id"],
                            "entry_output_tokens": int(entry["entry_output_tokens"]),
                        }
                        for entry in row["requests"]
                    ]
    else:
        for match in STATES_RE.finditer(text):
            index = int(match.group(1))
            if index not in switches:
                continue
            entries = []
            for item in match.group(3).split(";"):
                request_id, response, prompt = item.split(":")
                entries.append(
                    {
                        "request_id": request_id,
                        "entry_output_tokens": int(response),
                        "prompt_tokens": int(prompt),
                    }
                )
            switches[index]["cohort"] = entries
    return [switches[index] for index in sorted(switches)]


def trim_prompt_axis(policy: dict[str, Any], prompt_count: int) -> dict[str, Any]:
    table = policy["lookup_table"]
    old_prompts = int(table["prompt_bucket_count"])
    frontiers = int(table["frontier_count"])
    lives = int(table["live_batch_count"])
    prompt_count = min(prompt_count, old_prompts)
    cells = table["committed_frontiers"]
    kept: list[int] = []
    for frontier_index in range(frontiers):
        for prompt_index in range(prompt_count):
            start = (frontier_index * old_prompts + prompt_index) * lives
            kept.extend(int(value) for value in cells[start : start + lives])
    trimmed = dict(policy)
    trimmed["lookup_table"] = dict(table)
    trimmed["lookup_table"]["prompt_bucket_count"] = prompt_count
    trimmed["lookup_table"]["committed_frontiers"] = kept
    trimmed["description"] = (
        f"{policy.get('description', '')} [golden fixture: prompt axis "
        f"trimmed to {prompt_count} buckets]"
    )
    return trimmed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt-buckets", type=int, default=2)
    args = parser.parse_args()

    traces = sorted((args.run / "traces").glob("request_lifetimes_*.jsonl"))
    if len(traces) != 1:
        raise SystemExit(f"expected one trace file under {args.run / 'traces'}")
    lifetimes = read_lifetimes(traces[0])
    if len(lifetimes) % args.batch:
        raise SystemExit(
            f"{len(lifetimes)} requests is not a multiple of batch {args.batch}"
        )
    policy = json.loads(args.policy.read_text())
    table = policy["lookup_table"]
    step = int(table["prompt_bucket_step"])
    max_prompt = max(row["prompt_tokens"] for row in lifetimes)
    needed = round(max_prompt / step) + 1
    if needed > args.prompt_buckets:
        raise SystemExit(
            f"prompts up to {max_prompt} tokens need {needed} buckets; "
            f"pass --prompt-buckets {needed}"
        )
    switches = read_switches(
        args.run / "driver.log", args.run / "online_switch_cohorts.jsonl"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "policy.json").write_text(
        json.dumps(trim_prompt_axis(policy, args.prompt_buckets), separators=(",", ":"))
        + "\n"
    )
    with (args.out / "lifetimes.jsonl").open("w") as stream:
        for row in lifetimes:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    with (args.out / "switches.jsonl").open("w") as stream:
        for row in switches:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    (args.out / "meta.json").write_text(
        json.dumps(
            {
                "batch": args.batch,
                "rollouts": len(lifetimes) // args.batch,
                "switches": len(switches),
                "run": str(args.run),
                "policy": str(args.policy),
                "trace": str(traces[0]),
            },
            indent=2,
        )
        + "\n"
    )
    print(args.out, "rollouts", len(lifetimes) // args.batch, "switches", len(switches))


if __name__ == "__main__":
    main()

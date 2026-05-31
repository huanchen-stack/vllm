# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    from trace_schema import CallTrace
except ImportError:
    from .trace_schema import CallTrace

REQUEST_ID_SEPARATOR = "::"
SCHEDULE_EVENTS = {
    "scheduled_new",
    "scheduled_resumed",
    "scheduled_running",
}


def load_scheduler_events(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as trace_file:
        for line_no, line in enumerate(trace_file, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc
    return records


def scheduler_events_to_calls(
    records: list[dict[str, Any]],
    workload: str,
) -> list[CallTrace]:
    by_request: dict[str, dict[str, Any]] = {}
    for record in records:
        request_id = record.get("request_id")
        event = record.get("event")
        if not request_id or event == "schedule_step":
            continue

        request_key = str(request_id)
        aggregate = by_request.setdefault(
            request_key,
            {
                "request_id": request_key,
                "program_id": _program_id(record, request_key),
                "call_id": _call_id(request_key),
                "workload": workload,
                "queued_at_s": None,
                "first_scheduled_at_s": None,
                "finished_at_s": None,
                "prompt_tokens": 0,
                "output_tokens": 0,
                "cached_prompt_tokens": None,
                "metadata": {
                    "last_status": None,
                    "max_preemptions": 0,
                    "scheduler_events": [],
                },
            },
        )

        _update_common_fields(aggregate, record)
        timestamp = _optional_float(record.get("scheduler_time_s"))
        if event == "queued":
            aggregate["queued_at_s"] = _min_optional(
                aggregate["queued_at_s"], timestamp
            )
        elif event in SCHEDULE_EVENTS:
            aggregate["first_scheduled_at_s"] = _min_optional(
                aggregate["first_scheduled_at_s"], timestamp
            )
            _update_cached_tokens(aggregate, record)
        elif event == "finished":
            aggregate["finished_at_s"] = _max_optional(
                aggregate["finished_at_s"], timestamp
            )

    return [_call_trace_from_aggregate(record) for record in by_request.values()]


def _update_common_fields(
    aggregate: dict[str, Any],
    record: dict[str, Any],
) -> None:
    prompt_tokens = _optional_int(record.get("num_prompt_tokens"))
    if prompt_tokens is not None:
        aggregate["prompt_tokens"] = max(aggregate["prompt_tokens"], prompt_tokens)

    output_tokens = _optional_int(record.get("num_output_tokens"))
    if output_tokens is not None:
        aggregate["output_tokens"] = max(aggregate["output_tokens"], output_tokens)

    metadata = aggregate["metadata"]
    metadata["last_status"] = record.get("status", metadata["last_status"])
    preemptions = _optional_int(record.get("num_preemptions")) or 0
    metadata["max_preemptions"] = max(metadata["max_preemptions"], preemptions)
    metadata["scheduler_events"].append(record.get("event"))


def _update_cached_tokens(
    aggregate: dict[str, Any],
    record: dict[str, Any],
) -> None:
    local_cached = _optional_int(record.get("local_cached_tokens")) or 0
    external_cached = _optional_int(record.get("external_cached_tokens")) or 0
    cached_tokens = local_cached + external_cached
    previous = aggregate["cached_prompt_tokens"]
    if previous is None:
        aggregate["cached_prompt_tokens"] = cached_tokens
    else:
        aggregate["cached_prompt_tokens"] = max(previous, cached_tokens)


def _call_trace_from_aggregate(record: dict[str, Any]) -> CallTrace:
    return CallTrace(
        program_id=str(record.get("program_id") or "unknown"),
        call_id=str(record.get("call_id") or record["request_id"]),
        prompt_tokens=int(record.get("prompt_tokens") or 0),
        output_tokens=int(record.get("output_tokens") or 0),
        workload=str(record.get("workload") or "vllm-scheduler"),
        queued_at_s=_optional_float(record.get("queued_at_s")),
        first_scheduled_at_s=_optional_float(record.get("first_scheduled_at_s")),
        finished_at_s=_optional_float(record.get("finished_at_s")),
        cached_prompt_tokens=_optional_int(record.get("cached_prompt_tokens")),
        metadata={
            "request_id": record["request_id"],
            **record.get("metadata", {}),
        },
    )


def _program_id(record: dict[str, Any], request_id: str) -> str:
    program_id = record.get("program_id")
    if program_id:
        return str(program_id)
    if REQUEST_ID_SEPARATOR in request_id:
        return request_id.split(REQUEST_ID_SEPARATOR, 1)[0]
    return "unknown"


def _call_id(request_id: str) -> str:
    if REQUEST_ID_SEPARATOR not in request_id:
        return request_id
    _, call_id = request_id.split(REQUEST_ID_SEPARATOR, 1)
    return call_id or request_id


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _min_optional(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return min(current, candidate)


def _max_optional(current: float | None, candidate: float | None) -> float | None:
    if candidate is None:
        return current
    if current is None:
        return candidate
    return max(current, candidate)


def write_calls(path: Path, calls: list[CallTrace]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        for call in calls:
            output_file.write(json.dumps(asdict(call), sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Agentix scheduler JSONL to call-trace JSONL."
    )
    parser.add_argument("--scheduler-jsonl", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--workload", default="vllm-scheduler")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_scheduler_events(args.scheduler_jsonl)
    calls = scheduler_events_to_calls(records, args.workload)
    write_calls(args.output_jsonl, calls)


if __name__ == "__main__":
    main()

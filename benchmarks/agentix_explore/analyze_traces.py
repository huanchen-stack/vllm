# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

try:
    from trace_schema import CallTrace
except ImportError:
    from .trace_schema import CallTrace


def load_trace(path: Path) -> list[CallTrace]:
    calls: list[CallTrace] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                calls.append(CallTrace.from_json(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid trace record at {path}:{line_no}") from exc
    return calls


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[int(rank)]
    weight = rank - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def group_by_program(calls: list[CallTrace]) -> dict[str, list[CallTrace]]:
    programs: dict[str, list[CallTrace]] = defaultdict(list)
    for call in calls:
        programs[call.program_id].append(call)
    for program_calls in programs.values():
        program_calls.sort(key=call_sort_key)
    return dict(programs)


def call_sort_key(call: CallTrace) -> tuple[float, str]:
    ready_time = call.ready_time_s
    if ready_time is None:
        ready_time = call.queued_at_s
    if ready_time is None:
        ready_time = 0.0
    return ready_time, call.call_id


def describe(values: list[float]) -> str:
    if not values:
        return "n/a"
    return (
        f"n={len(values)}, mean={fmt(mean(values))}, "
        f"p50={fmt(median(values))}, p95={fmt(percentile(values, 95))}, "
        f"max={fmt(max(values))}"
    )


def longest_common_prefix(a: tuple[int, ...], b: tuple[int, ...]) -> int:
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit


def prefix_locality(calls: list[CallTrace]) -> tuple[list[float], list[float]]:
    calls_with_tokens = [c for c in calls if c.prompt_token_ids]
    intra_rates: list[float] = []
    inter_rates: list[float] = []

    previous_by_program: dict[str, CallTrace] = {}
    latest_by_program: dict[str, CallTrace] = {}

    for call in sorted(calls_with_tokens, key=call_sort_key):
        current_tokens = call.prompt_token_ids
        if not current_tokens:
            continue

        previous = previous_by_program.get(call.program_id)
        if previous and previous.prompt_token_ids:
            lcp = longest_common_prefix(current_tokens, previous.prompt_token_ids)
            intra_rates.append(lcp / max(1, len(current_tokens)))
        previous_by_program[call.program_id] = call

        other_candidates = [
            c for pid, c in latest_by_program.items() if pid != call.program_id
        ]
        other = max(other_candidates, key=call_sort_key, default=None)
        if other and other.prompt_token_ids:
            lcp = longest_common_prefix(current_tokens, other.prompt_token_ids)
            inter_rates.append(lcp / max(1, len(current_tokens)))
        latest_by_program[call.program_id] = call

    return intra_rates, inter_rates


def attained_service_correlations(
    programs: dict[str, list[CallTrace]],
) -> tuple[float | None, float | None]:
    served_vs_remaining_tokens: list[tuple[float, float]] = []
    served_vs_remaining_turns: list[tuple[float, float]] = []

    for program_calls in programs.values():
        token_work = [c.prompt_tokens + c.output_tokens for c in program_calls]
        total_tokens = sum(token_work)
        total_turns = len(program_calls)
        served_tokens = 0
        for idx, tokens in enumerate(token_work):
            remaining_tokens = total_tokens - served_tokens
            remaining_turns = total_turns - idx
            served_vs_remaining_tokens.append((served_tokens, remaining_tokens))
            served_vs_remaining_turns.append((idx, remaining_turns))
            served_tokens += tokens

    return (
        pearson_correlation(served_vs_remaining_tokens),
        pearson_correlation(served_vs_remaining_turns),
    )


def pearson_correlation(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 2:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_mean = mean(xs)
    y_mean = mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in points)
    x_den = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_den = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_den == 0 or y_den == 0:
        return None
    return numerator / (x_den * y_den)


def render_markdown(calls: list[CallTrace]) -> str:
    programs = group_by_program(calls)
    calls_per_program = [float(len(v)) for v in programs.values()]
    prompt_tokens = [float(c.prompt_tokens) for c in calls]
    output_tokens = [float(c.output_tokens) for c in calls]
    tool_latencies = [
        float(c.tool_latency_s) for c in calls if c.tool_latency_s is not None
    ]
    call_waits = [w for c in calls if (w := c.wait_s) is not None]
    call_execs = [e for c in calls if (e := c.execution_s) is not None]
    call_ratios: list[float] = []
    for call in calls:
        wait = call.wait_s
        execution = call.execution_s
        if wait is not None and execution is not None and execution > 0:
            call_ratios.append(wait / execution)
    cache_hit_rates = [
        hit_rate for c in calls if (hit_rate := c.cache_hit_rate) is not None
    ]
    intra_rates, inter_rates = prefix_locality(calls)

    program_waits: list[float] = []
    program_execs: list[float] = []
    program_ratios: list[float] = []
    for program_calls in programs.values():
        waits = [w for c in program_calls if (w := c.wait_s) is not None]
        execs = [e for c in program_calls if (e := c.execution_s) is not None]
        if waits:
            program_waits.append(sum(waits))
        if execs:
            program_execs.append(sum(execs))
        if waits and execs and sum(execs) > 0:
            program_ratios.append(sum(waits) / sum(execs))

    token_corr, turn_corr = attained_service_correlations(programs)

    lines = [
        "# Agentix-Style Trace Summary",
        "",
        "## Workload Shape",
        "",
        f"- Calls: {len(calls)}",
        f"- Programs: {len(programs)}",
        f"- Calls per program: {describe(calls_per_program)}",
        f"- Prompt tokens per call: {describe(prompt_tokens)}",
        f"- Output tokens per call: {describe(output_tokens)}",
        f"- Tool latency seconds: {describe(tool_latencies)}",
        "",
        "## Queueing And Execution",
        "",
        f"- Call wait seconds: {describe(call_waits)}",
        f"- Call execution seconds: {describe(call_execs)}",
        f"- Call wait/execution ratio: {describe(call_ratios)}",
        f"- Program wait seconds: {describe(program_waits)}",
        f"- Program execution seconds: {describe(program_execs)}",
        f"- Program wait/execution ratio: {describe(program_ratios)}",
        "",
        "## Prefix Locality",
        "",
        f"- Engine-reported cache hit rate: {describe(cache_hit_rates)}",
        f"- Intra-program LCP rate: {describe(intra_rates)}",
        f"- Inter-program LCP rate: {describe(inter_rates)}",
        "",
        "## Agentix Assumption Probe",
        "",
        "- Pearson correlation between attained token service and remaining "
        f"token work: {fmt(token_corr, 3)}",
        "- Pearson correlation between completed turns and remaining turns: "
        f"{fmt(turn_corr, 3)}",
        "",
        "Interpretation: a strong positive correlation supports the idea that "
        "higher attained service implies more remaining work. A weak or "
        "negative correlation is a warning sign for PLAS-style priority.",
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize Agentix-style agent program traces."
    )
    parser.add_argument(
        "--trace-jsonl",
        type=Path,
        required=True,
        help="Path to a JSONL trace with one LLM call per line.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=None,
        help="Optional path for a Markdown summary. Defaults to stdout.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calls = load_trace(args.trace_jsonl)
    report = render_markdown(calls)
    if args.output_md is None:
        print(report)
    else:
        args.output_md.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

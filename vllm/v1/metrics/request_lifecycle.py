# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any

from vllm.v1.metrics.stats import FinishedRequestStats


@dataclass(frozen=True)
class RequestLifecycleSpan:
    request_id: str
    arrival_time: float
    scheduled_time: float
    first_token_time: float
    finish_time: float
    queued_time: float
    prefill_time: float
    decode_time: float
    inference_time: float
    e2e_latency: float
    num_prompt_tokens: int
    num_generation_tokens: int
    finish_reason: str
    max_tokens_param: int | None
    is_corrupted: bool
    num_cached_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RequestLifecyclePoint:
    timestamp: float
    time_since_start: float
    live_requests: int
    queued_requests: int
    running_requests: int
    prefill_requests: int
    decoding_requests: int
    kv_cache_bytes: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RequestStepPoint:
    timestamp: float
    request_id: str
    prompt_tokens: int
    generation_tokens: int
    total_tokens: int
    finished: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _clip_timestamp(ts: float, *, lower: float, upper: float) -> float:
    return min(max(ts, lower), upper)


def request_span_from_finished_stats(
    finished_req: FinishedRequestStats,
    finish_time: float,
) -> RequestLifecycleSpan:
    arrival_time = finish_time - max(finished_req.e2e_latency, 0.0)
    scheduled_time = _clip_timestamp(
        arrival_time + max(finished_req.queued_time, 0.0),
        lower=arrival_time,
        upper=finish_time,
    )
    first_token_time = _clip_timestamp(
        scheduled_time + max(finished_req.prefill_time, 0.0),
        lower=scheduled_time,
        upper=finish_time,
    )

    return RequestLifecycleSpan(
        request_id=finished_req.request_id or "",
        arrival_time=arrival_time,
        scheduled_time=scheduled_time,
        first_token_time=first_token_time,
        finish_time=finish_time,
        queued_time=max(finished_req.queued_time, 0.0),
        prefill_time=max(finished_req.prefill_time, 0.0),
        decode_time=max(finished_req.decode_time, 0.0),
        inference_time=max(finished_req.inference_time, 0.0),
        e2e_latency=max(finished_req.e2e_latency, 0.0),
        num_prompt_tokens=finished_req.num_prompt_tokens,
        num_generation_tokens=finished_req.num_generation_tokens,
        finish_reason=str(finished_req.finish_reason),
        max_tokens_param=finished_req.max_tokens_param,
        is_corrupted=finished_req.is_corrupted,
        num_cached_tokens=finished_req.num_cached_tokens,
    )


def build_request_lifecycle_timeline(
    spans: list[RequestLifecycleSpan],
) -> list[RequestLifecyclePoint]:
    if not spans:
        return []

    deltas: dict[float, list[int]] = defaultdict(lambda: [0, 0, 0, 0])

    def add_delta(timestamp: float, idx: int, delta: int) -> None:
        deltas[timestamp][idx] += delta

    for span in spans:
        add_delta(span.arrival_time, 0, 1)
        add_delta(span.finish_time, 0, -1)

        add_delta(span.arrival_time, 1, 1)
        add_delta(span.scheduled_time, 1, -1)

        add_delta(span.scheduled_time, 2, 1)
        add_delta(span.finish_time, 2, -1)

        if span.first_token_time < span.finish_time:
            add_delta(span.first_token_time, 3, 1)
            add_delta(span.finish_time, 3, -1)

    start_time = min(deltas)
    live = queued = running = decoding = 0
    points: list[RequestLifecyclePoint] = []

    for timestamp in sorted(deltas):
        delta_live, delta_queued, delta_running, delta_decoding = deltas[timestamp]
        live += delta_live
        queued += delta_queued
        running += delta_running
        decoding += delta_decoding
        prefill = max(running - decoding, 0)
        points.append(
            RequestLifecyclePoint(
                timestamp=timestamp,
                time_since_start=timestamp - start_time,
                live_requests=live,
                queued_requests=queued,
                running_requests=running,
                prefill_requests=prefill,
                decoding_requests=decoding,
            )
        )

    return points


def build_kv_timeline(
    steps: list[RequestStepPoint],
    spans_by_request: dict[str, RequestLifecycleSpan],
    kv_bytes_per_token: float,
) -> list[RequestLifecyclePoint]:
    if not steps:
        return []

    deltas: dict[float, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    kv_updates: dict[float, list[RequestStepPoint]] = defaultdict(list)

    def add_delta(timestamp: float, idx: int, delta: int) -> None:
        deltas[timestamp][idx] += delta

    for span in spans_by_request.values():
        add_delta(span.arrival_time, 0, 1)
        add_delta(span.finish_time, 0, -1)

        add_delta(span.arrival_time, 1, 1)
        add_delta(span.scheduled_time, 1, -1)

        add_delta(span.scheduled_time, 2, 1)
        add_delta(span.finish_time, 2, -1)

        if span.first_token_time < span.finish_time:
            add_delta(span.first_token_time, 3, 1)
            add_delta(span.finish_time, 3, -1)

    timestamps = set(deltas)
    for step in steps:
        kv_updates[step.timestamp].append(step)
        timestamps.add(step.timestamp)

    start_time = min(timestamps)
    live = queued = running = decoding = 0
    active_tokens: dict[str, int] = {}
    points: list[RequestLifecyclePoint] = []

    for timestamp in sorted(timestamps):
        finished_request_ids: list[str] = []
        for step in kv_updates.get(timestamp, []):
            active_tokens[step.request_id] = step.total_tokens
            if step.finished:
                finished_request_ids.append(step.request_id)

        delta_live, delta_queued, delta_running, delta_decoding = deltas[timestamp]
        live += delta_live
        queued += delta_queued
        running += delta_running
        decoding += delta_decoding
        prefill = max(running - decoding, 0)

        for request_id in finished_request_ids:
            active_tokens.pop(request_id, None)

        points.append(
            RequestLifecyclePoint(
                timestamp=timestamp,
                time_since_start=timestamp - start_time,
                live_requests=live,
                queued_requests=queued,
                running_requests=running,
                prefill_requests=prefill,
                decoding_requests=decoding,
                kv_cache_bytes=sum(active_tokens.values()) * kv_bytes_per_token,
            )
        )

    return points

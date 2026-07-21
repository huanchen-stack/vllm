# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.v1.engine import FinishReason
from vllm.v1.metrics.request_lifecycle import (
    build_request_lifecycle_timeline,
    request_span_from_finished_stats,
)
from vllm.v1.metrics.stats import FinishedRequestStats


def test_request_span_from_finished_stats():
    finished_req = FinishedRequestStats(
        finish_reason=FinishReason.STOP,
        request_id="req-1",
        e2e_latency=4.0,
        queued_time=1.0,
        prefill_time=0.5,
        decode_time=2.5,
        inference_time=3.0,
        num_prompt_tokens=32,
        num_generation_tokens=8,
        max_tokens_param=16,
        is_corrupted=False,
        num_cached_tokens=4,
    )

    span = request_span_from_finished_stats(finished_req, finish_time=10.0)

    assert span.request_id == "req-1"
    assert span.arrival_time == 6.0
    assert span.scheduled_time == 7.0
    assert span.first_token_time == 7.5
    assert span.finish_time == 10.0
    assert span.finish_reason == "stop"


def test_build_request_lifecycle_timeline():
    req1 = request_span_from_finished_stats(
        FinishedRequestStats(
            finish_reason=FinishReason.STOP,
            request_id="req-1",
            e2e_latency=4.0,
            queued_time=1.0,
            prefill_time=0.5,
            decode_time=2.5,
            inference_time=3.0,
        ),
        finish_time=10.0,
    )
    req2 = request_span_from_finished_stats(
        FinishedRequestStats(
            finish_reason=FinishReason.LENGTH,
            request_id="req-2",
            e2e_latency=4.5,
            queued_time=0.5,
            prefill_time=0.5,
            decode_time=3.5,
            inference_time=4.0,
        ),
        finish_time=11.0,
    )

    timeline = build_request_lifecycle_timeline([req1, req2])
    by_offset = {round(point.time_since_start, 3): point for point in timeline}

    assert by_offset[0.0].live_requests == 1
    assert by_offset[0.5].live_requests == 2
    assert by_offset[0.5].queued_requests == 2
    assert by_offset[1.0].queued_requests == 0
    assert by_offset[1.5].running_requests == 2
    assert by_offset[1.5].prefill_requests == 0
    assert by_offset[1.5].decoding_requests == 2
    assert by_offset[1.0].prefill_requests == 2
    assert by_offset[4.0].live_requests == 1
    assert by_offset[5.0].live_requests == 0

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in scheduler trace hooks for Agentix workload experiments.

The trace sink is intentionally independent from vLLM's public metrics path.
It is disabled by default and only writes JSONL when AGENTIX_EXPLORE_TRACE_JSONL
is set. The request identity convention is kept external: encode the agent
program id in the request id as "program_id::call_id" when launching a
motivation-style workload replay.
"""

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

from vllm.v1.request import Request

AGENTIX_TRACE_PATH_ENV = "AGENTIX_EXPLORE_TRACE_JSONL"
AGENTIX_REQUEST_ID_SEPARATOR = "::"


class AgentixTraceSink:
    """Small JSONL trace writer used by local Agentix experiment branches."""

    def __init__(self, path: str | None) -> None:
        self.path = Path(path) if path else None
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "AgentixTraceSink":
        return cls(os.getenv(AGENTIX_TRACE_PATH_ENV))

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def emit(
        self,
        event: str,
        request: Request,
        timestamp_s: float | None = None,
        **extra: Any,
    ) -> None:
        if self.path is None:
            return

        wall_time_s = time.time()
        scheduler_time_s = timestamp_s if timestamp_s is not None else time.monotonic()
        record = {
            "event": event,
            "request_id": request.request_id,
            "program_id": _program_id_from_request_id(request.request_id),
            "scheduler_time_s": scheduler_time_s,
            "wall_time_s": wall_time_s,
            "arrival_time_s": request.arrival_time,
            "status": request.status.name,
            "client_index": request.client_index,
            "priority": request.priority,
            "num_prompt_tokens": request.num_prompt_tokens,
            "num_tokens": request.num_tokens,
            "num_computed_tokens": request.num_computed_tokens,
            "num_output_tokens": len(request.output_token_ids),
            "max_tokens": request.max_tokens,
            "num_preemptions": request.num_preemptions,
        }
        record.update(extra)

        with self._lock:
            with self.path.open("a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(record, sort_keys=True) + "\n")

    def emit_step(
        self,
        timestamp_s: float,
        waiting_count: int,
        running_count: int,
        skipped_waiting_count: int,
        total_num_scheduled_tokens: int,
        scheduled_new_count: int,
        scheduled_resumed_count: int,
        scheduled_running_count: int,
        preempted_count: int,
        finished_count: int,
    ) -> None:
        if self.path is None:
            return

        record = {
            "event": "schedule_step",
            "scheduler_time_s": timestamp_s,
            "wall_time_s": time.time(),
            "waiting_count": waiting_count,
            "running_count": running_count,
            "skipped_waiting_count": skipped_waiting_count,
            "total_num_scheduled_tokens": total_num_scheduled_tokens,
            "scheduled_new_count": scheduled_new_count,
            "scheduled_resumed_count": scheduled_resumed_count,
            "scheduled_running_count": scheduled_running_count,
            "preempted_count": preempted_count,
            "finished_count": finished_count,
        }
        with self._lock:
            with self.path.open("a", encoding="utf-8") as trace_file:
                trace_file.write(json.dumps(record, sort_keys=True) + "\n")


def _program_id_from_request_id(request_id: str) -> str | None:
    if AGENTIX_REQUEST_ID_SEPARATOR not in request_id:
        return None
    program_id, _ = request_id.split(AGENTIX_REQUEST_ID_SEPARATOR, 1)
    return program_id or None

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CallTrace:
    """One LLM call in an agentic program trace.

    The schema is intentionally independent of a concrete serving engine. It can
    hold pure workload data, client-side replay data, or server-side lifecycle
    timestamps when those are available.
    """

    program_id: str
    call_id: str
    prompt_tokens: int
    output_tokens: int
    workload: str = "unknown"
    parent_call_ids: tuple[str, ...] = field(default_factory=tuple)
    ready_time_s: float | None = None
    queued_at_s: float | None = None
    first_scheduled_at_s: float | None = None
    finished_at_s: float | None = None
    cached_prompt_tokens: int | None = None
    prompt_token_ids: tuple[int, ...] | None = None
    tool_name: str | None = None
    tool_latency_s: float | None = None
    thread_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CallTrace":
        parent_call_ids = tuple(str(x) for x in data.get("parent_call_ids", ()))
        prompt_token_ids = data.get("prompt_token_ids")
        if prompt_token_ids is not None:
            prompt_token_ids = tuple(int(x) for x in prompt_token_ids)

        known_fields = {
            "program_id",
            "call_id",
            "prompt_tokens",
            "output_tokens",
            "workload",
            "parent_call_ids",
            "ready_time_s",
            "queued_at_s",
            "first_scheduled_at_s",
            "finished_at_s",
            "cached_prompt_tokens",
            "prompt_token_ids",
            "tool_name",
            "tool_latency_s",
            "thread_id",
        }
        metadata = {k: v for k, v in data.items() if k not in known_fields}

        return cls(
            program_id=str(data["program_id"]),
            call_id=str(data["call_id"]),
            prompt_tokens=int(data["prompt_tokens"]),
            output_tokens=int(data["output_tokens"]),
            workload=str(data.get("workload", "unknown")),
            parent_call_ids=parent_call_ids,
            ready_time_s=_optional_float(data.get("ready_time_s")),
            queued_at_s=_optional_float(data.get("queued_at_s")),
            first_scheduled_at_s=_optional_float(
                data.get("first_scheduled_at_s")
            ),
            finished_at_s=_optional_float(data.get("finished_at_s")),
            cached_prompt_tokens=_optional_int(data.get("cached_prompt_tokens")),
            prompt_token_ids=prompt_token_ids,
            tool_name=_optional_str(data.get("tool_name")),
            tool_latency_s=_optional_float(data.get("tool_latency_s")),
            thread_id=_optional_str(data.get("thread_id")),
            metadata=metadata,
        )

    @property
    def wait_s(self) -> float | None:
        if self.queued_at_s is None or self.first_scheduled_at_s is None:
            return None
        return max(0.0, self.first_scheduled_at_s - self.queued_at_s)

    @property
    def execution_s(self) -> float | None:
        if self.first_scheduled_at_s is None or self.finished_at_s is None:
            return None
        return max(0.0, self.finished_at_s - self.first_scheduled_at_s)

    @property
    def cache_hit_rate(self) -> float | None:
        if self.cached_prompt_tokens is None or self.prompt_tokens <= 0:
            return None
        return self.cached_prompt_tokens / self.prompt_tokens


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


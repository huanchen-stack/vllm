# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler that holds early requests after their first token until the batch is ready.

vLLM samples a request's first token in its final prefill iteration.  With a normal
chunked-prefill schedule that request immediately starts decoding while later requests
are still prefilling.  Temporarily making those early requests appear caught up prevents
decode scheduling but keeps their KV blocks resident.  Once every request in the batch
has one token, normal batched decoding resumes for the entire batch, so a benchmark can
time steady-state decode at an exact batch size.

If the complete batch cannot be resident at once, scheduling eventually makes no
progress while early requests are held.  That is treated as the synchronized-batch
KV-capacity failure: the batch is aborted cleanly so the next grid cell can still run.

Shared by ``tpot_heatmap.py`` (this directory) and the LoRA kernel-ablation bench under
``tools/rollout_lora/``.  Requires an in-process EngineCore
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``) so the driver can read ``barrier_release_count``
and ``events``.
"""

from __future__ import annotations

import time
from typing import Any

from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import RequestStatus


class SynchronizedPrefillScheduler(Scheduler):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.barrier_enabled = True
        self.barrier_release_count = 0
        self.events: list[dict[str, Any]] = []
        self._sync_batch_ids: set[str] = set()
        self._sync_released = False

    def set_barrier_enabled(self, enabled: bool) -> None:
        if self.has_unfinished_requests():
            raise RuntimeError("Cannot toggle the barrier with unfinished requests")
        self.barrier_enabled = enabled
        self._sync_batch_ids.clear()
        self._sync_released = False

    def _begin_sync_batch_if_needed(self) -> None:
        active_ids = set(self.requests)
        if not active_ids:
            return
        if not self._sync_batch_ids or self._sync_batch_ids.isdisjoint(active_ids):
            self._sync_batch_ids = active_ids
            self._sync_released = False
        elif not self._sync_released:
            # LLM.generate()/enqueue() add the complete batch before stepping, but
            # accepting newly observed ids here makes the invariant explicit.
            self._sync_batch_ids.update(active_ids)

    def schedule(self):  # type: ignore[no-untyped-def]
        if not self.barrier_enabled:
            return super().schedule()
        self._begin_sync_batch_if_needed()
        if not self._sync_batch_ids or self._sync_released:
            return super().schedule()

        batch_requests = [
            self.requests[request_id]
            for request_id in self._sync_batch_ids
            if request_id in self.requests
        ]
        if batch_requests and all(
            request.num_output_tokens >= 1 for request in batch_requests
        ):
            self._sync_released = True
            self.barrier_release_count += 1
            self.events.append(
                {
                    "event": "release",
                    "batch_size": len(batch_requests),
                    "prompt_tokens": sum(r.num_prompt_tokens for r in batch_requests),
                    "timestamp": time.monotonic(),
                }
            )
            return super().schedule()

        held = [
            request
            for request in self.running
            if request.request_id in self._sync_batch_ids
            and request.num_output_tokens >= 1
        ]
        original_computed = {
            request.request_id: request.num_computed_tokens for request in held
        }
        try:
            # A caught-up request receives zero scheduled tokens.  This holds its
            # KV allocation without letting it decode ahead of the batch.
            for request in held:
                request.num_computed_tokens = max(
                    request.num_computed_tokens,
                    request.num_tokens_with_spec + request.num_output_placeholders,
                )
            output = super().schedule()
        finally:
            for request in held:
                if request.request_id in self.requests:
                    request.num_computed_tokens = original_computed[request.request_id]

        if output.total_num_scheduled_tokens == 0 and held:
            active = [
                self.requests[request_id]
                for request_id in self._sync_batch_ids
                if request_id in self.requests
            ]
            self.events.append(
                {
                    "event": "oom",
                    "reason": "synchronized batch exceeds resident KV-cache capacity",
                    "batch_size": len(active),
                    "prompt_tokens": sum(r.num_prompt_tokens for r in active),
                    "kv_cache_usage": self.kv_cache_manager.usage(),
                    "timestamp": time.monotonic(),
                }
            )
            aborted_ids = set(self.requests).intersection(self._sync_batch_ids)
            self.finish_requests(aborted_ids, RequestStatus.FINISHED_ABORTED)
            output.finished_req_ids.update(aborted_ids)
            self._sync_released = True
        return output

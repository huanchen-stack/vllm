# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Synthetic non-zero KV connector for the decode TPOT heatmap.

The scheduler side declares all but the final prompt token of every request as
externally computed; the worker side fills the allocated cache blocks (attention KV and
recurrent/conv state alike) with deterministic non-zero constants.  Long-context decode
cells therefore skip prefill entirely while attention still reads non-trivial history.
Loaded through ``kv_transfer_config.kv_connector_module_path``; benchmark-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    SupportsHMA,
)
from vllm.v1.attention.backend import AttentionMetadata

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


@dataclass
class SyntheticKVMetadata(KVConnectorMetadata):
    """Physical cache blocks that replace an external prefill."""

    blocks_by_request: dict[str, tuple[list[int], ...]]


class SyntheticKVScheduler:
    """Declare all but the final prompt token as externally computed."""

    def __init__(self) -> None:
        self.enabled = True
        self._filled_requests: set[str] = set()
        self._pending: dict[str, tuple[list[int], ...]] = {}

    def set_enabled(self, enabled: bool) -> None:
        if self._pending:
            raise RuntimeError("Cannot toggle synthetic KV with pending cache fills")
        self.enabled = enabled

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        if not self.enabled:
            return 0, False
        if request.request_id in self._filled_requests:
            return 0, False
        uncomputed = request.num_tokens - num_computed_tokens
        return max(0, uncomputed - 1), False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens <= 0:
            return
        # Fill every block allocated for the request. This is deliberately
        # group-aware: hybrid models may use different block counts for full
        # attention and recurrent-state cache groups.
        self._pending[request.request_id] = blocks.get_block_ids()
        self._filled_requests.add(request.request_id)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> SyntheticKVMetadata:
        del scheduler_output
        metadata = SyntheticKVMetadata(self._pending.copy())
        self._pending.clear()
        return metadata

    def request_finished(self, request: Request) -> None:
        self._filled_requests.discard(request.request_id)
        self._pending.pop(request.request_id, None)


def _iter_tensors(value: Any):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
    else:
        raise TypeError(f"Unsupported cache value type: {type(value)!r}")


def _tensor_view_key(tensor: torch.Tensor) -> tuple[Any, ...]:
    return (
        tensor.device.type,
        tensor.device.index,
        tensor.untyped_storage().data_ptr(),
    )


class SyntheticKVWorker:
    """Fill the physical cache blocks with deterministic non-zero values."""

    def __init__(self, vllm_config: VllmConfig, kv_cache_config: KVCacheConfig) -> None:
        transfer_config = vllm_config.kv_transfer_config
        assert transfer_config is not None
        self.float_value = float(
            transfer_config.get_from_extra_config("float_value", 0.015)
        )
        self.integer_value = int(
            transfer_config.get_from_extra_config("integer_value", 1)
        )
        if self.float_value == 0.0 or self.integer_value == 0:
            raise ValueError("Synthetic KV fill values must be non-zero")

        self.group_to_layers = {
            index: tuple(group.layer_names)
            for index, group in enumerate(kv_cache_config.kv_cache_groups)
        }
        self.kv_caches: dict[str, Any] | None = None
        self.initialized_views = 0

    def _fill_tensor(self, tensor: torch.Tensor, block_ids: list[int] | None) -> None:
        value: float | int
        value = self.float_value if tensor.is_floating_point() else self.integer_value
        if block_ids is None:
            tensor.fill_(value)
            return

        if tensor.ndim == 0:
            tensor.fill_(value)
            return
        valid_ids = sorted(
            {index for index in block_ids if 0 <= index < tensor.shape[0]}
        )
        if not valid_ids:
            return
        index = torch.tensor(valid_ids, dtype=torch.long, device=tensor.device)
        tensor.index_fill_(0, index, value)

    def register_kv_caches(self, kv_caches: dict[str, Any]) -> None:
        self.kv_caches = kv_caches
        seen: set[tuple[Any, ...]] = set()
        for cache in kv_caches.values():
            for tensor in _iter_tensors(cache):
                key = _tensor_view_key(tensor)
                if key in seen:
                    continue
                seen.add(key)
                self._fill_tensor(tensor, None)
        self.initialized_views = len(seen)
        torch.accelerator.synchronize()

    def start_fill_kv(self, metadata: SyntheticKVMetadata) -> None:
        if not metadata.blocks_by_request:
            return
        assert self.kv_caches is not None

        # ModelRunner zeroes newly allocated hybrid-cache blocks before the KV
        # connector runs. Refill those exact blocks here so the synthetic
        # history actually consumed by attention/SSM is non-zero.
        filled_views: set[tuple[tuple[Any, ...], tuple[int, ...]]] = set()
        for block_groups in metadata.blocks_by_request.values():
            for group_index, block_ids in enumerate(block_groups):
                for layer_name in self.group_to_layers.get(group_index, ()):
                    cache = self.kv_caches.get(layer_name)
                    if cache is None:
                        continue
                    for tensor in _iter_tensors(cache):
                        valid_ids = tuple(
                            sorted(
                                {
                                    index
                                    for index in block_ids
                                    if 0 <= index < tensor.shape[0]
                                }
                            )
                        )
                        key = (_tensor_view_key(tensor), valid_ids)
                        if not valid_ids or key in filled_views:
                            continue
                        filled_views.add(key)
                        self._fill_tensor(tensor, list(valid_ids))


class SyntheticKVConnector(KVConnectorBase_V1, SupportsHMA):
    """Benchmark-only connector supporting uniform and hybrid cache layouts."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self.scheduler: SyntheticKVScheduler | None = None
        self.worker: SyntheticKVWorker | None = None
        if role == KVConnectorRole.SCHEDULER:
            self.scheduler = SyntheticKVScheduler()
        elif role == KVConnectorRole.WORKER:
            self.worker = SyntheticKVWorker(vllm_config, kv_cache_config)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        assert self.worker is not None
        self.worker.register_kv_caches(kv_caches)

    def set_scheduler_enabled(self, enabled: bool) -> None:
        """Enable synthetic external hits for subsequent requests.

        This is intentionally scheduler-side: with no external tokens allocated,
        the worker receives empty metadata and performs no synthetic fill.  It lets a
        benchmark run real-prefill rollouts before switching the same engine to
        decode-only heatmap measurements.
        """
        if self.scheduler is None:
            raise RuntimeError("Synthetic KV mode can only be toggled on the scheduler")
        self.scheduler.set_enabled(enabled)

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        del forward_context, kwargs
        assert self.worker is not None
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, SyntheticKVMetadata)
        self.worker.start_fill_kv(metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        del layer_name

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        del layer_name, kv_layer, attn_metadata, kwargs

    def wait_for_save(self) -> None:
        pass

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int, bool]:
        assert self.scheduler is not None
        return self.scheduler.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        assert self.scheduler is not None
        self.scheduler.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> SyntheticKVMetadata:
        assert self.scheduler is not None
        return self.scheduler.build_connector_meta(scheduler_output)

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        assert self.scheduler is not None
        self.scheduler.request_finished(request)
        return False, None

    def request_finished_all_groups(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        del block_ids
        assert self.scheduler is not None
        self.scheduler.request_finished(request)
        return False, None

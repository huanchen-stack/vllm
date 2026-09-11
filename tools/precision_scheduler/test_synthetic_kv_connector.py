# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the synthetic KV connector (ported from the report-dir harness)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from synthetic_kv_connector import SyntheticKVScheduler, SyntheticKVWorker


class _TransferConfig:
    def __init__(self, values):
        self.values = values

    def get_from_extra_config(self, key, default):
        return self.values.get(key, default)


def _worker_config(float_value=0.015, integer_value=1):
    vllm_config = SimpleNamespace(
        kv_transfer_config=_TransferConfig(
            {"float_value": float_value, "integer_value": integer_value}
        )
    )
    kv_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=["attention"]),
            SimpleNamespace(layer_names=["mamba"]),
        ]
    )
    return vllm_config, kv_config


def test_worker_supports_tensor_lists_and_shared_views():
    vllm_config, kv_config = _worker_config()
    worker = SyntheticKVWorker(vllm_config, kv_config)
    attention = torch.zeros(8, 2, 4)
    recurrent = torch.zeros(8, 3)
    convolution = torch.zeros(8, 2)
    worker.register_kv_caches(
        {
            "attention": attention,
            "mamba": [recurrent, convolution],
            "shared_attention": attention,
        }
    )
    assert worker.initialized_views == 3
    assert torch.count_nonzero(attention) == attention.numel()
    assert torch.count_nonzero(recurrent) == recurrent.numel()
    assert torch.count_nonzero(convolution) == convolution.numel()
    attention[2:4].zero_()
    recurrent[1:3].zero_()
    convolution[1:3].zero_()
    worker.start_fill_kv(
        SimpleNamespace(blocks_by_request={"request": ([2, 3], [1, 2])})
    )
    assert torch.count_nonzero(attention[2:4]) == attention[2:4].numel()
    assert torch.count_nonzero(recurrent[1:3]) == recurrent[1:3].numel()
    assert torch.count_nonzero(convolution[1:3]) == convolution[1:3].numel()
    # untouched blocks keep their fill; out-of-range ids are ignored
    worker.start_fill_kv(SimpleNamespace(blocks_by_request={"r": ([99], [99])}))


def test_zero_fill_values_are_rejected():
    vllm_config, kv_config = _worker_config(float_value=0.0)
    with pytest.raises(ValueError, match="non-zero"):
        SyntheticKVWorker(vllm_config, kv_config)
    vllm_config, kv_config = _worker_config(integer_value=0)
    with pytest.raises(ValueError, match="non-zero"):
        SyntheticKVWorker(vllm_config, kv_config)


def test_scheduler_declares_all_but_last_token_and_can_be_disabled():
    scheduler = SyntheticKVScheduler()
    request = SimpleNamespace(request_id="request", num_tokens=128)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (127, False)
    assert scheduler.get_num_new_matched_tokens(request, 100) == (27, False)
    scheduler.set_enabled(False)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.set_enabled(True)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (127, False)


def test_scheduler_fills_once_per_request_and_clears_pending():
    scheduler = SyntheticKVScheduler()
    request = SimpleNamespace(request_id="request", num_tokens=64)
    blocks = SimpleNamespace(get_block_ids=lambda: ([1, 2], [3]))
    scheduler.update_state_after_alloc(request, blocks, 63)
    assert scheduler.get_num_new_matched_tokens(request, 63) == (0, False)
    with pytest.raises(RuntimeError, match="pending"):
        scheduler.set_enabled(False)
    metadata = scheduler.build_connector_meta(None)
    assert metadata.blocks_by_request == {"request": ([1, 2], [3])}
    assert scheduler.build_connector_meta(None).blocks_by_request == {}
    scheduler.request_finished(request)
    assert scheduler.get_num_new_matched_tokens(request, 0) == (63, False)
    scheduler.update_state_after_alloc(request, blocks, 0)
    assert scheduler.build_connector_meta(None).blocks_by_request == {}

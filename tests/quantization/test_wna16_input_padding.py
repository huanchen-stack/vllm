# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""VLLM_MARLIN_INPUT_PADDING: pad a symmetric groupwise WNA16 layer's K to a Marlin
tile multiple with one all-zero group and a unit scale (CPU tests)."""

import pytest
import torch
from torch import nn

from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_wNa16 as wna16_mod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa: E501
    CompressedTensorsWNA16,
)

NEMOTRON_K, NEMOTRON_N, GROUP = (
    15680,
    4480,
    64,
)  # mixer.down_proj of Nemotron-Nano-9B-v2
PADDED_K = 15744

# The module-scoped distributed env below is torn down once at module end
# instead of by the per-test cleanup fixture in tests/conftest.py.
pytestmark = pytest.mark.skip_global_cleanup


class _FakeKernel:
    def __init__(self, config, **kwargs):
        self.config = config


@pytest.fixture(scope="module")
def dist_env():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.utils.network_utils import get_open_port

    with set_current_vllm_config(VllmConfig()):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            local_rank=0,
            backend="gloo",
        )
        ensure_model_parallel_initialized(1, 1)
    yield
    cleanup_dist_env_and_memory()


@pytest.fixture
def marlin_only(monkeypatch, dist_env):
    # Kernel selection queries the device capability; keep the test CPU-only.
    monkeypatch.setattr(wna16_mod, "choose_mp_linear_kernel", lambda cfg: _FakeKernel)
    monkeypatch.setattr(wna16_mod, "MarlinLinearKernel", _FakeKernel)


def _noop_loader(param, loaded_weight, *args, **kwargs):
    param.data.copy_(loaded_weight)


def _create(
    monkeypatch, enabled, k=NEMOTRON_K, n=NEMOTRON_N, group=GROUP, tp_split=False
):
    monkeypatch.setattr(wna16_mod.envs, "VLLM_MARLIN_INPUT_PADDING", enabled)
    scheme = CompressedTensorsWNA16(
        strategy="group", num_bits=4, group_size=group, symmetric=True, layer_name="l"
    )
    layer = nn.Module()
    scheme.create_weights(
        layer,
        output_size=n,
        input_size=k * (2 if tp_split else 1),
        output_partition_sizes=[n],
        input_size_per_partition=k,
        params_dtype=torch.bfloat16,
        weight_loader=_noop_loader,
    )
    return scheme, layer


# ----------------------------------------------------------------------------- #
# create_weights                                                                 #
# ----------------------------------------------------------------------------- #


def test_padding_off_by_default_keeps_vanilla_shapes(monkeypatch, marlin_only):
    scheme, layer = _create(monkeypatch, enabled=False)
    assert scheme.input_padding == 0
    assert tuple(layer.weight_packed.shape) == (NEMOTRON_N, NEMOTRON_K // 8)
    assert tuple(layer.weight_scale.shape) == (NEMOTRON_N, NEMOTRON_K // GROUP)
    assert layer.weight_packed.weight_loader is _noop_loader
    assert scheme.kernel.config.partition_weight_shape == (NEMOTRON_K, NEMOTRON_N)


def test_padding_pads_nemotron_down_proj_to_128_multiple(monkeypatch, marlin_only):
    scheme, layer = _create(monkeypatch, enabled=True)
    assert scheme.input_padding == 64
    assert tuple(layer.weight_packed.shape) == (NEMOTRON_N, PADDED_K // 8)
    assert tuple(layer.weight_scale.shape) == (NEMOTRON_N, PADDED_K // GROUP)
    assert scheme.kernel.config.partition_weight_shape == (PADDED_K, NEMOTRON_N)
    assert scheme.kernel.config.full_weight_shape == (PADDED_K, NEMOTRON_N)


@pytest.mark.parametrize("k", [8192, 10240, 15360, 21504])
def test_aligned_k_is_never_padded(monkeypatch, marlin_only, k):
    # Phi-4-mini (8192) and every Gemma-4 size are already 128-aligned.
    scheme, layer = _create(monkeypatch, enabled=True, k=k)
    assert scheme.input_padding == 0
    assert tuple(layer.weight_packed.shape) == (NEMOTRON_N, k // 8)


def test_padding_must_append_whole_groups(monkeypatch, marlin_only):
    with pytest.raises(ValueError, match="complete quantization groups"):
        _create(monkeypatch, enabled=True, group=128)


def test_row_parallel_shards_fall_through_unpadded(monkeypatch, marlin_only):
    # TP>1 row-parallel: input_size != input_size_per_partition -> no padding
    # (documented fallthrough to the unpadded kernel choice).
    scheme, layer = _create(monkeypatch, enabled=True, tp_split=True)
    assert scheme.input_padding == 0
    assert tuple(layer.weight_packed.shape) == (NEMOTRON_N, NEMOTRON_K // 8)


def test_unsupported_schemes_are_not_padded(monkeypatch, marlin_only):
    monkeypatch.setattr(wna16_mod.envs, "VLLM_MARLIN_INPUT_PADDING", True)
    from compressed_tensors.quantization import ActivationOrdering

    for kwargs in (
        {"symmetric": False},
        {"actorder": ActivationOrdering.GROUP},
    ):
        scheme = CompressedTensorsWNA16(
            strategy="group", num_bits=4, group_size=GROUP, layer_name="l", **kwargs
        )
        layer = nn.Module()
        scheme.create_weights(
            layer,
            output_size=NEMOTRON_N,
            input_size=NEMOTRON_K,
            output_partition_sizes=[NEMOTRON_N],
            input_size_per_partition=NEMOTRON_K,
            params_dtype=torch.bfloat16,
            weight_loader=_noop_loader,
        )
        assert scheme.input_padding == 0


# ----------------------------------------------------------------------------- #
# loaders (parameter identity, not shape heuristics)                             #
# ----------------------------------------------------------------------------- #


def _loader_capture():
    captured = []

    def loader(param, loaded_weight, *args, **kwargs):
        captured.append((loaded_weight.clone(), args, kwargs))
        assert tuple(loaded_weight.shape) == tuple(param.shape)

    return loader, captured


def test_packed_weight_loader_pads_with_zero_columns():
    loader, captured = _loader_capture()
    packed = torch.arange(4 * 1960, dtype=torch.int32).reshape(4, 1960)
    target = nn.Parameter(torch.empty(4, 1968, dtype=torch.int32), requires_grad=False)
    CompressedTensorsWNA16._load_with_input_padding(
        target,
        packed,
        "shard-arg",
        weight_loader=loader,
        kind="weight_packed",
        original_input_size=NEMOTRON_K,
        padded_input_size=PADDED_K,
        pack_factor=8,
        group_size=GROUP,
        extra="kw",
    )
    value, args, kwargs = captured[-1]
    assert torch.equal(value[:, :1960], packed)
    assert torch.count_nonzero(value[:, 1960:]) == 0
    assert args == ("shard-arg",) and kwargs == {"extra": "kw"}


def test_scale_loader_pads_with_unit_scale_group():
    loader, captured = _loader_capture()
    scales = torch.full((4, 245), 0.25, dtype=torch.bfloat16)
    target = nn.Parameter(
        torch.empty(4, 246, dtype=torch.bfloat16), requires_grad=False
    )
    CompressedTensorsWNA16._load_with_input_padding(
        target,
        scales,
        weight_loader=loader,
        kind="weight_scale",
        original_input_size=NEMOTRON_K,
        padded_input_size=PADDED_K,
        pack_factor=8,
        group_size=GROUP,
    )
    value = captured[-1][0]
    assert torch.equal(value[:, :245], scales)
    assert torch.all(value[:, 245:] == 1)


def test_shape_loader_rewrites_only_the_input_dimension():
    loader, captured = _loader_capture()
    shape = torch.tensor([NEMOTRON_N, NEMOTRON_K], dtype=torch.int64)
    target = nn.Parameter(torch.empty(2, dtype=torch.int64), requires_grad=False)
    CompressedTensorsWNA16._load_with_input_padding(
        target,
        shape,
        weight_loader=loader,
        kind="weight_shape",
        original_input_size=NEMOTRON_K,
        padded_input_size=PADDED_K,
        pack_factor=8,
        group_size=GROUP,
    )
    assert captured[-1][0].tolist() == [NEMOTRON_N, PADDED_K]
    # A shape that does not match the original K is a checkpoint mismatch.
    with pytest.raises(ValueError, match="weight_shape"):
        CompressedTensorsWNA16._load_with_input_padding(
            target,
            torch.tensor([NEMOTRON_N, 1234], dtype=torch.int64),
            weight_loader=loader,
            kind="weight_shape",
            original_input_size=NEMOTRON_K,
            padded_input_size=PADDED_K,
            pack_factor=8,
            group_size=GROUP,
        )


def test_already_padded_tensor_is_passed_through():
    loader, captured = _loader_capture()
    packed = torch.zeros(4, 1968, dtype=torch.int32)
    target = nn.Parameter(torch.empty(4, 1968, dtype=torch.int32), requires_grad=False)
    CompressedTensorsWNA16._load_with_input_padding(
        target,
        packed,
        weight_loader=loader,
        kind="weight_packed",
        original_input_size=NEMOTRON_K,
        padded_input_size=PADDED_K,
        pack_factor=8,
        group_size=GROUP,
    )
    assert torch.equal(captured[-1][0], packed)


def test_apply_weights_pads_activations(monkeypatch, marlin_only):
    scheme, layer = _create(monkeypatch, enabled=True)
    seen = {}

    def fake_apply(layer, x, bias):
        seen["shape"] = tuple(x.shape)
        seen["tail_zero"] = bool(torch.count_nonzero(x[:, NEMOTRON_K:]) == 0)
        return x

    scheme.kernel.apply_weights = fake_apply
    scheme.apply_weights(layer, torch.ones(3, NEMOTRON_K), None)
    assert seen == {"shape": (3, PADDED_K), "tail_zero": True}

    scheme_off, layer_off = _create(monkeypatch, enabled=False)
    scheme_off.kernel.apply_weights = fake_apply
    scheme_off.apply_weights(layer_off, torch.ones(3, NEMOTRON_K), None)
    assert seen["shape"] == (3, NEMOTRON_K)

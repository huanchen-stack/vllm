# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the single-adapter rollout LoRA fast path (ROLLOUT_QLORA).

Design doc: docs/design/rollout_lora_fastpath.md.

Unit tier (no GPU work beyond process-group init):
  * custom-op fake implementation matches the real one (dtype/shape),
  * packed A/B buffer layout (slices at multiples of max_lora_rank),
  * adapter-slot selection table,
  * lazy Punica metadata preparation with a mocked prepare_tensors.

GPU tier:
  * fast path (fused and per-slice) vs the Punica reference for every LoRA
    linear layer class, including the 4-slice variable-slice layer,
  * no host sync after warm-up under torch.cuda.set_sync_debug_mode("error"),
  * kernel presence via torch.profiler (no Punica kernels, GEMMs present),
  * CUDA-graph capture/replay identical to eager,
  * zero-B adapter equals the base layer bit-exactly,
  * base_forward_override contract and LoRA-first dual-stream ordering.
"""

from dataclasses import dataclass
from importlib import reload
from unittest.mock import MagicMock

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.config.lora import LoRAConfig
from vllm.forward_context import set_forward_context
from vllm.lora.layers import (
    ColumnParallelLinearWithLoRA,
    LoRAMapping,
    MergedColumnParallelLinearVariableSliceWithLoRA,
    MergedColumnParallelLinearWithLoRA,
    MergedQKVParallelLinearWithLoRA,
    QKVParallelLinearWithLoRA,
    ReplicatedLinearWithLoRA,
    RowParallelLinearWithLoRA,
)
from vllm.lora.ops.torch_ops.rollout_lora_ops import (
    _rollout_lora_matmul_fake,
    rollout_lora_matmul,
)
from vllm.lora.punica_wrapper.punica_gpu import (
    PunicaWrapperGPU,
    select_rollout_single_lora_index,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

TOLERANCES = {
    torch.float16: (5e-3, 5e-3),
    torch.float32: (5e-3, 5e-3),
    torch.bfloat16: (3e-2, 2e-2),
}

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="rollout fast path is CUDA only"
)

DEVICE = "cuda:0"
HIDDEN = 512
MAX_LORAS = 2
SLOT = 1  # the adapter lives in slot 1 so that slot selection is non-trivial
LORA_ID = 7


@pytest.fixture(autouse=True)
def _reset_default_device(reset_default_device):
    yield


# --------------------------------------------------------------------------
# unit tier
# --------------------------------------------------------------------------


def test_rollout_lora_matmul_fake_matches_real():
    # The op is registered for the platform dispatch key (CUDA) plus Meta.
    x = torch.rand(4, 32, dtype=torch.float16, device=DEVICE)
    lora_a = torch.rand(8, 32, dtype=torch.float32, device=DEVICE)
    lora_b = torch.rand(16, 8, dtype=torch.float32, device=DEVICE)
    real = rollout_lora_matmul(x, lora_a, lora_b, 16, 1.0)
    fake = _rollout_lora_matmul_fake(x, lora_a, lora_b, 16, 1.0)
    assert real.shape == fake.shape == (4, 16)
    assert real.dtype == fake.dtype == torch.float32
    torch.testing.assert_close(real, x.float() @ lora_a.T @ lora_b.T)
    scaled = rollout_lora_matmul(x, lora_a, lora_b, 16, 0.5)
    torch.testing.assert_close(scaled, (x.float() @ lora_a.T) * 0.5 @ lora_b.T)

    with FakeTensorMode():
        fx = torch.empty(4, 32, dtype=torch.float16)
        fa = torch.empty(8, 32, dtype=torch.bfloat16)
        fb = torch.empty(16, 8, dtype=torch.bfloat16)
        out = torch.ops.vllm.rollout_lora_matmul(fx, fa, fb, 16, 1.0)
        assert out.shape == (4, 16)
        assert out.dtype == torch.bfloat16


@pytest.mark.parametrize(
    ("index_mapping", "lora_index_to_id", "max_loras", "expected"),
    [
        # empty mapping: profile / dummy run
        ((), [None], 1, 0),
        ((), [None, None], 2, None),
        # exactly one positive id -> its slot
        ((7, 7, 7), [None, 7], 2, 1),
        ((7, 0, 7), [None, 7], 2, 1),
        ((7, 7), [3], 1, None),  # id not loaded
        # two distinct ids -> Punica
        ((7, 8), [7, 8], 2, None),
        # only zeros
        ((0, 0), [7], 1, 0),
        ((0, 0), [None, 7], 2, 1),
        ((0, 0), [None, None], 2, None),
    ],
)
def test_select_rollout_single_lora_index(
    index_mapping, lora_index_to_id, max_loras, expected
):
    assert (
        select_rollout_single_lora_index(index_mapping, lora_index_to_id, max_loras)
        == expected
    )


def _make_wrapper(lora_config: LoRAConfig) -> PunicaWrapperGPU:
    return PunicaWrapperGPU(1024, 64, DEVICE, lora_config=lora_config)


def _spy_prepare(wrapper: PunicaWrapperGPU) -> tuple[MagicMock, MagicMock]:
    token_spy = MagicMock(wraps=wrapper.token_mapping_meta.prepare_tensors)
    prompt_spy = MagicMock(wraps=wrapper.prompt_mapping_meta.prepare_tensors)
    wrapper.token_mapping_meta.prepare_tensors = token_spy
    wrapper.prompt_mapping_meta.prepare_tensors = prompt_spy
    return token_spy, prompt_spy


@pytest.mark.parametrize("rollout_qlora", ["0", "1"])
def test_lazy_punica_metadata(dist_init, monkeypatch, rollout_qlora):
    monkeypatch.setenv("ROLLOUT_QLORA", rollout_qlora)
    lora_config = LoRAConfig(
        max_loras=MAX_LORAS, max_lora_rank=8, lora_dtype=torch.float16
    )
    wrapper = _make_wrapper(lora_config)
    token_spy, prompt_spy = _spy_prepare(wrapper)
    id_to_index: list[int | None] = [None] * MAX_LORAS
    id_to_index[SLOT] = LORA_ID

    single = LoRAMapping([LORA_ID] * 8, [LORA_ID] * 2, is_prefill=False)
    wrapper.update_metadata(single, id_to_index, MAX_LORAS, 512)
    if rollout_qlora == "1":
        assert wrapper._rollout_single_lora_index == SLOT
        assert token_spy.call_count == prompt_spy.call_count == 0
        # A Punica entry point prepares it exactly once, lazily.
        wrapper._ensure_punica_metadata_prepared()
        wrapper._ensure_punica_metadata_prepared()
        assert token_spy.call_count == prompt_spy.call_count == 1
    else:
        assert wrapper._rollout_single_lora_index is None
        assert token_spy.call_count == prompt_spy.call_count == 1

    # A multi-adapter batch always prepares eagerly (vanilla behaviour).
    id_to_index[0] = LORA_ID + 1
    mixed = LoRAMapping([LORA_ID] * 4 + [LORA_ID + 1] * 4, [LORA_ID, LORA_ID + 1])
    token_spy.reset_mock()
    prompt_spy.reset_mock()
    wrapper.update_metadata(mixed, id_to_index, MAX_LORAS, 512)
    assert wrapper._rollout_single_lora_index is None
    assert token_spy.call_count == prompt_spy.call_count == 1


def test_profile_run_mapping_uses_fast_path(dist_init, monkeypatch):
    """Single slot, no adapter loaded, all-zero / empty dummy mapping.

    This is what profile_run and CUDA-graph capture (num_active_loras=0)
    present; the manager passes lora_slots + 1 as max_loras, so the table
    must key on the configured slot count and keep the torch path.
    """
    monkeypatch.setenv("ROLLOUT_QLORA", "1")
    lora_config = LoRAConfig(max_loras=1, max_lora_rank=8, lora_dtype=torch.float16)
    wrapper = _make_wrapper(lora_config)
    token_spy, _ = _spy_prepare(wrapper)
    for mapping in (
        LoRAMapping([0] * 8, [0] * 2, is_prefill=True),
        LoRAMapping([], [], is_prefill=True),
    ):
        wrapper.update_metadata(mapping, [None], 2, 512)
        assert wrapper._rollout_single_lora_index == 0
    assert token_spy.call_count == 0


@dataclass
class FakeConfig:
    hidden_size = HIDDEN
    num_key_value_heads = 4
    num_attention_heads = 8


def _make_layer(kind: str, dtype: torch.dtype, prefix: str):
    """Build (base_linear, lora_layer, output_slices) for a layer class."""
    kw = dict(bias=False, params_dtype=dtype, prefix=prefix)
    if kind == "column":
        base = ColumnParallelLinear(HIDDEN, HIDDEN, **kw)
        lora = ColumnParallelLinearWithLoRA(base)
    elif kind == "row":
        base = RowParallelLinear(HIDDEN, HIDDEN, **kw)
        lora = RowParallelLinearWithLoRA(base)
    elif kind == "replicated":
        base = ReplicatedLinear(HIDDEN, HIDDEN, **kw)
        lora = ReplicatedLinearWithLoRA(base)
    elif kind == "merged_column":
        base = MergedColumnParallelLinear(HIDDEN, [HIDDEN, HIDDEN], **kw)
        lora = MergedColumnParallelLinearWithLoRA(base)
    elif kind == "qkv":
        base = QKVParallelLinear(HIDDEN, 64, 8, 4, **kw)
        lora = QKVParallelLinearWithLoRA(base)
    elif kind == "merged_qkv":
        base = QKVParallelLinear(HIDDEN, 64, 8, 4, **kw)
        lora = MergedQKVParallelLinearWithLoRA(base)
    elif kind == "variable_slice":
        # Qwen3.5 in_proj_qkvz-like: q, k, v, z with unequal sizes.
        base = MergedColumnParallelLinear(HIDDEN, [256, 256, 512, 512], **kw)
        lora = MergedColumnParallelLinearVariableSliceWithLoRA(base)
    else:
        raise ValueError(kind)
    base.weight.data = torch.rand_like(base.weight.data) - 0.5
    return base, lora


LAYER_KINDS = [
    "column",
    "row",
    "replicated",
    "merged_column",
    "qkv",
    "merged_qkv",
    "variable_slice",
]


def _random_adapter(lora, rank: int, dtype: torch.dtype, seed: int):
    """Random per-slice (A, B) for the layer, sized for its output slices."""
    gen = torch.Generator(device=DEVICE).manual_seed(seed)
    slices = tuple(lora.output_slices)
    a_list, b_list = [], []
    for out_size in slices:
        a = torch.rand(rank, HIDDEN, generator=gen, device=DEVICE) - 0.5
        b = torch.rand(out_size, rank, generator=gen, device=DEVICE) - 0.5
        a_list.append(a.to(dtype))
        b_list.append((b * 0.1).to(dtype))
    return a_list, b_list


def _set_adapter(lora, a_list, b_list):
    if isinstance(lora, MergedColumnParallelLinearVariableSliceWithLoRA):
        # Single A shared by all slices, B concatenated (vanilla API).
        lora.set_lora(SLOT, a_list[0], torch.cat(b_list, dim=0))
        return [a_list[0]] * len(b_list), b_list
    if lora.n_slices > 1:
        lora.set_lora(SLOT, a_list, b_list)
    else:
        lora.set_lora(SLOT, a_list[0], b_list[0])
    return a_list, b_list


def _reference(base, x, a_list, b_list):
    out = base(x)[0].clone()
    offset = 0
    for a, b in zip(a_list, b_list):
        size = b.shape[0]
        out[:, offset : offset + size] += x @ a.T @ b.T
        offset += size
    return out


def _single_adapter_mapping(num_tokens: int) -> tuple[LoRAMapping, list]:
    id_to_index: list[int | None] = [None] * MAX_LORAS
    id_to_index[SLOT] = LORA_ID
    mapping = LoRAMapping([LORA_ID] * num_tokens, [LORA_ID] * num_tokens)
    return mapping, id_to_index


def _build(kind, dtype, lora_config, prefix, seed=0):
    """Wrapper + layer + adapter, ready for forward with a single adapter."""
    set_random_seed(seed)
    wrapper = _make_wrapper(lora_config)
    base, lora = _make_layer(kind, dtype, prefix)
    model_config = FakeConfig() if kind in ("qkv", "merged_qkv") else None
    lora.create_lora_weights(MAX_LORAS, lora_config, model_config=model_config)
    lora.set_mapping(wrapper)
    a_list, b_list = _random_adapter(lora, lora_config.max_lora_rank, dtype, seed)
    a_list, b_list = _set_adapter(lora, a_list, b_list)
    return wrapper, base, lora, a_list, b_list


@pytest.mark.parametrize("kind", ["merged_qkv", "variable_slice"])
def test_packed_buffer_layout(dist_init, default_vllm_config, monkeypatch, kind):
    monkeypatch.setenv("ROLLOUT_QLORA", "1")
    torch.set_default_device(DEVICE)
    max_rank = 8
    lora_config = LoRAConfig(
        max_loras=MAX_LORAS, max_lora_rank=max_rank, lora_dtype=torch.float16
    )
    wrapper, base, lora, a_list, b_list = _build(
        kind, torch.float16, lora_config, f"layout_{kind}"
    )
    n = lora.n_slices
    assert n in (3, 4)
    assert lora.rollout_lora_a_stacked.shape == (MAX_LORAS, 1, n * max_rank, HIDDEN)
    assert lora.rollout_lora_b_stacked.shape == (
        MAX_LORAS,
        1,
        sum(lora.output_slices),
        n * max_rank,
    )
    packed_a = lora.rollout_lora_a_stacked[SLOT, 0]
    packed_b = lora.rollout_lora_b_stacked[SLOT, 0]
    # Slice s occupies rank rows [s*R, (s+1)*R): A concatenated along rank,
    # B block-diagonal, straight from the (zero-padded) stacked buffers.
    expected_a = torch.cat([lora.lora_a_stacked[s][SLOT, 0] for s in range(n)])
    expected_b = torch.block_diag(*[lora.lora_b_stacked[s][SLOT, 0] for s in range(n)])
    assert torch.equal(packed_a, expected_a)
    assert torch.equal(packed_b, expected_b)
    # And those stacked buffers hold the adapter given to set_lora.
    for s in range(n):
        assert torch.equal(packed_a[s * max_rank : (s + 1) * max_rank], a_list[s])
    # Other slots stay zero; reset_lora zeroes the packed buffers too.
    assert not lora.rollout_lora_a_stacked[1 - SLOT].any()
    lora.reset_lora(SLOT)
    assert not lora.rollout_lora_a_stacked[SLOT].any()
    assert not lora.rollout_lora_b_stacked[SLOT].any()

    if kind == "merged_qkv":
        # Sub-rank adapter and a None slice (packed group expansion path).
        small_a = [a[:4] if s != 1 else None for s, a in enumerate(a_list)]
        small_b = [b[:, :4] if s != 1 else None for s, b in enumerate(b_list)]
        lora.set_lora(SLOT, small_a, small_b)
        packed_a = lora.rollout_lora_a_stacked[SLOT, 0]
        packed_b = lora.rollout_lora_b_stacked[SLOT, 0]
        assert torch.equal(packed_a[0:4], small_a[0])
        assert not packed_a[4 : 2 * max_rank].any()  # padding + None slice
        off = lora.output_slices[0]
        assert torch.equal(packed_b[:off, 0:4], small_b[0])
        assert not packed_b[off : off + lora.output_slices[1]].any()


def test_vanilla_layer_has_no_packed_buffers(
    dist_init, default_vllm_config, monkeypatch
):
    monkeypatch.delenv("ROLLOUT_QLORA", raising=False)
    torch.set_default_device(DEVICE)
    lora_config = LoRAConfig(
        max_loras=MAX_LORAS, max_lora_rank=8, lora_dtype=torch.float16
    )
    wrapper, base, lora, _, _ = _build("merged_qkv", torch.float16, lora_config, "van")
    assert lora.rollout_lora_a_stacked is None
    assert lora._rollout_lora_kwargs() == {}
    assert not wrapper._rollout_fast_path_enabled


# --------------------------------------------------------------------------
# GPU tier
# --------------------------------------------------------------------------


def _forward(wrapper, lora, x):
    mapping, id_to_index = _single_adapter_mapping(x.shape[0])
    wrapper.update_metadata(mapping, id_to_index, MAX_LORAS, 512)
    out = lora(x)
    return out[0] if isinstance(out, tuple) else out


MODES = {
    "punica": {"ROLLOUT_QLORA": "0"},
    "torch": {"ROLLOUT_QLORA": "1", "VLLM_ROLLOUT_LORA_FUSE_PACKED": "0"},
    "torch-fused": {"ROLLOUT_QLORA": "1", "VLLM_ROLLOUT_LORA_FUSE_PACKED": "1"},
}


@torch.inference_mode()
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", LAYER_KINDS)
def test_fastpath_matches_punica_reference(
    dist_init, default_vllm_config, monkeypatch, kind, dtype
):
    torch.set_default_device(DEVICE)
    lora_config = LoRAConfig(max_loras=MAX_LORAS, max_lora_rank=16, lora_dtype=dtype)
    num_tokens = 48
    results = {}
    for mode, env in MODES.items():
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        wrapper, base, lora, a_list, b_list = _build(
            kind, dtype, lora_config, f"ref_{kind}_{mode}"
        )
        set_random_seed(1)
        x = torch.rand(num_tokens, HIDDEN, dtype=dtype) - 0.5
        out = _forward(wrapper, lora, x)
        if mode == "punica":
            assert wrapper._rollout_single_lora_index is None
        else:
            assert wrapper._rollout_single_lora_index == SLOT
            assert wrapper._rollout_fast_path_logged
            assert not wrapper._rollout_fallback_logged
        expected = _reference(base, x, a_list, b_list)
        rtol, atol = TOLERANCES[dtype]
        torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)
        results[mode] = out

        # After reset the LoRA contribution is gone.
        lora.reset_lora(SLOT)
        out = _forward(wrapper, lora, x)
        torch.testing.assert_close(out, base(x)[0], rtol=rtol, atol=atol)

    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(
        results["torch"], results["punica"], rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        results["torch-fused"], results["punica"], rtol=rtol, atol=atol
    )


@torch.inference_mode()
def test_mixed_batch_falls_back_to_punica(dist_init, default_vllm_config, monkeypatch):
    monkeypatch.setenv("ROLLOUT_QLORA", "1")
    torch.set_default_device(DEVICE)
    dtype = torch.float16
    lora_config = LoRAConfig(max_loras=MAX_LORAS, max_lora_rank=16, lora_dtype=dtype)
    wrapper, base, lora, a_list, b_list = _build(
        "merged_qkv", dtype, lora_config, "mix"
    )
    # second adapter in slot 0
    a2, b2 = _random_adapter(lora, 16, dtype, seed=5)
    lora.set_lora(0, a2, b2)
    id_to_index = [LORA_ID + 1, LORA_ID]
    x = torch.rand(8, HIDDEN, dtype=dtype) - 0.5
    mapping = LoRAMapping([LORA_ID] * 4 + [LORA_ID + 1] * 4, [LORA_ID, LORA_ID + 1])
    wrapper.update_metadata(mapping, id_to_index, MAX_LORAS, 512)
    assert wrapper._rollout_single_lora_index is None
    out = lora(x)[0]
    assert wrapper._rollout_fallback_logged
    expected = torch.cat(
        [_reference(base, x[:4], a_list, b_list), _reference(base, x[4:], a2, b2)]
    )
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(out, expected, rtol=rtol, atol=atol)


@torch.inference_mode()
def test_zero_adapter_equals_base_exactly(dist_init, default_vllm_config, monkeypatch):
    monkeypatch.setenv("ROLLOUT_QLORA", "1")
    torch.set_default_device(DEVICE)
    dtype = torch.bfloat16
    lora_config = LoRAConfig(max_loras=MAX_LORAS, max_lora_rank=16, lora_dtype=dtype)
    wrapper, base, lora, a_list, b_list = _build(
        "variable_slice", dtype, lora_config, "z"
    )
    lora.set_lora(SLOT, a_list[0], torch.zeros_like(torch.cat(b_list)))
    x = torch.rand(16, HIDDEN, dtype=dtype) - 0.5
    out = _forward(wrapper, lora, x)
    assert wrapper._rollout_single_lora_index == SLOT
    assert torch.equal(out, base(x)[0])


@torch.inference_mode()
@pytest.mark.parametrize("rollout_qlora", ["0", "1"])
def test_no_host_sync_after_warmup(
    dist_init, default_vllm_config, monkeypatch, rollout_qlora
):
    monkeypatch.setenv("ROLLOUT_QLORA", rollout_qlora)
    torch.set_default_device(DEVICE)
    dtype = torch.float16
    lora_config = LoRAConfig(max_loras=MAX_LORAS, max_lora_rank=16, lora_dtype=dtype)
    wrapper, base, lora, _, _ = _build("merged_qkv", dtype, lora_config, "sync")
    x = torch.rand(4, HIDDEN, dtype=dtype)
    # warm-up: first call compiles the torch op and may synchronize
    _forward(wrapper, lora, x)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        if rollout_qlora == "1":
            _forward(wrapper, lora, x)
        else:
            # Punica metadata preparation carries a D2H sync (torch.all ->
            # host flag); this is the sync the fast path removes.
            with pytest.raises(RuntimeError, match="synchroniz"):
                _forward(wrapper, lora, x)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    torch.cuda.synchronize()


def _kernel_names(fn) -> list[str]:
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return [e.key for e in prof.key_averages() if e.device_type.name == "CUDA"]


@torch.inference_mode()
@pytest.mark.parametrize("rollout_qlora", ["0", "1"])
def test_kernel_presence(dist_init, default_vllm_config, monkeypatch, rollout_qlora):
    monkeypatch.setenv("ROLLOUT_QLORA", rollout_qlora)
    torch.set_default_device(DEVICE)
    dtype = torch.float16
    lora_config = LoRAConfig(max_loras=MAX_LORAS, max_lora_rank=16, lora_dtype=dtype)
    wrapper, base, lora, _, _ = _build("merged_qkv", dtype, lora_config, "prof")
    x = torch.rand(4, HIDDEN, dtype=dtype)
    _forward(wrapper, lora, x)  # warm-up / compile
    names = _kernel_names(lambda: _forward(wrapper, lora, x))
    punica = [
        n for n in names if "_lora_shrink_kernel" in n or "_lora_expand_kernel" in n
    ]
    gemm = [n for n in names if "gemm" in n.lower() or "cutlass" in n.lower()]
    if rollout_qlora == "1":
        assert not punica, punica
        assert gemm, names
    else:
        assert len(punica) >= 2, names


@torch.inference_mode()
def test_cudagraph_replay_matches_eager(dist_init, default_vllm_config, monkeypatch):
    monkeypatch.setenv("ROLLOUT_QLORA", "1")
    torch.set_default_device(DEVICE)
    dtype = torch.bfloat16
    lora_config = LoRAConfig(max_loras=MAX_LORAS, max_lora_rank=16, lora_dtype=dtype)
    wrapper, base, lora, _, _ = _build("variable_slice", dtype, lora_config, "graph")
    static_x = torch.rand(8, HIDDEN, dtype=dtype) - 0.5
    mapping, id_to_index = _single_adapter_mapping(8)
    wrapper.update_metadata(mapping, id_to_index, MAX_LORAS, 512)
    eager = lora(static_x)[0].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            lora(static_x)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_out = lora(static_x)[0]
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, eager)
    # New input through the same graph equals eager on that input.
    new_x = torch.rand(8, HIDDEN, dtype=dtype) - 0.5
    static_x.copy_(new_x)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(static_out, lora(new_x)[0])


@torch.inference_mode()
def test_base_forward_override(dist_init, default_vllm_config, monkeypatch):
    monkeypatch.setenv("ROLLOUT_QLORA", "1")
    torch.set_default_device(DEVICE)
    dtype = torch.float16
    lora_config = LoRAConfig(max_loras=MAX_LORAS, max_lora_rank=16, lora_dtype=dtype)
    wrapper, base, lora, a_list, b_list = _build(
        "merged_qkv", dtype, lora_config, "ovr"
    )
    x = torch.rand(8, HIDDEN, dtype=dtype) - 0.5
    reference = _reference(base, x, a_list, b_list)

    def override(x_, bias):
        return base.quant_method.apply(base, x_, bias) * 2

    spy = MagicMock(wraps=override)
    lora.set_base_forward_override(spy)
    assert lora.base_forward_override is spy
    out = _forward(wrapper, lora, x)
    spy.assert_called_once()
    rtol, atol = TOLERANCES[dtype]
    torch.testing.assert_close(out, reference + base(x)[0], rtol=rtol, atol=atol)
    lora.set_base_forward_override(None)
    out = _forward(wrapper, lora, x)
    torch.testing.assert_close(out, reference, rtol=rtol, atol=atol)


@torch.inference_mode()
def test_dual_stream_lora_first_and_override_precedence(
    dist_init, default_vllm_config, monkeypatch
):
    monkeypatch.setenv("ROLLOUT_QLORA", "1")
    monkeypatch.setenv("VLLM_LORA_ENABLE_DUAL_STREAM", "1")
    import vllm.lora.layers.base_linear as base_linear_mod

    if not hasattr(base_linear_mod, "lora_linear_async"):
        reload(base_linear_mod)  # registers torch.ops.vllm.lora_linear_async
    torch.set_default_device(DEVICE)
    dtype = torch.float16
    lora_config = LoRAConfig(max_loras=MAX_LORAS, max_lora_rank=16, lora_dtype=dtype)
    wrapper, base, lora, a_list, b_list = _build("column", dtype, lora_config, "dual")
    assert lora._enable_aux_cuda_stream
    assert lora._rollout_lora_enabled
    x = torch.rand(8, HIDDEN, dtype=dtype) - 0.5
    reference = _reference(base, x, a_list, b_list)
    rtol, atol = TOLERANCES[dtype]

    # The registered op resolves the layer through the forward context.
    default_vllm_config.compilation_config.static_forward_context[lora.layer_name] = (
        lora
    )
    async_spy = MagicMock(wraps=lora._apply_async_impl)
    exec_spy = MagicMock(wraps=lora._execute_lora_async)
    lora._apply_async_impl = async_spy
    lora._execute_lora_async = exec_spy
    with set_forward_context(None, default_vllm_config):
        out = _forward(wrapper, lora, x)
    async_spy.assert_called_once()
    exec_spy.assert_called_once()
    assert exec_spy.call_args.kwargs["lora_first"] is True
    torch.testing.assert_close(out, reference, rtol=rtol, atol=atol)

    # With an override installed the dual-stream op is bypassed entirely.
    async_spy.reset_mock()
    lora.set_base_forward_override(
        lambda x_, bias: base.quant_method.apply(base, x_, bias)
    )
    with set_forward_context(None, default_vllm_config):
        out = _forward(wrapper, lora, x)
    async_spy.assert_not_called()
    torch.testing.assert_close(out, reference, rtol=rtol, atol=atol)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU gates for the custom-op binder (GEMMA4 audit tests 2-4).

Run through the decision-13 launcher on one GPU. The base linears are
``FakeLinear`` stand-ins (real ``LinearBase`` subclasses whose "INT4" twin
simply carries different weights): the tests check the binder mechanics
(one compiled graph, one CUDA graph per precision, shared LoRA delta), not
the Marlin kernel, which the Qwen3.5-9B shadow validation covers.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.dual_precision import (
    BASE_PRECISION_BF16,
    BASE_PRECISION_INT4,
    bind_dual_precision,
    get_binding,
)
from vllm.model_executor.dual_precision.loader import attach_shadow_layers

from .fakes import FakeModel, build_int4_model, build_wrapped_model

pytestmark = pytest.mark.gpu_smoke

if not torch.cuda.is_available():
    pytest.skip("needs a GPU", allow_module_level=True)

NAMES = ["model.layers.0.mlp.gate_up_proj", "model.layers.0.mlp.down_proj"]
HIDDEN = 64
DEVICE = "cuda"
DTYPE = torch.bfloat16


class Block(nn.Module):
    """gate_up -> silu -> down, the shape of one compiled MLP block."""

    def __init__(self, model: FakeModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.model.get_submodule(NAMES[0])(x)
        return self.model.get_submodule(NAMES[1])(F.silu(up))


def _reference(block: Block, x: torch.Tensor, precision: str) -> torch.Tensor:
    """Eager math with the weights that ``precision`` should select."""
    h = x
    for name in NAMES:
        wrapper = block.model.get_submodule(name)
        binding = get_binding(wrapper)
        base = (
            binding.int4_or_fallback
            if precision == BASE_PRECISION_INT4
            else binding.bf16
        )
        out = F.linear(h, base.weight) + F.linear(
            F.linear(h, wrapper.lora_a), wrapper.lora_b
        )
        h = F.silu(out) if name == NAMES[0] else out
    return h


@pytest.fixture
def block_and_config():
    torch.manual_seed(0)
    model = build_wrapped_model(NAMES, HIDDEN, device=DEVICE, dtype=DTYPE)
    int4_model = build_int4_model(NAMES, set(NAMES), HIDDEN, device=DEVICE, dtype=DTYPE)
    vllm_config = VllmConfig()
    state = attach_shadow_layers(
        model,
        int4_model,
        bf16_layer_policy="none",
        module_policy="all",
        num_layers=1,
        static_forward_context=vllm_config.compilation_config.static_forward_context,
        dtype=DTYPE,
    )
    assert state.attached == 2
    # Unit-variance activations keep bf16 rounding well inside the tolerances.
    for name in NAMES:
        wrapper = model.get_submodule(name)
        wrapper.base_layer.weight.data.mul_(HIDDEN**-0.5)
        get_binding(wrapper).int4_or_fallback.weight.data.mul_(HIDDEN**-0.5)
        wrapper.lora_a.normal_(std=0.1)
        wrapper.lora_b.normal_(std=0.1)
    return Block(model), vllm_config


def test_override_binder_under_torch_compile_one_graph_serves_both_precisions(
    block_and_config,
):
    block, vllm_config = block_and_config
    x = torch.randn(4, HIDDEN, device=DEVICE, dtype=DTYPE)
    torch._dynamo.reset()
    from torch._dynamo.utils import counters

    counters.clear()
    compiled = torch.compile(block, fullgraph=True, dynamic=False)

    with set_forward_context(None, vllm_config):
        outputs = {}
        for precision in (
            BASE_PRECISION_BF16,
            BASE_PRECISION_INT4,
            BASE_PRECISION_BF16,
        ):
            bind_dual_precision(block.model, precision)
            outputs[precision] = compiled(x)
            torch.testing.assert_close(
                outputs[precision],
                _reference(block, x, precision),
                rtol=5e-2,
                atol=5e-2,
            )
    # One Dynamo graph: the precision switch is invisible to torch.compile
    # (no recompilation, no guard on the selected base).
    assert counters["stats"]["unique_graphs"] == 1
    assert not torch.allclose(
        outputs[BASE_PRECISION_BF16], outputs[BASE_PRECISION_INT4]
    )


def test_one_full_cuda_graph_per_precision(block_and_config):
    block, vllm_config = block_and_config
    static_x = torch.randn(4, HIDDEN, device=DEVICE, dtype=DTYPE)
    compiled = torch.compile(block, fullgraph=True, dynamic=False)
    graphs: dict[str, tuple[torch.cuda.CUDAGraph, torch.Tensor]] = {}
    references = {}

    with set_forward_context(None, vllm_config):
        for precision in (BASE_PRECISION_BF16, BASE_PRECISION_INT4):
            bind_dual_precision(block.model, precision)
            references[precision] = _reference(block, static_x, precision)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    compiled(static_x)
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_out = compiled(static_x)
            graphs[precision] = (graph, static_out)

    # Replay each graph twice; each reproduces its own precision regardless of
    # the current bind (the selection was captured, as the dispatcher relies on).
    for precision, (graph, static_out) in graphs.items():
        other = (
            BASE_PRECISION_INT4
            if precision == BASE_PRECISION_BF16
            else BASE_PRECISION_BF16
        )
        bind_dual_precision(block.model, other)
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(
                static_out, references[precision], rtol=5e-2, atol=5e-2
            )
    assert not torch.allclose(
        graphs[BASE_PRECISION_BF16][1], graphs[BASE_PRECISION_INT4][1]
    )


def test_lora_delta_shared_across_precisions(block_and_config):
    block, vllm_config = block_and_config
    wrapper = block.model.get_submodule(NAMES[0])
    binding = get_binding(wrapper)
    x = torch.randn(8, HIDDEN, device=DEVICE, dtype=DTYPE)
    deltas = {}
    with set_forward_context(None, vllm_config):
        for precision, base in (
            (BASE_PRECISION_BF16, binding.bf16),
            (BASE_PRECISION_INT4, binding.int4_or_fallback),
        ):
            bind_dual_precision(block.model, precision)
            deltas[precision] = (wrapper(x) - F.linear(x, base.weight)).float()
    expected = F.linear(F.linear(x, wrapper.lora_a), wrapper.lora_b).float()
    torch.testing.assert_close(
        deltas[BASE_PRECISION_BF16], expected, rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(
        deltas[BASE_PRECISION_INT4], expected, rtol=5e-2, atol=5e-2
    )
    torch.testing.assert_close(
        deltas[BASE_PRECISION_BF16], deltas[BASE_PRECISION_INT4], rtol=5e-2, atol=5e-2
    )


def test_real_lora_wrapper_routes_apply_through_override():
    """Runs once C1's hook is merged; the wrapper must call the installed
    override for the base output and apply LoRA afterwards."""
    from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA

    if not hasattr(BaseLinearLayerWithLoRA, "set_base_forward_override"):
        pytest.skip("BaseLinearLayerWithLoRA.set_base_forward_override (C1) not merged")

    from vllm.config import set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )
    from vllm.lora.layers import ColumnParallelLinearWithLoRA
    from vllm.model_executor.dual_precision.binding import install_binding
    from vllm.model_executor.layers.linear import ColumnParallelLinear
    from vllm.utils.network_utils import get_open_port

    vllm_config = VllmConfig()
    with set_current_vllm_config(vllm_config):
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{get_open_port()}",
            local_rank=0,
        )
        ensure_model_parallel_initialized(1, 1)
        try:
            torch.manual_seed(0)
            # ColumnParallelLinearWithLoRA routes apply() through the base
            # class (ReplicatedLinearWithLoRA does not: it calls base_layer()
            # directly and never sees the override; see the design doc).
            base = ColumnParallelLinear(
                HIDDEN,
                HIDDEN,
                bias=False,
                params_dtype=DTYPE,
                prefix="model.layers.0.mlp.x",
            ).to(DEVICE)
            shadow = ColumnParallelLinear(
                HIDDEN,
                HIDDEN,
                bias=False,
                params_dtype=DTYPE,
                prefix="model.layers.0.mlp.x",
            ).to(DEVICE)
            base.weight.data.normal_(std=HIDDEN**-0.5)
            shadow.weight.data.normal_(std=HIDDEN**-0.5)
            wrapper = ColumnParallelLinearWithLoRA(base)
            wrapper.output_slices = (HIDDEN,)
            binding = install_binding(
                wrapper,
                "model.layers.0.mlp.x",
                shadow,
                0,
                vllm_config.compilation_config.static_forward_context,
            )
            assert wrapper.base_layer is base
            x = torch.randn(2, HIDDEN, device=DEVICE, dtype=DTYPE)
            # Stub the LoRA application so the test does not need a punica
            # wrapper; the contract is base output from the override first.
            wrapper._apply_lora_to_output = lambda x, output: output + 1
            with set_forward_context(None, vllm_config):
                binding.active = binding.bf16
                torch.testing.assert_close(
                    wrapper.apply(x), F.linear(x, base.weight) + 1
                )
                binding.active = binding.int4_or_fallback
                torch.testing.assert_close(
                    wrapper.apply(x), F.linear(x, shadow.weight) + 1
                )
        finally:
            cleanup_dist_env_and_memory()

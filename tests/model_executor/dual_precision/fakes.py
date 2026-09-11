# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-friendly stand-ins for vLLM linears and the C1 LoRA-wrapper contract.

``FakeLinear`` is a real ``LinearBase`` subclass (so name matching and
``isinstance`` checks see it) that skips the distributed-group lookups of the
production constructor. ``FakeInt4Linear`` adds a ``qweight`` attribute so the
policy code treats it as GPTQ-packed. ``StubLoRAWrapper`` mimics the
``BaseLinearLayerWithLoRA`` contract that C1 adds: a ``base_layer``, a
``base_forward_override`` attribute and ``set_base_forward_override``; LoRA is
applied synchronously after the base output.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.model_executor.layers.linear import LinearBase


class FakeLinearMethod:
    def apply(
        self, layer: nn.Module, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        return F.linear(x, layer.weight, bias)


class FakeLinear(LinearBase):
    """``LinearBase`` without distributed init; ``F.linear`` on ``weight``."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        prefix: str,
        weight: torch.Tensor | None = None,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str = "cpu",
    ) -> None:
        nn.Module.__init__(self)
        self.input_size = input_size
        self.output_size = output_size
        self.input_size_per_partition = input_size
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]
        self.prefix = prefix
        self.tp_rank = 0
        self.tp_size = 1
        self.skip_bias_add = False
        self.return_bias = True
        self.quant_method = FakeLinearMethod()
        if weight is None:
            weight = torch.randn(output_size, input_size, dtype=dtype, device=device)
        self.weight = nn.Parameter(
            weight.to(dtype=dtype, device=device), requires_grad=False
        )
        self.bias = None

    def forward(self, x: torch.Tensor):
        return self.quant_method.apply(self, x, None), None


class FakeInt4Linear(FakeLinear):
    """A ``FakeLinear`` that looks GPTQ-packed to the policy code."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The policy code only checks for the attribute; keep the plain weight
        # as the "dequantized" operand so outputs stay comparable.
        self.qweight = nn.Parameter(
            torch.zeros(1, dtype=torch.int32), requires_grad=False
        )


class StubLoRAWrapper(nn.Module):
    """Minimal model of the C1 contract for ``BaseLinearLayerWithLoRA``."""

    def __init__(self, base_layer: LinearBase, rank: int = 4) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.base_forward_override: (
            Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor] | None
        ) = None
        self.output_slices = (base_layer.output_size_per_partition,)
        weight = base_layer.weight
        self.lora_a = nn.Parameter(
            torch.zeros(
                rank, base_layer.input_size, dtype=weight.dtype, device=weight.device
            ),
            requires_grad=False,
        )
        self.lora_b = nn.Parameter(
            torch.zeros(
                base_layer.output_size_per_partition,
                rank,
                dtype=weight.dtype,
                device=weight.device,
            ),
            requires_grad=False,
        )

    def set_base_forward_override(self, fn) -> None:
        self.base_forward_override = fn

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        override = self.base_forward_override
        if override is not None:
            output = override(x, bias)
        else:
            output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
        return output + F.linear(F.linear(x, self.lora_a), self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.apply(x, None)


def make_named_linears(
    names: list[str],
    hidden: int,
    cls: type[FakeLinear] = FakeLinear,
    seed: int = 0,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, FakeLinear]:
    generator = torch.Generator().manual_seed(seed)
    return {
        name: cls(
            hidden,
            hidden,
            prefix=name,
            weight=torch.randn(hidden, hidden, generator=generator),
            dtype=dtype,
            device=device,
        )
        for name in names
    }


class FakeModel(nn.Module):
    """Registers modules under dotted names (``layers.0.mlp.down_proj``)."""

    def __init__(self, named_modules: dict[str, nn.Module]) -> None:
        super().__init__()
        for name, module in named_modules.items():
            parent: nn.Module = self
            parts = name.split(".")
            for part in parts[:-1]:
                if not hasattr(parent, part) or not isinstance(
                    getattr(parent, part), nn.Module
                ):
                    parent.add_module(part, nn.Module())
                parent = getattr(parent, part)
            parent.add_module(parts[-1], module)


def build_wrapped_model(
    names: list[str], hidden: int, seed: int = 0, device="cpu", dtype=torch.float32
) -> FakeModel:
    """A BF16-style model whose linears are all LoRA-wrapped."""
    linears = make_named_linears(names, hidden, seed=seed, device=device, dtype=dtype)
    return FakeModel({name: StubLoRAWrapper(layer) for name, layer in linears.items()})


def build_int4_model(
    names: list[str],
    quantized: set[str],
    hidden: int,
    seed: int = 1,
    device="cpu",
    dtype=torch.float32,
) -> FakeModel:
    """An INT4-style model: ``quantized`` names are ``FakeInt4Linear``."""
    modules: dict[str, nn.Module] = {}
    generator = torch.Generator().manual_seed(seed)
    for name in names:
        cls = FakeInt4Linear if name in quantized else FakeLinear
        modules[name] = cls(
            hidden,
            hidden,
            prefix=name,
            weight=torch.randn(hidden, hidden, generator=generator),
            dtype=dtype,
            device=device,
        )
    return FakeModel(modules)


def module_dict_snapshot(module: nn.Module) -> dict[str, tuple]:
    """Identity snapshot of ``_modules``/``_parameters``/``_buffers`` of every
    submodule; two snapshots compare equal only if topology is untouched."""
    snapshot: dict[str, tuple] = {}
    for name, sub in module.named_modules():
        snapshot[name] = (
            tuple((k, id(v)) for k, v in sub._modules.items()),
            tuple((k, id(v)) for k, v in sub._parameters.items()),
            tuple((k, id(v)) for k, v in sub._buffers.items()),
        )
    return snapshot

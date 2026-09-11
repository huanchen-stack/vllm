# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch two-GEMM LoRA op for the single-adapter rollout fast path.

``rollout_lora_matmul(x, lora_a, lora_b, output_size, scale)`` computes
``(x @ lora_a.T) * scale @ lora_b.T`` and replaces the Punica shrink/expand
kernels when exactly one adapter is active for every token. It is registered
as a custom op so that it stays opaque to vLLM's piecewise torch.compile
graph and can be captured inside CUDA graphs; the inner matmuls are compiled
once with ``dynamic=True`` so that varying token counts do not recompile.

Numerics: the rank-sized intermediate is kept in the LoRA weight dtype
(bf16/fp16), whereas Punica accumulates its shrink output in fp32. The two
paths therefore agree to tolerance, not bitwise. The result is returned in
``lora_a.dtype``; callers cast to the activation dtype.
"""

import torch

from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def _compiled_rollout_lora_matmul(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    x = x.to(dtype=lora_a.dtype)
    hidden = torch.matmul(x, lora_a.t())
    if scale != 1.0:
        hidden = hidden * scale
    return torch.matmul(hidden, lora_b.t())


def _rollout_lora_matmul(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    output_size: int,
    scale: float,
) -> torch.Tensor:
    # output_size is only needed by the fake implementation.
    del output_size
    return _compiled_rollout_lora_matmul(x, lora_a, lora_b, scale)


def _rollout_lora_matmul_fake(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    output_size: int,
    scale: float,
) -> torch.Tensor:
    del lora_b, scale
    # Must match the real implementation, which returns lora_a.dtype
    # (x is cast to the LoRA dtype before the first GEMM).
    return torch.empty((x.size(0), output_size), device=x.device, dtype=lora_a.dtype)


direct_register_custom_op(
    op_name="rollout_lora_matmul",
    op_func=_rollout_lora_matmul,
    fake_impl=_rollout_lora_matmul_fake,
)
rollout_lora_matmul = torch.ops.vllm.rollout_lora_matmul


__all__ = ["rollout_lora_matmul"]

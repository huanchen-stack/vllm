# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

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
    del output_size
    return _compiled_rollout_lora_matmul(x, lora_a, lora_b, scale)


def _rollout_lora_matmul_fake(
    x: torch.Tensor,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    output_size: int,
    scale: float,
) -> torch.Tensor:
    del lora_a, lora_b, scale
    return torch.empty((x.size(0), output_size), device=x.device, dtype=x.dtype)


try:
    direct_register_custom_op(
        op_name="rollout_lora_matmul",
        op_func=_rollout_lora_matmul,
        fake_impl=_rollout_lora_matmul_fake,
    )
    rollout_lora_matmul = torch.ops.vllm.rollout_lora_matmul
except AttributeError:
    rollout_lora_matmul = _rollout_lora_matmul


__all__ = ["rollout_lora_matmul"]

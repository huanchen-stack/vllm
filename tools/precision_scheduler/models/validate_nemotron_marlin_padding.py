# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical oracle for VLLM_MARLIN_INPUT_PADDING on one real Nemotron-H layer.

Loads ``backbone.layers.<L>.mixer.down_proj`` (K=15680, N=4480, group 64) from a
compressed-tensors W4A16 Nemotron-Nano-9B-v2 snapshot, runs it through the
Triton W4A16 kernel at the original K and through Marlin at K padded to 15744
(one all-zero group with unit scale, activation zero-padded by 64), and reports
cosine similarity, max abs error and relative RMSE.  Exit code 1 if the cosine
is below ``--acceptance-cosine``.

Example::

    python tools/precision_scheduler/models/validate_nemotron_marlin_padding.py \
        --snapshot /path/to/Nemotron-Nano-9B-v2-quantized.w4a16/snapshots/<rev> \
        --output validation.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

MARLIN_TILE = 128
DEFAULT_LAYER = "backbone.layers.1.mixer.down_proj"
DEFAULT_ACCEPTANCE_COSINE = 0.99999


class _Layer(torch.nn.Module):
    pass


def checkpoint_tensor(snapshot: Path, key: str) -> torch.Tensor:
    from safetensors import safe_open

    for shard in sorted(snapshot.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            if key in handle.keys():  # noqa: SIM118 - safe_open handle is not a dict
                return handle.get_tensor(key)
    raise KeyError(f"{key} not found under {snapshot}")


def build_layer(
    packed: torch.Tensor, scales: torch.Tensor, shape: torch.Tensor, device: str
) -> _Layer:
    from vllm.model_executor.parameter import (
        BasevLLMParameter,
        GroupQuantScaleParameter,
        PackedvLLMParameter,
    )

    def noop(*args, **kwargs):
        return None

    layer = _Layer()
    layer.register_parameter(
        "weight_packed",
        PackedvLLMParameter(
            data=packed.to(device),
            weight_loader=noop,
            input_dim=1,
            output_dim=0,
            packed_factor=8,
            packed_dim=1,
        ),
    )
    layer.register_parameter(
        "weight_scale",
        GroupQuantScaleParameter(
            data=scales.to(device), weight_loader=noop, input_dim=1, output_dim=0
        ),
    )
    layer.register_parameter(
        "weight_shape", BasevLLMParameter(data=shape.to(device), weight_loader=noop)
    )
    return layer


def build_kernel(kernel_cls, k: int, n: int, group_size: int):
    from vllm.model_executor.kernels.linear import MPLinearLayerConfig
    from vllm.scalar_type import scalar_types

    config = MPLinearLayerConfig(
        full_weight_shape=(k, n),
        partition_weight_shape=(k, n),
        weight_type=scalar_types.uint4b8,
        act_type=torch.bfloat16,
        group_size=group_size,
        zero_points=False,
        has_g_idx=False,
    )
    return kernel_cls(
        config,
        w_q_param_name="weight_packed",
        w_s_param_name="weight_scale",
        w_zp_param_name=None,
        w_gidx_param_name=None,
    )


def compare_padded_marlin_vs_triton(
    snapshot: Path,
    layer_name: str = DEFAULT_LAYER,
    samples: int = 8,
    seed: int = 7,
    device: str = "cuda",
) -> dict:
    """Run the comparison; assumes the vLLM distributed env is initialized."""
    from vllm.model_executor.kernels.linear import MarlinLinearKernel
    from vllm.model_executor.kernels.linear.mixed_precision.triton_w4a16 import (
        TritonW4A16LinearKernel,
    )

    packed = checkpoint_tensor(snapshot, f"{layer_name}.weight_packed")
    scales = checkpoint_tensor(snapshot, f"{layer_name}.weight_scale")
    shape = checkpoint_tensor(snapshot, f"{layer_name}.weight_shape")
    n, k = int(shape[0]), int(shape[1])
    group_size = k // scales.shape[1]
    padding = (-k) % MARLIN_TILE
    if padding == 0 or padding % group_size != 0:
        raise ValueError(
            f"{layer_name}: K={k} needs padding {padding}, group {group_size}; "
            "not a padding case"
        )
    padded_k = k + padding

    original = build_layer(packed.clone(), scales.clone(), shape.clone(), device)
    padded = build_layer(
        F.pad(packed, (0, padding // 8)),
        F.pad(scales, (0, padding // group_size), value=1.0),
        torch.tensor([n, padded_k], dtype=shape.dtype),
        device,
    )
    triton = build_kernel(TritonW4A16LinearKernel, k, n, group_size)
    marlin = build_kernel(MarlinLinearKernel, padded_k, n, group_size)
    triton.process_weights_after_loading(original)
    marlin.process_weights_after_loading(padded)

    torch.manual_seed(seed)
    x = torch.randn(samples, k, device=device, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = triton.apply_weights(original, x, None)
        actual = marlin.apply_weights(padded, F.pad(x, (0, padding)), None)
    e, a = expected.float().flatten(), actual.float().flatten()
    return {
        "layer": layer_name,
        "samples": samples,
        "original_k": k,
        "padded_k": padded_k,
        "n": n,
        "group_size": group_size,
        "cosine": F.cosine_similarity(e, a, dim=0).item(),
        "max_abs": (e - a).abs().max().item(),
        "relative_rmse": (
            (e - a).square().mean().sqrt() / e.square().mean().sqrt()
        ).item(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--acceptance-cosine", type=float, default=DEFAULT_ACCEPTANCE_COSINE
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
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
        )
        ensure_model_parallel_initialized(1, 1)
    try:
        result = compare_padded_marlin_vs_triton(
            args.snapshot, args.layer, args.samples, args.seed
        )
    finally:
        destroy_model_parallel()
        destroy_distributed_environment()
    result["acceptance_cosine"] = args.acceptance_cosine
    result["accepted"] = result["cosine"] >= args.acceptance_cosine
    text = json.dumps(result, indent=2)
    print(text)
    if args.output is not None:
        args.output.write_text(text + "\n")
    return 0 if result["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())

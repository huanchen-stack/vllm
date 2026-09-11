# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from functools import partial

import torch
import torch.nn.functional as F
from compressed_tensors.quantization import ActivationOrdering

from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import (
    MarlinLinearKernel,
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsScheme,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    get_marlin_input_dtype,
    marlin_repeat_scales_on_all_ranks,
)
from vllm.model_executor.parameter import (
    BasevLLMParameter,
    ChannelQuantScaleParameter,
    GroupQuantScaleParameter,
    PackedColumnParameter,
    PackedvLLMParameter,
    RowvLLMParameter,
)
from vllm.scalar_type import scalar_types

logger = init_logger(__name__)

__all__ = ["CompressedTensorsWNA16"]
WNA16_SUPPORTED_TYPES_MAP = {4: scalar_types.uint4b8, 8: scalar_types.uint8b128}
WNA16_ZP_SUPPORTED_TYPES_MAP = {4: scalar_types.uint4, 8: scalar_types.uint8}
WNA16_SUPPORTED_BITS = list(WNA16_SUPPORTED_TYPES_MAP.keys())
# Marlin requires the input dimension K to be a multiple of its tile size.
MARLIN_INPUT_TILE = 128


class CompressedTensorsWNA16(CompressedTensorsScheme):
    _kernel_backends_being_used: set[str] = set()

    def __init__(
        self,
        strategy: str,
        num_bits: int,
        group_size: int | None = None,
        symmetric: bool | None = True,
        actorder: ActivationOrdering | None = None,
        layer_name: str | None = None,
    ):
        self.pack_factor = 32 // num_bits
        self.strategy = strategy
        self.symmetric = symmetric
        self.group_size = -1 if group_size is None else group_size
        self.has_g_idx = actorder == ActivationOrdering.GROUP
        self.layer_name = layer_name
        # Number of zero input columns appended so K is a Marlin tile multiple
        # (VLLM_MARLIN_INPUT_PADDING); 0 means the layer is untouched.
        self.input_padding = 0

        if self.group_size == -1 and self.strategy != "channel":
            raise ValueError(
                "Marlin kernels require group quantization or "
                "channelwise quantization, but found no group "
                "size and strategy is not channelwise."
            )

        if num_bits not in WNA16_SUPPORTED_TYPES_MAP:
            raise ValueError(
                f"Unsupported num_bits = {num_bits}. "
                f"Supported num_bits = {WNA16_SUPPORTED_TYPES_MAP.keys()}"
            )

        self.quant_type = (
            WNA16_ZP_SUPPORTED_TYPES_MAP[num_bits]
            if not self.symmetric
            else WNA16_SUPPORTED_TYPES_MAP[num_bits]
        )

    @classmethod
    def get_min_capability(cls) -> int:
        # Turing and up
        return 75

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)

        original_input_size = input_size_per_partition
        input_padding = self._marlin_input_padding(input_size, input_size_per_partition)
        if input_padding:
            input_size += input_padding
            input_size_per_partition += input_padding
            logger.info_once(
                "VLLM_MARLIN_INPUT_PADDING: padding WNA16 layer %s input K "
                "from %d to %d (one all-zero group per %d columns).",
                self.layer_name,
                original_input_size,
                input_size_per_partition,
                self.group_size,
            )

        mp_linear_kernel_config = MPLinearLayerConfig(
            full_weight_shape=(input_size, output_size),
            partition_weight_shape=(
                input_size_per_partition,
                output_size_per_partition,
            ),
            weight_type=self.quant_type,
            act_type=params_dtype,
            group_size=self.group_size,
            zero_points=not self.symmetric,
            has_g_idx=self.has_g_idx,
        )

        kernel_type = choose_mp_linear_kernel(mp_linear_kernel_config)

        if kernel_type.__name__ not in self._kernel_backends_being_used:
            logger.info("Using %s for CompressedTensorsWNA16", kernel_type.__name__)
            self._kernel_backends_being_used.add(kernel_type.__name__)

        if kernel_type is MarlinLinearKernel:
            input_dtype = get_marlin_input_dtype(self.layer_name)
            if input_dtype is not None:
                mp_linear_kernel_config.act_type = input_dtype

        # If group_size is -1, we are in channelwise case.
        group_size = self.group_size if self.group_size != -1 else input_size
        row_parallel = input_size != input_size_per_partition
        partition_scales = not marlin_repeat_scales_on_all_ranks(
            self.has_g_idx, self.group_size, row_parallel
        )

        scales_and_zp_size = input_size // group_size

        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        def _padded_loader(kind: str) -> Callable:
            # One wrapper per parameter so padding is keyed on parameter identity,
            # never on tensor shape heuristics.
            if not input_padding:
                return weight_loader
            return partial(
                self._load_with_input_padding,
                weight_loader=weight_loader,
                kind=kind,
                original_input_size=original_input_size,
                padded_input_size=input_size_per_partition,
                pack_factor=self.pack_factor,
                group_size=group_size,
            )

        weight = PackedvLLMParameter(
            input_dim=1,
            output_dim=0,
            weight_loader=_padded_loader("weight_packed"),
            packed_factor=self.pack_factor,
            packed_dim=1,
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
        )

        weight_scale_args = {
            "weight_loader": _padded_loader("weight_scale"),
            "data": torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
        }

        zeros_args = {
            "weight_loader": weight_loader,
            "data": torch.zeros(
                output_size_per_partition // self.pack_factor,
                scales_and_zp_size,
                dtype=torch.int32,
            ),
        }

        if not partition_scales:
            weight_scale = ChannelQuantScaleParameter(output_dim=0, **weight_scale_args)

            if not self.symmetric:
                qzeros = PackedColumnParameter(
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )
        else:
            weight_scale = GroupQuantScaleParameter(
                output_dim=0, input_dim=1, **weight_scale_args
            )
            if not self.symmetric:
                qzeros = PackedvLLMParameter(
                    input_dim=1,
                    output_dim=0,
                    packed_dim=0,
                    packed_factor=self.pack_factor,
                    **zeros_args,
                )

        # A 2D array defining the original shape of the weights
        # before packing
        weight_shape = BasevLLMParameter(
            data=torch.empty(2, dtype=torch.int64),
            weight_loader=_padded_loader("weight_shape"),
        )

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

        if not self.symmetric:
            layer.register_parameter("weight_zero_point", qzeros)

        # group index (for activation reordering)
        if self.has_g_idx:
            weight_g_idx = RowvLLMParameter(
                data=torch.empty(
                    input_size_per_partition,
                    dtype=torch.int32,
                ),
                input_dim=0,
                weight_loader=weight_loader,
            )
            layer.register_parameter("weight_g_idx", weight_g_idx)

        self.kernel = kernel_type(
            mp_linear_kernel_config,
            w_q_param_name="weight_packed",
            w_s_param_name="weight_scale",
            w_zp_param_name="weight_zero_point",
            w_gidx_param_name="weight_g_idx",
        )
        self.input_padding = input_padding

    def _marlin_input_padding(
        self, input_size: int, input_size_per_partition: int
    ) -> int:
        """Zero input columns to append so K becomes a Marlin tile multiple.

        Only symmetric, groupwise, non-actorder layers whose full K equals the
        partition K are eligible: with tensor parallelism a row-parallel layer
        shards K, and padding only the last shard would misalign the scales
        across ranks, so such layers fall through to the regular kernel choice
        (Triton W4A16 when K % 128 != 0).  Column-parallel layers keep the full
        K on every rank and are padded normally.
        """
        if not envs.VLLM_MARLIN_INPUT_PADDING:
            return 0
        if not self.symmetric or self.has_g_idx or self.group_size <= 0:
            return 0
        if input_size != input_size_per_partition:
            return 0
        remainder = input_size_per_partition % MARLIN_INPUT_TILE
        if remainder == 0:
            return 0
        padding = MARLIN_INPUT_TILE - remainder
        if padding % self.group_size != 0:
            raise ValueError(
                "VLLM_MARLIN_INPUT_PADDING must append complete quantization "
                f"groups, got padding={padding} for K={input_size_per_partition} "
                f"with group_size={self.group_size} (layer {self.layer_name})."
            )
        return padding

    @staticmethod
    def _load_with_input_padding(
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        *args,
        weight_loader: Callable,
        kind: str,
        original_input_size: int,
        padded_input_size: int,
        pack_factor: int,
        group_size: int,
        **kwargs,
    ) -> None:
        """Pad one checkpoint tensor along its (compressed) input dimension.

        ``kind`` names the parameter this loader was attached to: the packed
        int32 weight gets ``padding // pack_factor`` zero columns (packed zeros
        decode to zero), the group scales get ``padding // group_size`` columns
        of 1.0 (a unit scale for the all-zero group avoids NaN/inf), and the
        two-element ``weight_shape`` gets its K entry rewritten.
        """
        padding = padded_input_size - original_input_size
        if kind == "weight_shape":
            loaded_weight = loaded_weight.clone()
            if int(loaded_weight[1]) == padded_input_size:
                pass  # already padded (e.g. re-loaded weights)
            elif int(loaded_weight[1]) == original_input_size:
                loaded_weight[1] = padded_input_size
            else:
                raise ValueError(
                    "VLLM_MARLIN_INPUT_PADDING: weight_shape K "
                    f"{int(loaded_weight[1])} does not match the layer's "
                    f"input size {original_input_size}."
                )
        else:
            if kind == "weight_packed":
                pad, fill = padding // pack_factor, 0
            elif kind == "weight_scale":
                pad, fill = padding // group_size, 1.0
            else:
                raise ValueError(f"unexpected padded parameter kind {kind!r}")
            if loaded_weight.shape[-1] + pad == param.shape[-1]:
                loaded_weight = F.pad(loaded_weight, (0, pad), value=fill)
            elif loaded_weight.shape[-1] != param.shape[-1]:
                raise ValueError(
                    f"VLLM_MARLIN_INPUT_PADDING: cannot pad {kind} from "
                    f"{tuple(loaded_weight.shape)} to {tuple(param.shape)}."
                )
        weight_loader(param, loaded_weight, *args, **kwargs)

    # Checkpoints are serialized in compressed-tensors format, which is
    # different from the format the kernel may want. Handle repacking here.
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        self.kernel.process_weights_after_loading(layer)

    def apply_weights(
        self, layer: torch.nn.Module, x: torch.Tensor, bias: torch.Tensor | None
    ) -> torch.Tensor:
        if self.input_padding:
            x = F.pad(x, (0, self.input_padding))
        return self.kernel.apply_weights(layer, x, bias)

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import torch
from transformers import PretrainedConfig

from vllm import envs
from vllm.config import get_current_vllm_config
from vllm.config.lora import LoRAConfig
from vllm.distributed.utils import divide
from vllm.forward_context import (
    ForwardContext,
    get_forward_context,
    is_forward_context_available,
)
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    LinearBase,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.platforms import current_platform
from vllm.utils.multi_stream_utils import (
    execute_in_parallel,
    maybe_execute_in_parallel,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .base import BaseLayerWithLoRA
from .utils import _get_lora_aux_cuda_stream, _get_lora_device

if envs.VLLM_LORA_ENABLE_DUAL_STREAM:

    def lora_linear_async(
        layer_name: str,
        output_size: int,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        forward_context: ForwardContext = get_forward_context()
        self = forward_context.no_compile_layers[layer_name]
        return self._apply_async_impl(x, bias)

    def lora_linear_async_fake(
        layer_name: str,
        output_size: int,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # The real function reshapes output back to the original 3D shape
        # when the input has an extra batch dimension (transformers backend).
        if x.ndim == 3:
            return torch.empty(
                (x.size(0), x.size(1), output_size),
                device=x.device,
                dtype=x.dtype,
            )
        return torch.empty(
            (x.size(0), output_size),
            device=x.device,
            dtype=x.dtype,
        )

    direct_register_custom_op(
        op_name="lora_linear_async",
        op_func=lora_linear_async,
        fake_impl=lora_linear_async_fake,
    )


class BaseLinearLayerWithLoRA(BaseLayerWithLoRA):
    def __init__(self, base_layer: LinearBase):
        super().__init__()

        self._enable_aux_cuda_stream = envs.VLLM_LORA_ENABLE_DUAL_STREAM
        self.base_layer = base_layer
        # Single-adapter rollout fast path (ROLLOUT_QLORA). Read once at
        # construction so that the forward path never touches os.environ.
        self._rollout_lora_enabled = envs.ROLLOUT_QLORA
        # Packed per-slot LoRA A/B buffers for the rollout fast path, see
        # _create_rollout_lora_weights. None unless ROLLOUT_QLORA is set.
        self.rollout_lora_a_stacked: torch.Tensor | None = None
        self.rollout_lora_b_stacked: torch.Tensor | None = None
        # Optional replacement for the base GEMM in apply(); see
        # set_base_forward_override. Used by the dual-precision component to
        # select between base weight sets outside the compiled graph.
        self.base_forward_override: (
            Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor] | None
        ) = None
        self.input_size = self.base_layer.input_size
        # Ensure tp_size and tp_rank consistency with the base_layer.
        self.tp_size = self.base_layer.tp_size
        self.tp_rank = self.base_layer.tp_rank
        self.device = _get_lora_device(self.base_layer)
        self._init_lora_stream_context()
        self.output_slices: tuple[int, ...]
        self.output_size: int
        self.n_slices: int

    def _init_lora_stream_context(self) -> None:
        if not self._enable_aux_cuda_stream:
            return
        vllm_config = get_current_vllm_config()
        self._lora_stream = _get_lora_aux_cuda_stream()
        assert current_platform.is_cuda_alike()
        self._events = [torch.cuda.Event(), torch.cuda.Event()]
        # lora_linear avoids prefix conflicts with the base layer
        self.layer_name = self.base_layer.prefix + ".lora_linear_async"
        compilation_config = vllm_config.compilation_config
        if self.layer_name in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer name: {}".format(self.layer_name))
        compilation_config.static_forward_context[self.layer_name] = self

    def create_lora_weights(
        self,
        max_loras: int,
        lora_config: LoRAConfig,
        model_config: PretrainedConfig | None = None,
    ) -> None:
        self.lora_config = lora_config
        if isinstance(self.base_layer, ReplicatedLinear):
            lora_a_out_size = lora_config.max_lora_rank
            lora_b_out_size = self.output_size

        elif isinstance(self.base_layer, ColumnParallelLinear):
            lora_a_out_size = (
                lora_config.max_lora_rank
                if not lora_config.fully_sharded_loras
                else divide(lora_config.max_lora_rank, self.tp_size)
            )
            lora_b_out_size = self.output_size

        elif isinstance(self.base_layer, RowParallelLinear):
            lora_a_out_size = lora_config.max_lora_rank
            lora_b_out_size = (
                self.output_size
                if not lora_config.fully_sharded_loras
                else divide(self.output_size, self.tp_size)
            )
        else:
            raise NotImplementedError

        self.lora_a_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_a_out_size,
                self.input_size,
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for _ in range(self.n_slices)
        )
        self.lora_b_stacked = tuple(
            torch.zeros(
                max_loras,
                1,
                lora_b_out_size,
                lora_config.max_lora_rank,
                dtype=lora_config.lora_dtype,
                device=self.device,
            )
            for _ in range(self.n_slices)
        )
        self.output_slices = (self.lora_b_stacked[0].shape[2],)
        self._create_rollout_lora_weights(max_loras)

    def set_base_forward_override(
        self,
        fn: Callable[[torch.Tensor, torch.Tensor | None], torch.Tensor] | None,
    ) -> None:
        """Install (or clear with None) a replacement for the base GEMM.

        When set, apply() computes the base output as ``fn(x, bias)`` and then
        applies LoRA synchronously on the same stream; the dual-stream op is
        not used. The override must return a tensor of the same shape and
        dtype as ``self.base_layer.quant_method.apply(self.base_layer, x,
        bias)`` would.
        """
        self.base_forward_override = fn

    def _create_rollout_lora_weights(self, max_loras: int) -> None:
        """Allocate the packed LoRA buffers used by the rollout fast path.

        Layout (per adapter slot ``i``):
          rollout_lora_a_stacked[i, 0] = cat(lora_a_stacked[s][i, 0] for s)
              shape (n_slices * R, input_size)
          rollout_lora_b_stacked[i, 0] = block_diag(lora_b_stacked[s][i, 0])
              shape (sum(output_slices), n_slices * R)
        where R is the per-slice rank *capacity* of the stacked buffers
        (lora_config.max_lora_rank), not the adapter's actual rank: slice
        ``s`` occupies rank rows ``[s * R, (s + 1) * R)`` and output rows
        ``[offset_s, offset_s + output_slices[s])``; unused rank rows and
        columns stay zero and contribute nothing. With this layout a merged
        layer (QKV, gate-up, Qwen3.5 in_proj) becomes one ``x @ A^T @ B^T``
        GEMM pair.
        """
        if (
            not self._rollout_lora_enabled
            or not current_platform.is_cuda_alike()
            or self.lora_config.fully_sharded_loras
        ):
            # The fast path lives in PunicaWrapperGPU and is gated off for
            # fully sharded LoRA there as well; do not allocate what it
            # would never read.
            self.rollout_lora_a_stacked = None
            self.rollout_lora_b_stacked = None
            return

        total_rank = sum(weight.shape[2] for weight in self.lora_a_stacked)
        total_output = sum(self.output_slices)
        self.rollout_lora_a_stacked = torch.zeros(
            max_loras,
            1,
            total_rank,
            self.input_size,
            dtype=self.lora_config.lora_dtype,
            device=self.device,
        )
        self.rollout_lora_b_stacked = torch.zeros(
            max_loras,
            1,
            total_output,
            total_rank,
            dtype=self.lora_config.lora_dtype,
            device=self.device,
        )

    def _refresh_rollout_lora_weights(self, index: int) -> None:
        """Rebuild the packed buffers of slot ``index`` from lora_*_stacked."""
        if self.rollout_lora_a_stacked is None or self.rollout_lora_b_stacked is None:
            return

        self.rollout_lora_a_stacked[index].zero_()
        self.rollout_lora_b_stacked[index].zero_()

        rank_offset = 0
        output_offset = 0
        for slice_idx, output_size in enumerate(self.output_slices):
            lora_a = self.lora_a_stacked[slice_idx][index, 0]
            lora_b = self.lora_b_stacked[slice_idx][index, 0]
            # Rank capacity of this slice's stacked buffer (max_lora_rank).
            rank = lora_a.shape[0]
            assert lora_b.shape == (output_size, rank)

            self.rollout_lora_a_stacked[
                index, 0, rank_offset : rank_offset + rank, :
            ].copy_(lora_a, non_blocking=True)
            self.rollout_lora_b_stacked[
                index,
                0,
                output_offset : output_offset + output_size,
                rank_offset : rank_offset + rank,
            ].copy_(lora_b, non_blocking=True)

            rank_offset += rank
            output_offset += output_size

    def _rollout_lora_kwargs(self) -> dict[str, torch.Tensor]:
        """Extra add_lora_linear kwargs; empty unless the fast path is on."""
        if self.rollout_lora_a_stacked is None or self.rollout_lora_b_stacked is None:
            return {}
        return {
            "rollout_lora_a_stacked": self.rollout_lora_a_stacked,
            "rollout_lora_b_stacked": self.rollout_lora_b_stacked,
        }

    def reset_lora(self, index: int):
        for s_index in range(self.n_slices):
            self.lora_a_stacked[s_index][index] = 0
            self.lora_b_stacked[s_index][index] = 0
        if self.rollout_lora_a_stacked is not None:
            self.rollout_lora_a_stacked[index] = 0
        if self.rollout_lora_b_stacked is not None:
            self.rollout_lora_b_stacked[index] = 0

    def set_lora(
        self,
        index: int,
        lora_a: torch.Tensor | list[torch.Tensor],
        lora_b: torch.Tensor | list[torch.Tensor],
    ):
        # Except for QKVParallelLinearWithLoRA and
        # MergedColumnParallelLinearWithLoRA, all other linear LoRA layers
        # store weights in a tuple of size 1. These two layers will
        # override this function.
        assert isinstance(lora_a, torch.Tensor)
        assert isinstance(lora_b, torch.Tensor)
        assert (
            len(self.lora_a_stacked) == len(self.lora_b_stacked) == self.n_slices == 1
        )

        self.reset_lora(index)
        if self.tp_size > 1:
            lora_a = self.slice_lora_a(lora_a)
            lora_b = self.slice_lora_b(lora_b)

        self.lora_a_stacked[0][index, 0, : lora_a.shape[0], : lora_a.shape[1]].copy_(
            lora_a, non_blocking=True
        )
        self.lora_b_stacked[0][index, 0, : lora_b.shape[0], : lora_b.shape[1]].copy_(
            lora_b, non_blocking=True
        )
        self._refresh_rollout_lora_weights(index)

    def apply(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        if self.base_forward_override is not None:
            # The override owns the base GEMM (e.g. the dual-precision base
            # selector); LoRA is applied synchronously on the same stream and
            # the dual-stream op is deliberately not used (design decision 1).
            return self._apply_sync(x, bias)
        # is_forward_context_available for tower modules
        if self._enable_aux_cuda_stream and is_forward_context_available():
            output_size = sum(self.output_slices)
            return torch.ops.vllm.lora_linear_async(
                self.layer_name, output_size, x, bias
            )
        else:
            return self._apply_sync(x, bias)

    def _base_forward(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Base GEMM: the installed override if any, else quant_method.apply."""
        if self.base_forward_override is not None:
            return self.base_forward_override(x, bias)
        return self.base_layer.quant_method.apply(self.base_layer, x, bias)

    def _apply_sync(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        output = self._base_forward(x, bias)
        return self._apply_lora_to_output(x, output)

    def _apply_base_forward(self, x: torch.Tensor) -> torch.Tensor:
        base_output = self.base_layer(x)
        output = base_output[0] if isinstance(base_output, tuple) else base_output
        return self._apply_lora_to_output(x, output)

    def _apply_lora_to_output(
        self, x: torch.Tensor, output: torch.Tensor
    ) -> torch.Tensor:
        original_shape = output.shape if output.ndim == 3 else None

        # In transformers backend, x and output have extra batch dimension like
        # (1, seq_len, hidden_dim), while punica expects (seq_len, hidden_dim),
        # therefore we need to flatten the batch dimensions.
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        lora_output: torch.Tensor | None = self.punica_wrapper.add_lora_linear(
            output,
            x,
            self.lora_a_stacked,
            self.lora_b_stacked,
            1.0,
            self.output_slices,
            **self._rollout_lora_kwargs(),
        )
        if not current_platform.can_update_inplace():
            output = lora_output

        # Reshape the flattened output back to its original shape,
        # as some MM encoders cannot handle flattened inputs.
        if original_shape is not None:
            output = output.reshape(original_shape)

        return output

    def _execute_lora_async(
        self,
        base_fn: Callable[[], torch.Tensor],
        lora_fn: Callable[[], torch.Tensor],
        lora_first: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run base_fn on the current stream and lora_fn on the LoRA stream.

        ``lora_first=False`` is the vanilla order (base GEMM launched first,
        then LoRA on the aux stream). ``lora_first=True`` launches the LoRA
        GEMMs on the aux stream before the base GEMM, so the small LoRA
        kernels are already queued when the large base GEMM starts; this is
        the order used by the rollout fast path.
        """
        if lora_first:
            output, aux_results = execute_in_parallel(
                base_fn,
                [lora_fn],
                self._events[0],
                [self._events[1]],
                [self._lora_stream],
                enable=True,
            )
            return output, aux_results[0]
        return maybe_execute_in_parallel(
            base_fn,
            lora_fn,
            self._events[0],
            self._events[1],
            self._lora_stream,
        )

    def _apply_async_impl(
        self, x: torch.Tensor, bias: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Forward pass with base linear and LoRA on separate CUDA streams
        for overlap, using maybe_execute_in_parallel.
        Base layer runs on default stream; LoRA runs on aux stream.
        """
        assert envs.VLLM_LORA_ENABLE_DUAL_STREAM
        assert x.ndim in (2, 3)
        num_tokens = x.size(0) if x.ndim == 2 else x.size(1)
        output_size = sum(self.output_slices)

        def base_fn() -> torch.Tensor:
            return self.base_layer.quant_method.apply(self.base_layer, x, bias)

        def lora_fn() -> torch.Tensor:
            # Must be zeros, not empty: _lora_expand_kernel exits early (without
            # writing) when lora_id == -1 (no active LoRA). If uninitialized,
            # output.add_(lora_result) below would corrupt the base output.
            lora_output = torch.zeros(
                (num_tokens, output_size),
                device=self.device,
                dtype=x.dtype,
            )

            # Flatten the batch dimension for the transformers backend
            # (which uses shape (1, seq_len, hidden)), matching _apply_sync.
            x_2d = x.flatten(0, 1) if x.ndim == 3 else x
            self.punica_wrapper.add_lora_linear(
                lora_output,
                x_2d,
                self.lora_a_stacked,
                self.lora_b_stacked,
                1.0,
                self.output_slices,
                add_inputs=False,
                **self._rollout_lora_kwargs(),
            )
            return lora_output

        output, lora_result = self._execute_lora_async(
            base_fn, lora_fn, lora_first=self._rollout_lora_enabled
        )

        original_shape = output.shape if output.ndim == 3 else None

        # In transformers backend, x and output have extra batch dimension like
        # (1, seq_len, hidden_dim), while punica expects (seq_len, hidden_dim),
        # therefore we need to flatten the batch dimensions.
        if x.ndim == 3 and output.ndim == 3:
            output = output.flatten(0, 1)
            x = x.flatten(0, 1)

        output.add_(lora_result)

        # Reshape the flattened output back to its original shape,
        # as some MM encoders cannot handle flattened inputs.
        if original_shape is not None:
            output = output.reshape(original_shape)

        return output

    @property
    def weight(self) -> torch.Tensor:
        # unquantizedLinear
        if hasattr(self.base_layer, "weight"):
            return self.base_layer.weight
        # Compressed Tensor
        elif hasattr(self.base_layer, "weight_packed"):
            return self.base_layer.weight_packed
        # GPTQ/AWQ
        elif hasattr(self.base_layer, "qweight"):
            return self.base_layer.qweight
        # marlin
        elif hasattr(self.base_layer, "B"):
            return self.base_layer.B
        else:
            raise ValueError(f"Unsupported base layer: {self.base_layer}")

    @property
    def bias(self) -> torch.Tensor | None:
        if hasattr(self.base_layer, "bias"):
            return self.base_layer.bias
        else:
            return None

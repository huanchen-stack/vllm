# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlashInfer autotune must exercise the shadow, not only the BF16 base.

A dummy run binds nothing unless told to, so an autotune forward executed
whatever was bound at start-up, the BF16 base, and the quantized shadow's
GEMMs kept FlashInfer's untuned default tactics. On Phi-4-mini with an NVFP4
shadow that was 24 us per Cutlass GEMM at batch 1 against 8-17 us tuned.
"""

from types import SimpleNamespace

from vllm.model_executor.dual_precision import BASE_PRECISION_BF16, BASE_PRECISION_INT4
from vllm.model_executor.warmup.kernel_warmup import (
    _autotune_base_precisions,
    _restore_bf16,
)


def test_vanilla_runner_autotunes_once_and_binds_nothing():
    runner = SimpleNamespace(dual_precision_enabled=False)
    assert _autotune_base_precisions(runner) == [None]


def test_runner_without_the_attribute_is_treated_as_vanilla():
    assert _autotune_base_precisions(SimpleNamespace()) == [None]


def test_dual_precision_runner_autotunes_bf16_then_the_shadow():
    runner = SimpleNamespace(dual_precision_enabled=True)
    assert _autotune_base_precisions(runner) == [
        BASE_PRECISION_BF16,
        BASE_PRECISION_INT4,
    ]


def test_restore_bf16_calls_the_runner_hook_when_present():
    calls: list[str] = []
    runner = SimpleNamespace(_restore_bf16_binding=lambda: calls.append("restored"))
    _restore_bf16(runner)
    assert calls == ["restored"]
    _restore_bf16(SimpleNamespace())  # vanilla runner: no hook, no error

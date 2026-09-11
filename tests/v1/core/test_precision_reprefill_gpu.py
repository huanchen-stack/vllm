# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU smoke: re-prefill fires once per batch on a real engine and every
survivor is preempted at the crossing.

Runs an in-process vLLM engine (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) on
Qwen3.5-4B with ``VLLM_DUAL_PRECISION_POLICY=fixed_frontier:32`` and
``VLLM_DUAL_PRECISION_REPREFILL=1``, generates two batches of eight prompts
and records ``preempted_req_ids`` / ``num_unfinished_requests`` from every
``schedule()`` through a scheduler subclass.

Dual-precision residency (C2/C3) is deliberately not enabled: the base
weights stay BF16 for the whole run. What this tier verifies is the
preempt-and-recompute path itself (one idle trigger step per batch, every
unfinished request preempted exactly once, generation completing with the
original ``max_tokens`` budget, and the recomputed KV reproducing the
uninterrupted greedy continuation), which is independent of which weights
the recompute runs on.

Launch under the decision-13 launcher on one free GPU:

    run_gpu.sh --gpus 2 --timeout 1500 -- python -m pytest -p no:cacheprovider -q \
        tests/v1/core/test_precision_reprefill_gpu.py -m gpu_smoke
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

DEFAULT_MODEL = (
    "/data/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/"
    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)
FRONTIER = 32
NUM_PROMPTS = 8
MAX_TOKENS = 128


@pytest.mark.gpu_smoke
def test_reprefill_fires_once_per_batch_on_a_real_engine(request, tmp_path: Path, monkeypatch):
    markexpr = request.config.getoption("-m", default="") or ""
    if "gpu_smoke" not in markexpr and os.environ.get("PS_RUN_GPU_SMOKE") != "1":
        pytest.skip("gpu_smoke tier: run with -m gpu_smoke under run_gpu.sh")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("run under run_gpu.sh (decision 13): CUDA_VISIBLE_DEVICES is unset")
    model = os.environ.get("PRECISION_SWITCH_SMOKE_MODEL", DEFAULT_MODEL)
    if not Path(model).exists():
        pytest.skip(f"checkpoint not found: {model}")

    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    monkeypatch.setenv("VLLM_DUAL_PRECISION_POLICY", f"fixed_frontier:{FRONTIER}")
    monkeypatch.setenv("VLLM_DUAL_PRECISION_REPREFILL", "1")
    monkeypatch.setenv("VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS", "")
    monkeypatch.setenv("VLLM_DUAL_PRECISION_RELOAD_POLICY_EACH_ROLLOUT", "0")

    from vllm import LLM, SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler

    class RecordingScheduler(Scheduler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # (unfinished, longest response, precision, preempted ids,
            #  scheduled tokens, {preempted id: num_preemptions})
            self.steps: list[tuple] = []

        def schedule(self):
            output = super().schedule()
            longest = max(
                (r.num_cumulative_output_tokens for r in self.requests.values()),
                default=0,
            )
            preempted = set(output.preempted_req_ids)
            self.steps.append(
                (
                    output.num_unfinished_requests,
                    longest,
                    output.dual_precision_base_precision,
                    preempted,
                    output.total_num_scheduled_tokens,
                    {rid: self.requests[rid].num_preemptions for rid in preempted},
                )
            )
            return output

    llm = LLM(
        model=model,
        max_model_len=1024,
        max_num_seqs=NUM_PROMPTS,
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        enable_prefix_caching=False,
        scheduler_cls=RecordingScheduler,
        seed=0,
    )
    try:
        engine_core = llm.llm_engine.engine_core.engine_core
        scheduler = engine_core.scheduler
        assert isinstance(scheduler, RecordingScheduler)
        assert scheduler.precision_switcher is not None
        assert scheduler.precision_reprefill_enabled

        prompts = [
            f"Question {i}: explain why the sky is blue in detail."
            for i in range(NUM_PROMPTS)
        ]
        params = SamplingParams(max_tokens=MAX_TOKENS, ignore_eos=True, temperature=0.0)

        def one_batch(engine):
            start = len(scheduler.steps)
            outputs = engine.generate(prompts, params)
            assert len(outputs) == NUM_PROMPTS
            # The generation cap is not reset by the re-prefill.
            assert all(len(o.outputs[0].token_ids) == MAX_TOKENS for o in outputs)
            steps = scheduler.steps[start:]
            triggers = [s for s in steps if s[3]]
            assert len(triggers) == 1, f"expected one trigger per batch, got {triggers}"
            unfinished, longest, precision, preempted, scheduled, preemptions = (
                triggers[0]
            )
            # Preempted == unfinished at the crossing, on the switching step,
            # which is an idle engine step; every survivor preempted once.
            assert len(preempted) == unfinished == NUM_PROMPTS
            assert precision == "int4"
            assert longest >= FRONTIER
            assert scheduled == 0
            assert all(n == 1 for n in preemptions.values())
            index = steps.index(triggers[0])
            assert all(s[2] == "bf16" for s in steps[:index])
            assert all(s[2] == "int4" for s in steps[index:] if s[0] > 0)
            return [list(o.outputs[0].token_ids) for o in outputs]

        first = one_batch(llm)
        second = one_batch(llm)
        assert len(scheduler.precision_switcher.switches) == 2
        assert scheduler.precision_switcher.rollout_index == 2
        assert first == second

        # Reference: the same batch without re-prefill (the flag can be
        # cleared on the live scheduler; the switcher still switches).
        scheduler.precision_reprefill_enabled = False
        start = len(scheduler.steps)
        reference = llm.generate(prompts, params)
        assert not any(s[3] for s in scheduler.steps[start:])
        reference = [list(o.outputs[0].token_ids) for o in reference]
        # Tokens generated before the switch are bit-identical (same batch,
        # same greedy decode); the recomputed KV reproduces the uninterrupted
        # continuation up to kernel-level numerics.
        for ours, theirs in zip(first, reference):
            assert ours[:FRONTIER] == theirs[:FRONTIER]
        identical = sum(ours == theirs for ours, theirs in zip(first, reference))
        agreement = (
            sum(
                sum(a == b for a, b in zip(ours, theirs)) / MAX_TOKENS
                for ours, theirs in zip(first, reference)
            )
            / NUM_PROMPTS
        )
        print(
            f"\nreprefill vs continuous: identical={identical}/{NUM_PROMPTS}, "
            f"token agreement={agreement:.3f}"
        )
        assert identical >= NUM_PROMPTS // 2
        assert agreement >= 0.9
    finally:
        del llm


if __name__ == "__main__":
    raise SystemExit(
        pytest.main([__file__, "-m", "gpu_smoke", "-p", "no:cacheprovider", "-q", "-s"])
    )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU smoke: the precision signal reaches SchedulerOutput and the cohort file
is written on a real engine.

Runs an in-process vLLM engine (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) on
Qwen3.5-4B with ``VLLM_DUAL_PRECISION_POLICY=fixed_frontier:32`` and eight
prompts of 128 tokens, records ``dual_precision_base_precision`` from every
``schedule()`` through a scheduler subclass and checks the switch-cohort JSONL.

Dual-precision residency (C2) and the precision-keyed dispatch (C3) are not
part of this component: the model does not change weights here. What is
verified is the scheduler-side contract that those components consume.

Launch under the decision-13 launcher on one free GPU:

    run_gpu.sh --gpus 7 --timeout 1500 -- python -m pytest -p no:cacheprovider -q \
        tests/v1/core/test_precision_switch_gpu.py -m gpu_smoke
"""

from __future__ import annotations

import json
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
def test_precision_signal_and_cohort_file_on_a_real_engine(request, tmp_path: Path):
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

    cohorts = tmp_path / "obs" / "online_switch_cohorts.jsonl"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_DUAL_PRECISION_POLICY"] = f"fixed_frontier:{FRONTIER}"
    os.environ["VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS"] = str(cohorts)
    os.environ["VLLM_DUAL_PRECISION_RELOAD_POLICY_EACH_ROLLOUT"] = "0"

    from vllm import LLM, SamplingParams
    from vllm.v1.core.sched.scheduler import Scheduler

    class RecordingScheduler(Scheduler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.signal: list[tuple[int, int, str | None]] = []

        def schedule(self):
            output = super().schedule()
            longest = max(
                (r.num_cumulative_output_tokens for r in self.requests.values()),
                default=0,
            )
            self.signal.append(
                (
                    output.num_unfinished_requests,
                    longest,
                    output.dual_precision_base_precision,
                )
            )
            return output

    llm = LLM(
        model=model,
        max_model_len=1024,
        max_num_seqs=NUM_PROMPTS,
        gpu_memory_utilization=0.45,
        enforce_eager=True,
        scheduler_cls=RecordingScheduler,
        seed=0,
    )
    try:
        engine_core = llm.llm_engine.engine_core.engine_core
        scheduler = engine_core.scheduler
        assert isinstance(scheduler, RecordingScheduler)
        assert scheduler.precision_switcher is not None
        assert scheduler.precision_switcher.policy.kind == "fixed_frontier"

        prompts = [
            f"Question {i}: explain why the sky is blue in detail."
            for i in range(NUM_PROMPTS)
        ]
        params = SamplingParams(max_tokens=MAX_TOKENS, ignore_eos=True, temperature=0.0)
        outputs = llm.generate(prompts, params)
        assert len(outputs) == NUM_PROMPTS
        assert all(len(o.outputs[0].token_ids) == MAX_TOKENS for o in outputs)

        signal = scheduler.signal
        assert signal, "schedule() was never observed"
        assert all(p in ("bf16", "int4") for _, _, p in signal)
        first_int4 = next(i for i, (_, _, p) in enumerate(signal) if p == "int4")
        assert first_int4 > 0
        assert all(p == "bf16" for _, _, p in signal[:first_int4])
        assert all(p == "int4" for n, _, p in signal[first_int4:] if n > 0)
        # The switch fires at the first step where the longest response has
        # reached the frontier; the step before was still below it.
        assert signal[first_int4][1] >= FRONTIER
        assert signal[first_int4 - 1][1] < FRONTIER
        assert all(n == NUM_PROMPTS for n, _, _ in signal[:first_int4])

        switches = scheduler.precision_switcher.switches
        assert len(switches) == 1
        event = switches[0]
        assert event.committed_frontier == FRONTIER
        assert event.applied_response_tokens >= FRONTIER
        assert event.applied_live_requests == NUM_PROMPTS
        assert len(event.cohort) == NUM_PROMPTS

        assert cohorts.exists()
        rows = [json.loads(line) for line in cohorts.read_text().splitlines()]
        assert len(rows) == 1
        record = rows[0]
        assert record["event"] == "switch_cohort"
        assert record["rollout_index"] == 1
        assert record["policy_kind"] == "fixed_frontier"
        assert record["policy_revision"] == 0
        assert set(record["trigger"]) == {
            "committed_frontier",
            "applied_response_tokens",
            "applied_live_requests",
            "decision_live_requests",
            "median_prompt_tokens",
            "reason",
        }
        assert record["trigger"]["committed_frontier"] == FRONTIER
        assert len(record["requests"]) == NUM_PROMPTS
        for entry in record["requests"]:
            assert set(entry) == {"request_id", "entry_output_tokens", "prompt_tokens"}
            assert FRONTIER - 2 <= entry["entry_output_tokens"] <= FRONTIER + 2
            assert entry["prompt_tokens"] > 0
    finally:
        del llm


if __name__ == "__main__":
    raise SystemExit(
        pytest.main([__file__, "-m", "gpu_smoke", "-p", "no:cacheprovider", "-q", "-s"])
    )

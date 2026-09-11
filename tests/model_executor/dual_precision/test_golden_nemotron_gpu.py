# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Golden: dual-resident W4 greedy ids on Nemotron-Nano-9B-v2 reproduce the
stage0 weight audit (``dual_w4_stream0_greedy/responses.jsonl``).

Protocol (run_config.json of the archive): 2 BigMath-hard prompts, temperature
0, max_tokens 1024, LoRA enabled with no adapter, INT4 selected for every
forward, ``BF16_LAYERS=none``. The archive was produced at vLLM 34e66a3 and is
bit-identical across two runs there; the clean branch must match it, and the
prefix agreement with ``standalone_w4_greedy`` must be at least the recorded
215 / 671 tokens (the dual-resident W4 model keeps 27 conv1d linears in BF16).

Preconditions that other components supply, each a skip until merged:
* C9: ``VLLM_MARLIN_INPUT_PADDING`` (the RedHatAI w4a16 down_proj has
  K=15680, not a Marlin tile multiple; vanilla finds no kernel).
* C4: ``VLLM_DUAL_PRECISION_POLICY=uniform_w4`` (INT4 on every forward through
  the runner's bind call sites).
"""

import json
import os
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.gpu_smoke

ARCHIVE = Path(
    "/data/huanchen/verl/.codex-report/precision-scheduling-validation/stage0/weight_audit"
)
HF_HUB = "/data/huggingface/hub"
NEMOTRON_BF16 = (
    f"{HF_HUB}/models--nvidia--NVIDIA-Nemotron-Nano-9B-v2/snapshots/"
    "6533e8de2c68e4536bf7c411d7a3ce5734111476"
)
NEMOTRON_W4A16 = (
    f"{HF_HUB}/models--RedHatAI--NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16/"
    "snapshots/6cd9a7fcbb09bff44a641f23bd9e6afe8a96c7c6"
)
RECORDED_PREFIX_AGREEMENT = {"bigmath_hard-000": 671, "bigmath_hard-001": 215}


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _prefix_agreement(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def _skip_unless_ready() -> None:
    import vllm.envs as envs

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    for path in (NEMOTRON_BF16, NEMOTRON_W4A16):
        if not os.path.isdir(path):
            pytest.skip(f"checkpoint not available: {path}")
    if not (ARCHIVE / "dual_w4_stream0_greedy" / "responses.jsonl").exists():
        pytest.skip(f"archive not available: {ARCHIVE}")
    if "VLLM_MARLIN_INPUT_PADDING" not in envs.environment_variables:
        pytest.skip("VLLM_MARLIN_INPUT_PADDING (C9 Marlin K-padding) not merged")
    if "VLLM_DUAL_PRECISION_POLICY" not in envs.environment_variables:
        pytest.skip("VLLM_DUAL_PRECISION_POLICY (C4 runtime switch) not merged")


def test_archive_is_self_consistent():
    """CPU part: the two archived dual runs agree, standalone diverges as recorded."""
    if not (ARCHIVE / "dual_w4_stream0_greedy" / "responses.jsonl").exists():
        pytest.skip(f"archive not available: {ARCHIVE}")
    stream0 = {
        r["prompt_id"]: r
        for r in _load_jsonl(ARCHIVE / "dual_w4_stream0_greedy" / "responses.jsonl")
    }
    stream1 = {
        r["prompt_id"]: r
        for r in _load_jsonl(ARCHIVE / "dual_w4_stream1_greedy" / "responses.jsonl")
    }
    standalone = {
        r["prompt_id"]: r
        for r in _load_jsonl(ARCHIVE / "standalone_w4_greedy" / "responses.jsonl")
    }
    assert set(stream0) == set(RECORDED_PREFIX_AGREEMENT)
    for prompt_id, expected in RECORDED_PREFIX_AGREEMENT.items():
        assert (
            stream0[prompt_id]["output_token_ids"]
            == stream1[prompt_id]["output_token_ids"]
        )
        assert (
            _prefix_agreement(
                stream0[prompt_id]["output_token_ids"],
                standalone[prompt_id]["output_token_ids"],
            )
            == expected
        )


def test_dual_resident_w4_greedy_matches_archive(monkeypatch):
    _skip_unless_ready()
    from vllm import LLM, SamplingParams

    monkeypatch.setenv("VLLM_DUAL_PRECISION_ROLLOUT", "1")
    monkeypatch.setenv("VLLM_DUAL_PRECISION_INT4_MODEL", NEMOTRON_W4A16)
    monkeypatch.setenv("VLLM_DUAL_PRECISION_BF16_LAYERS", "none")
    monkeypatch.setenv("VLLM_DUAL_PRECISION_POLICY", "uniform_w4")
    monkeypatch.setenv("VLLM_MARLIN_INPUT_PADDING", "1")
    golden = {
        r["prompt_id"]: r
        for r in _load_jsonl(ARCHIVE / "dual_w4_stream0_greedy" / "responses.jsonl")
    }
    standalone = {
        r["prompt_id"]: r
        for r in _load_jsonl(ARCHIVE / "standalone_w4_greedy" / "responses.jsonl")
    }

    llm = LLM(
        model=NEMOTRON_BF16,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=17408,
        max_num_seqs=2,
        gpu_memory_utilization=0.85,
        seed=20260728,
        trust_remote_code=True,
    )
    prompts = [
        {"prompt_token_ids": golden[p]["prompt_token_ids"]} for p in sorted(golden)
    ]
    outputs = llm.generate(
        prompts, SamplingParams(temperature=0.0, top_p=0.95, top_k=20, max_tokens=1024)
    )
    for prompt_id, output in zip(sorted(golden), outputs):
        ids = list(output.outputs[0].token_ids)
        assert ids == golden[prompt_id]["output_token_ids"], prompt_id
        assert (
            _prefix_agreement(ids, standalone[prompt_id]["output_token_ids"])
            >= RECORDED_PREFIX_AGREEMENT[prompt_id]
        )

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU smoke: BF16-only 2x2 heatmap with the synthetic KV connector and decode barrier.

Run under the decision-13 launcher, e.g. ``run_gpu.sh --gpus 6 --timeout 1500 --
python -m pytest -m gpu_smoke tools/precision_scheduler`` with
``TPOT_HEATMAP_SMOKE_MODEL`` pointing at a Qwen3.5-4B/9B checkpoint.  The W4 row needs
the dual-precision runtime (C2/C4) and is not part of this smoke.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from tpot_heatmap import completed_keys, load_rows

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixtures" / "smoke_qwen35_4b_bf16_2x2.json"
DEFAULT_MODEL = (
    "/data/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/"
    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
)


def check_rows(rows: list[dict], repetitions: int) -> None:
    ok = [r for r in rows if r["status"] == "ok"]
    assert len(ok) == 4, rows
    assert completed_keys(rows) == {
        (1, 1024, "bf16"),
        (1, 4096, "bf16"),
        (8, 1024, "bf16"),
        (8, 4096, "bf16"),
    }
    for row in ok:
        # the barrier released exactly once per repetition and held every request
        assert row["barrier_release_count"] == repetitions
        assert row["first_token_spread_ms"] == 0.0
        # synthetic KV: all but the final prompt token came from the connector
        assert row["external_kv_tokens_per_request"] == row["seq_len"] - 1
        assert len(row["request_tpot_ms"]) == repetitions * row["measurement_steps"]
        assert 1.0 < row["median_tpot_ms"] < 100.0


def test_archived_smoke_evidence_is_consistent():
    payload = json.loads(FIXTURE.read_text())
    check_rows(payload["cells"], payload["manifest"]["measurement_repetitions"])
    assert payload["manifest"]["policy_env"] == {"VLLM_DUAL_PRECISION_POLICY": None}


@pytest.mark.gpu_smoke
def test_bf16_2x2_heatmap_smoke(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.skip("run under run_gpu.sh (decision 13): CUDA_VISIBLE_DEVICES is unset")
    model = os.environ.get("TPOT_HEATMAP_SMOKE_MODEL", DEFAULT_MODEL)
    if not Path(model).exists():
        pytest.skip(f"checkpoint not found: {model}")
    output = tmp_path / "heatmap"
    command = [
        sys.executable,
        str(HERE / "tpot_heatmap.py"),
        "--model",
        model,
        "--batch-sizes",
        "1,8",
        "--seq-lens",
        "1024,4096",
        "--precisions",
        "bf16",
        "--warmup-steps",
        "1",
        "--measurement-steps",
        "3",
        "--measurement-repetitions",
        "2",
        "--initial-precision-warmup-steps",
        "3",
        "--gpu-memory-utilization",
        "0.5",
        "--output-dir",
        str(output),
        "--no-plots",
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1500)
    assert result.returncode == 0, result.stderr[-4000:]
    check_rows(load_rows(output / "cells.jsonl"), repetitions=2)
    heatmap = json.loads((output / "heatmap.json").read_text())
    assert all(v is not None for row in heatmap["bf16_tpot_ms"] for v in row)
    assert all(v is None for row in heatmap["int4_tpot_ms"] for v in row)
    # a second invocation must not start an engine
    again = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert again.returncode == 0 and "no engine was started" in again.stdout

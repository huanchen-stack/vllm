# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for the heatmap helpers (no engine): axes, cells, resume, matrix."""

from __future__ import annotations

import pytest
from tpot_heatmap import (
    POLICY_ENV,
    KVCapacityError,
    append_jsonl,
    atomic_write_json,
    build_cells,
    completed_keys,
    expected_keys,
    is_capacity_failure,
    load_rows,
    matrix_payload,
    median_speedup,
    parse_args,
    parse_axis,
    percentile,
    precision_environment,
    preserved_run_status,
)


def test_parse_axis_list_and_inclusive_range():
    assert parse_axis("8,1,4,4") == [1, 4, 8]
    assert parse_axis("2:8:2") == [2, 4, 6, 8]
    with pytest.raises(ValueError):
        parse_axis("1,2:4")
    with pytest.raises(ValueError):
        parse_axis("0,1")


def test_cells_are_sorted_by_required_kv():
    cells = build_cells([1, 4], [128, 512], 10, [(16, 1024), (1, 256)])
    requirements = [cell.required_kv_bytes for cell in cells]
    assert requirements == sorted(requirements)
    assert cells[0].required_blocks_by_group == (9, 139)  # 1 * ceil(139/16), 1 * 139


def test_matrix_and_resume_keys():
    rows = [
        {
            "status": "ok",
            "batch_size": 1,
            "seq_len": 128,
            "precision": "bf16",
            "median_tpot_ms": 2.0,
            "required_kv_bytes": 10,
        },
        {
            "status": "ok",
            "batch_size": 1,
            "seq_len": 128,
            "precision": "int4",
            "median_tpot_ms": 1.0,
            "required_kv_bytes": 10,
        },
        {
            "status": "capacity_failure",
            "batch_size": 2,
            "seq_len": 128,
            "precision": "int4",
            "required_kv_bytes": 20,
        },
    ]
    assert completed_keys(rows) == {(1, 128, "bf16"), (1, 128, "int4")}
    payload = matrix_payload(rows, [1, 2], [128])
    assert payload["required_kv_bytes"] == [[10], [20]]
    assert payload["speedup_bf16_over_int4"] == [[2.0], [None]]
    assert payload["bf16_tpot_ms"] == [[2.0], [None]]
    assert median_speedup(payload) == 2.0
    assert expected_keys([1, 2], [128], ("bf16",)) == {
        (1, 128, "bf16"),
        (2, 128, "bf16"),
    }


def test_preserved_run_does_not_repeat_failure():
    expected = {(1, 128, "bf16"), (1, 128, "int4")}
    interrupted = {
        "status": "running_cell",
        "current_cell": {"batch_size": 1, "seq_len": 128, "precision": "int4"},
    }
    assert preserved_run_status(interrupted, {(1, 128, "bf16")}, expected, False) == 2
    assert preserved_run_status(interrupted, {(1, 128, "bf16")}, expected, True) is None
    assert (
        preserved_run_status({"status": "capacity_exhausted"}, set(), expected, True)
        == 3
    )
    assert preserved_run_status({"status": "complete"}, expected, expected, False) == 0
    assert (
        preserved_run_status(
            {"status": "running", "current_cell": None}, set(), expected, False
        )
        is None
    )


def test_atomic_and_append_outputs(tmp_path):
    json_path = tmp_path / "progress.json"
    rows_path = tmp_path / "cells.jsonl"
    atomic_write_json(json_path, {"status": "running"})
    assert json_path.read_text() == '{\n  "status": "running"\n}\n'
    append_jsonl(rows_path, {"cell": 1})
    append_jsonl(rows_path, {"cell": 2})
    assert load_rows(rows_path) == [{"cell": 1}, {"cell": 2}]
    assert not list(tmp_path.glob(".*.tmp"))
    rows_path.write_text('{"a": 1}\nnot json\n')
    with pytest.raises(ValueError, match="Invalid JSONL"):
        load_rows(rows_path)


def test_cuda_oom_is_not_recoverable_kv_capacity():
    assert is_capacity_failure(KVCapacityError("KV cache capacity"))
    assert is_capacity_failure(RuntimeError("Not enough KV cache blocks"))
    assert not is_capacity_failure(RuntimeError("CUDA out of memory"))


def test_percentile():
    assert percentile([], 0.5) != percentile([], 0.5)  # NaN
    assert percentile([3.0], 0.9) == 3.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == 2.5


def test_precision_environment_is_the_policy_seam():
    assert precision_environment("bf16", None) == {POLICY_ENV: None}
    int4 = precision_environment("int4", "/ckpt/int4")
    assert int4[POLICY_ENV] == "uniform_w4"
    assert int4["VLLM_DUAL_PRECISION_INT4_MODEL"] == "/ckpt/int4"
    with pytest.raises(ValueError, match="--int4-model"):
        precision_environment("int4", None)
    with pytest.raises(ValueError):
        precision_environment("fp8", None)


def test_parse_args_validation(tmp_path):
    base = [
        "--model",
        "m",
        "--batch-sizes",
        "1",
        "--seq-lens",
        "64",
        "--output-dir",
        str(tmp_path),
    ]
    args = parse_args(base + ["--precisions", "bf16"])
    assert args.precision_list == ("bf16",)
    assert args.language_model_only is True
    with pytest.raises(ValueError):
        parse_args(base + ["--precisions", "bf16,bf16"])
    with pytest.raises(ValueError):
        parse_args(base + ["--measurement-steps", "0"])
    with pytest.raises(ValueError):
        parse_args(base + ["--synthetic-float-value", "0"])

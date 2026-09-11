# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU smoke: padded Marlin vs Triton W4A16 on the real Nemotron-Nano-9B-v2
mixer.down_proj (K=15680 -> 15744).  Skips when the snapshot is absent."""

import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools" / "precision_scheduler" / "models"))

DEFAULT_SNAPSHOT = (
    "/data/huggingface/hub/models--RedHatAI--NVIDIA-Nemotron-Nano-9B-v2-quantized.w4a16"
    "/snapshots/6cd9a7fcbb09bff44a641f23bd9e6afe8a96c7c6"
)
SNAPSHOT = Path(os.environ.get("NEMOTRON_W4A16_SNAPSHOT", DEFAULT_SNAPSHOT))

pytestmark = [pytest.mark.gpu_smoke, pytest.mark.skip_global_cleanup]


@pytest.fixture(scope="module")
def dist_env():
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        cleanup_dist_env_and_memory,
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
    yield
    cleanup_dist_env_and_memory()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
@pytest.mark.skipif(
    not SNAPSHOT.exists(),
    reason=f"Nemotron W4A16 snapshot not found at {SNAPSHOT} (NEMOTRON_W4A16_SNAPSHOT)",
)
def test_padded_marlin_matches_triton_on_nemotron_down_proj(dist_env):
    from validate_nemotron_marlin_padding import compare_padded_marlin_vs_triton

    result = compare_padded_marlin_vs_triton(SNAPSHOT)

    assert result["original_k"] == 15680 and result["padded_k"] == 15744
    assert result["cosine"] >= 0.99999, result
    # Reported for the record (archived oracle:
    # nemotron_padded_marlin_numerical_validation.json).
    print({k: result[k] for k in ("cosine", "max_abs", "relative_rmse")})

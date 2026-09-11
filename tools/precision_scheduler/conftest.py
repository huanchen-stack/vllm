# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "gpu_smoke: one-GPU smoke tests run on request under run_gpu.sh"
    )

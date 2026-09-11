# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU tests for inject_extra_prompt_logprobs (VLLM_PROMPT_LOGPROB_EXTRA_TOKEN_IDS)."""

import pytest
import torch

from vllm.v1.sample.prompt_logprob_extra import inject_extra_prompt_logprobs


def _gathered(raw: torch.Tensor, k: int, targets: list[int]):
    """Mimic Sampler.gather_logprobs: column 0 is the target token, then the top-k."""
    topk_lp, topk_ids = raw.topk(k, dim=-1)
    tgt = torch.tensor(targets)
    token_ids = torch.cat([tgt[:, None], topk_ids], dim=-1)
    logprobs = torch.cat([raw.gather(-1, tgt[:, None]), topk_lp], dim=-1)
    return token_ids, logprobs


def test_replaces_last_columns_with_exact_logprobs():
    torch.manual_seed(0)
    raw = torch.log_softmax(torch.randn(4, 16), dim=-1)
    token_ids, logprobs = _gathered(raw, k=5, targets=[1, 2, 3, 4])
    before_ids, before_lp = token_ids.clone(), logprobs.clone()

    out_ids, out_lp = inject_extra_prompt_logprobs(
        token_ids, logprobs, raw, [11, 7], num_prompt_logprobs=5
    )

    assert out_ids is token_ids and out_lp is logprobs  # in place
    assert torch.equal(out_ids[:, :-2], before_ids[:, :-2])
    assert torch.equal(out_lp[:, :-2], before_lp[:, :-2])
    assert torch.equal(out_ids[:, -2:], torch.tensor([[11, 7]] * 4))
    assert torch.allclose(out_lp[:, -2:], raw[:, [11, 7]])


def test_empty_extra_ids_is_noop():
    raw = torch.log_softmax(torch.randn(2, 8), dim=-1)
    token_ids, logprobs = _gathered(raw, k=3, targets=[0, 1])
    before = token_ids.clone(), logprobs.clone()
    inject_extra_prompt_logprobs(token_ids, logprobs, raw, [], num_prompt_logprobs=3)
    assert torch.equal(token_ids, before[0]) and torch.equal(logprobs, before[1])


def test_rejects_more_ids_than_prompt_logprobs():
    raw = torch.log_softmax(torch.randn(2, 8), dim=-1)
    token_ids, logprobs = _gathered(raw, k=1, targets=[0, 1])
    with pytest.raises(ValueError):
        inject_extra_prompt_logprobs(
            token_ids, logprobs, raw, [3, 4], num_prompt_logprobs=1
        )

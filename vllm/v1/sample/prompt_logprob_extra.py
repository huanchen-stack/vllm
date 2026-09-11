# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in diagnostics: exact prompt logprobs for a fixed set of token ids.

Behind ``VLLM_PROMPT_LOGPROB_EXTRA_TOKEN_IDS``. In prompt-logprob mode the sampler
returns the top-k ``(token_ids, logprobs)`` per prompt position; the EOS-hazard and
layer-sensitivity studies need the exact logprob of specific ids (e.g. the EOS ids)
at every position, which the top-k rows do not guarantee to contain. The helper
overwrites the last ``len(extra_ids)`` top-k columns with those ids and their exact
logprobs gathered from the full distribution.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def inject_extra_prompt_logprobs(
    token_ids: torch.Tensor,
    logprobs: torch.Tensor,
    raw_logprobs: torch.Tensor,
    extra_ids: Sequence[int],
    num_prompt_logprobs: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Replace the last ``len(extra_ids)`` columns of ``token_ids``/``logprobs``.

    The tensors are modified in place.

    Args:
        token_ids: ``[num_tokens, num_prompt_logprobs + 1]`` gathered top-k token ids
            (column 0 is the target token).
        logprobs: matching gathered logprobs.
        raw_logprobs: ``[num_tokens, vocab]`` full log-softmax the top-k was
            gathered from.
        extra_ids: token ids whose exact logprobs are wanted.
        num_prompt_logprobs: the request's ``prompt_logprobs`` (top-k columns).

    Returns:
        The same two tensors (modified in place). ``extra_ids`` empty is a no-op.

    Raises:
        ValueError: when ``num_prompt_logprobs < len(extra_ids)``; the columns would
            otherwise overwrite the target-token column.
    """
    num_extra = len(extra_ids)
    if num_extra == 0:
        return token_ids, logprobs
    if num_prompt_logprobs < num_extra:
        raise ValueError(
            "prompt_logprobs must be at least the number of "
            f"VLLM_PROMPT_LOGPROB_EXTRA_TOKEN_IDS ({num_prompt_logprobs} < {num_extra})"
        )
    extra = torch.tensor(
        list(extra_ids), device=token_ids.device, dtype=token_ids.dtype
    )
    extra_logprobs = raw_logprobs.index_select(dim=-1, index=extra.to(torch.long))
    token_ids[:, -num_extra:] = extra
    logprobs[:, -num_extra:] = extra_logprobs.to(logprobs.dtype)
    return token_ids, logprobs

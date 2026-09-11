# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compatibility config for encoder-free Gemma 4 Unified checkpoints.

Transformers 5.5 contains the Gemma 4 text configuration used by the Unified
12B/31B checkpoints, but does not yet register the ``gemma4_unified`` top-level
model type.  For language-model-only serving the nested text configuration is
wire-compatible with :class:`Gemma4Config`; the vision/audio pipeline is not
constructed by vLLM's text-only registry entry.
"""

from transformers.models.gemma4.configuration_gemma4 import Gemma4Config


class Gemma4UnifiedConfig(Gemma4Config):
    """Parse Gemma 4 Unified checkpoints with Transformers 5.5."""

    model_type = "gemma4_unified"


__all__ = ["Gemma4UnifiedConfig"]

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derive vLLM ``LinearBase`` module-name fixtures from HF checkpoints.

The fixtures feed ``test_attach_counts.py``. They are derived from the
safetensors headers of a BF16 checkpoint and its INT4 shadow with the HF ->
vLLM module-name mapping (fused ``qkv_proj`` / ``gate_up_proj`` / ``in_proj_*``
and the ``model.language_model`` -> ``language_model.model`` prefix), so the
test never needs a GPU or a model download.

The Qwen3.5-9B list was cross-checked against a meta-device
``initialize_model`` of the real vLLM model (286 ``LinearBase`` modules, 152
quantized); the Nemotron and Gemma lists reproduce the ``Loaded N GPTQ shadow
linear layers`` counts in the archived run logs.

Usage::

    python derive_module_names.py <name> <bf16_snapshot> <int4_snapshot>
"""

from __future__ import annotations

import json
import re
import struct
import sys
from pathlib import Path

_FUSED = {
    "q_proj": "qkv_proj",
    "k_proj": "qkv_proj",
    "v_proj": "qkv_proj",
    "gate_proj": "gate_up_proj",
    "up_proj": "gate_up_proj",
    "in_proj_qkv": "in_proj_qkvz",
    "in_proj_z": "in_proj_qkvz",
    "in_proj_b": "in_proj_ba",
    "in_proj_a": "in_proj_ba",
}
_QUANT_SUFFIXES = (".qweight", ".weight_packed")


def _tensor_names(snapshot: Path) -> list[str]:
    index = snapshot / "model.safetensors.index.json"
    if index.exists():
        return sorted(json.loads(index.read_text())["weight_map"])
    names: list[str] = []
    for file in sorted(snapshot.glob("*.safetensors")):
        with file.open("rb") as stream:
            size = struct.unpack("<Q", stream.read(8))[0]
            names.extend(json.loads(stream.read(size)))
    return sorted(names)


def to_vllm_name(hf_module: str, siblings: set[str]) -> str:
    """Map an HF linear module name onto its vLLM module name."""
    name = hf_module
    for hf_prefix, vllm_prefix in (
        ("model.language_model.", "language_model.model."),
        ("model.visual.", "visual."),
        ("model.embed_vision.", "embed_vision."),
        ("model.embed_audio.", "embed_audio."),
    ):
        if name.startswith(hf_prefix):
            name = vllm_prefix + name[len(hf_prefix) :]
            break
    head, _, leaf = hf_module.rpartition(".")
    # ``up_proj`` is only fused into ``gate_up_proj`` next to a ``gate_proj``
    # (Nemotron-H MLPs keep a plain ``up_proj``).
    if leaf == "up_proj" and f"{head}.gate_proj" not in siblings:
        return name
    vhead, _, vleaf = name.rpartition(".")
    vleaf = _FUSED.get(vleaf, vleaf)
    return f"{vhead}.{vleaf}" if vhead else vleaf


def derive(bf16_snapshot: Path, int4_snapshot: Path, linear_filter) -> dict:
    bf16_tensors = _tensor_names(bf16_snapshot)
    int4_tensors = _tensor_names(int4_snapshot)
    bf16_hf = {
        t[: -len(".weight")]
        for t in bf16_tensors
        if t.endswith(".weight") and linear_filter(t[: -len(".weight")])
    }
    int4_hf = {
        t[: -len(suffix)]
        for t in int4_tensors
        for suffix in _QUANT_SUFFIXES
        if t.endswith(suffix)
    }
    bf16_modules = sorted({to_vllm_name(n, bf16_hf) for n in bf16_hf})
    int4_quantized = sorted({to_vllm_name(n, int4_hf) for n in int4_hf})
    return {
        "bf16_linear_modules": bf16_modules,
        "int4_quantized_modules": int4_quantized,
    }


def main() -> None:
    name, bf16_snapshot, int4_snapshot = sys.argv[1:4]
    # Linear leaves that vLLM builds as ``LinearBase``. Excluded: ``mtp.*``
    # (drafter, not loaded), ``patch_embed.proj`` (Conv3d), and the Gemma4
    # vision/audio towers (plain ``nn.Linear`` in vLLM); the Gemma4
    # ``per_layer_*`` projections and multimodal embedding projections are
    # ``LinearBase``.
    linear = re.compile(
        r"(proj|projection|fc\d|linear|qkv|conv1d|in_proj_[a-z]+|per_layer_input_gate"
        r"|embedding_projection)$"
    )

    def is_linear(module: str) -> bool:
        return bool(linear.search(module)) and not module.startswith(
            (
                "mtp.",
                "model.visual.patch_embed",
                "model.vision_tower",
                "model.audio_tower",
            )
        )

    result = derive(Path(bf16_snapshot), Path(int4_snapshot), is_linear)
    out = Path(__file__).with_name(f"{name}_modules.json")
    out.write_text(json.dumps(result, indent=0) + "\n")
    print(
        out, len(result["bf16_linear_modules"]), len(result["int4_quantized_modules"])
    )


if __name__ == "__main__":
    main()

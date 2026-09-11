# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Audit a BF16 / W4A16 checkpoint pair and build a zero-output LoRA adapter.

Defaults target Phi-4-mini-reasoning (microsoft/Phi-4-mini-reasoning +
alishafique/Phi-4-mini-reasoning-quantized.gptq-w4a16-llmcompressor) but every
path is a CLI argument, so the same tool prepares any dense pair.

Steps:

1. Audit: the W4 checkpoint must be compressed-tensors (or GPTQ/AutoRound)
   4-bit, not AWQ; both checkpoints must agree on model_type, vocab size, EOS id
   and chat template; a stable tokenizer fingerprint is recorded for both.
2. Optionally render chat prompts from a JSONL of ``{"messages": [...]}`` rows
   (``--prompts-jsonl``) and check every prompt ends with ``--assistant-marker``
   (``<|assistant|>`` for Phi-4) so generation enters the reasoning turn.
3. Build a rank-``--rank`` LoRA adapter with ``lora_B == 0`` whose
   ``target_modules`` are the model's native fused linear names (Phi3:
   qkv_proj / gate_up_proj / o_proj / down_proj).  A zero adapter reproduces the
   base model exactly and is the LoRA the rollout engine is primed with.

No manifests are mutated; the audit is written to ``--output-dir/audit.json``.
Phi-4-mini-flash-reasoning (Phi4FlashForCausalLM) is unsupported by vLLM V1 and
is rejected by the audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

DEFAULT_BF16 = "microsoft/Phi-4-mini-reasoning"
DEFAULT_W4 = "alishafique/Phi-4-mini-reasoning-quantized.gptq-w4a16-llmcompressor"
UNSUPPORTED_ARCHITECTURES = {"Phi4FlashForCausalLM"}
ACCEPTED_QUANT_METHODS = {"compressed-tensors", "gptq", "auto-round"}


def stable_tokenizer_fingerprint(tokenizer: Any) -> str:
    payload = {
        "class": type(tokenizer).__name__,
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template": tokenizer.chat_template,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def quantization_summary(config: Any) -> dict[str, Any]:
    raw = getattr(config, "quantization_config", None)
    if raw is None and hasattr(config, "text_config"):
        raw = getattr(config.text_config, "quantization_config", None)
    if hasattr(raw, "to_dict"):
        raw = raw.to_dict()
    raw = dict(raw or {})
    method = raw.get("quant_method")
    bits = raw.get("bits")
    if bits is None:
        groups = raw.get("config_groups") or {}
        for group in groups.values():
            weights = group.get("weights") or {}
            if "num_bits" in weights:
                bits = weights["num_bits"]
                break
    return {"quant_method": method, "num_bits": bits, "raw": raw}


def audit_pair(
    bf16_cfg: Any, w4_cfg: Any, bf16_tok: Any, w4_tok: Any
) -> dict[str, Any]:
    """Raise RuntimeError on an unusable pair; return the audit record otherwise."""
    architectures = list(getattr(bf16_cfg, "architectures", None) or [])
    if set(architectures) & UNSUPPORTED_ARCHITECTURES:
        raise RuntimeError(f"{architectures} is not supported by vLLM V1")
    quant = quantization_summary(w4_cfg)
    if quant["quant_method"] not in ACCEPTED_QUANT_METHODS or quant["num_bits"] != 4:
        raise RuntimeError(
            "expected a compressed-tensors/GPTQ/AutoRound W4A16 checkpoint, got "
            f"{quant['quant_method']} {quant['num_bits']}-bit (AWQ is not accepted)"
        )
    if (bf16_cfg.model_type, bf16_cfg.vocab_size) != (
        w4_cfg.model_type,
        w4_cfg.vocab_size,
    ):
        raise RuntimeError("architecture mismatch between BF16 and W4 checkpoints")
    if (
        len(bf16_tok) != len(w4_tok)
        or bf16_tok.eos_token_id != w4_tok.eos_token_id
        or bf16_tok.chat_template != w4_tok.chat_template
    ):
        raise RuntimeError("tokenizer / chat template mismatch between checkpoints")
    return {
        "architectures": architectures,
        "model_type": bf16_cfg.model_type,
        "vocab_size": len(bf16_tok),
        "eos_token_id": bf16_tok.eos_token_id,
        "max_position_embeddings": getattr(bf16_cfg, "max_position_embeddings", None),
        "quant_method": quant["quant_method"],
        "num_bits": quant["num_bits"],
        "quantization_config": quant["raw"],
        "bf16_tokenizer_fingerprint": stable_tokenizer_fingerprint(bf16_tok),
        "w4_tokenizer_fingerprint": stable_tokenizer_fingerprint(w4_tok),
    }


def linear_shapes(snapshot: Path) -> dict[str, tuple[int, int]]:
    """Return {module_name: (out_features, in_features)} for every 2-D layer weight."""
    from safetensors import safe_open

    shapes: dict[str, tuple[int, int]] = {}
    for file in sorted(snapshot.glob("*.safetensors")):
        with safe_open(str(file), framework="pt", device="cpu") as handle:
            for key in handle.keys():  # noqa: SIM118 - not a dict
                if not key.endswith(".weight") or ".layers." not in key:
                    continue
                shape = tuple(handle.get_slice(key).get_shape())
                if len(shape) != 2:
                    continue
                name = key[: -len(".weight")]
                leaf = name.rsplit(".", 1)[-1]
                if leaf in {"embed_tokens", "lm_head"} or "norm" in leaf:
                    continue
                shapes[name] = (int(shape[0]), int(shape[1]))
    # Multimodal checkpoints (Gemma-4) also carry vision/audio tower linears; the
    # rollout engine is text-only, so restrict to the language tower if present.
    language = {
        name: shape
        for name, shape in shapes.items()
        if name.startswith("model.language_model.layers.")
    }
    if language:
        shapes = language
    if not shapes:
        raise RuntimeError(f"no linear layer shapes found in {snapshot}")
    return shapes


def build_zero_lora(
    shapes: dict[str, tuple[int, int]],
    output: Path,
    base_model: str,
    rank: int = 16,
    seed: int = 0,
) -> dict[str, Any]:
    """Write a PEFT LoRA adapter with random lora_A and all-zero lora_B."""
    from safetensors.torch import save_file

    targets = sorted({name.rsplit(".", 1)[-1] for name in shapes})
    generator = torch.Generator().manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for name, (out_features, in_features) in shapes.items():
        prefix = f"base_model.model.{name}"
        tensors[f"{prefix}.lora_A.weight"] = (
            torch.randn(rank, in_features, generator=generator, dtype=torch.float32)
            .mul_(0.01)
            .to(torch.float16)
        )
        tensors[f"{prefix}.lora_B.weight"] = torch.zeros(
            out_features, rank, dtype=torch.float16
        )
    output.mkdir(parents=True, exist_ok=True)
    config = {
        "peft_type": "LORA",
        "base_model_name_or_path": base_model,
        "task_type": "CAUSAL_LM",
        "inference_mode": True,
        "r": rank,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "bias": "none",
        "target_modules": targets,
        "init_lora_weights": True,
        "use_rslora": False,
        "use_dora": False,
    }
    (output / "adapter_config.json").write_text(json.dumps(config, indent=2) + "\n")
    save_file(tensors, str(output / "adapter_model.safetensors"))
    inventory = {
        "rank": rank,
        "linear_modules": len(shapes),
        "target_module_leaves": targets,
        "all_lora_B_zero": True,
    }
    (output / "inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    return inventory


def render_prompts(
    tokenizer: Any, prompts_jsonl: Path, output: Path, assistant_marker: str
) -> dict[str, Any]:
    digest = hashlib.sha256()
    count = 0
    max_prompt = 0
    with (
        prompts_jsonl.open(encoding="utf-8") as source,
        output.open("w", encoding="utf-8") as target,
    ):
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            prompt = tokenizer.apply_chat_template(
                row["messages"], tokenize=False, add_generation_prompt=True
            )
            if assistant_marker and not prompt.endswith(assistant_marker):
                raise RuntimeError(
                    f"prompt does not end with {assistant_marker!r}; generation "
                    "would not enter the assistant turn"
                )
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            payload = {
                k: v for k, v in row.items() if k not in {"messages", "question"}
            }
            payload.update({"prompt": prompt, "prompt_tokens": prompt_tokens})
            encoded = (json.dumps(payload, ensure_ascii=False) + "\n").encode()
            digest.update(encoded)
            target.write(encoded.decode())
            count += 1
            max_prompt = max(max_prompt, prompt_tokens)
    return {
        "path": str(output),
        "rows": count,
        "sha256": digest.hexdigest(),
        "prompt_token_max": max_prompt,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--bf16", default=DEFAULT_BF16, help="BF16 checkpoint path or hub id"
    )
    parser.add_argument(
        "--w4", default=DEFAULT_W4, help="W4A16 checkpoint path or hub id"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--adapter-dir", type=Path, default=None, help="default: <output-dir>/adapter"
    )
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--prompts-jsonl", type=Path, default=None)
    parser.add_argument("--assistant-marker", default="<|assistant|>")
    parser.add_argument("--skip-adapter", action="store_true")
    args = parser.parse_args(argv)

    from transformers import AutoConfig, AutoTokenizer

    bf16_cfg = AutoConfig.from_pretrained(args.bf16)
    w4_cfg = AutoConfig.from_pretrained(args.w4)
    bf16_tok = AutoTokenizer.from_pretrained(args.bf16)
    w4_tok = AutoTokenizer.from_pretrained(args.w4)
    audit = audit_pair(bf16_cfg, w4_cfg, bf16_tok, w4_tok)
    audit.update({"bf16": str(args.bf16), "w4": str(args.w4)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.prompts_jsonl is not None:
        audit["rendered_prompts"] = render_prompts(
            bf16_tok,
            args.prompts_jsonl,
            args.output_dir / "rendered_prompts.jsonl",
            args.assistant_marker,
        )
    if not args.skip_adapter:
        bf16_path = Path(args.bf16)
        if not bf16_path.exists():
            from huggingface_hub import snapshot_download

            bf16_path = Path(
                snapshot_download(args.bf16, allow_patterns=["*.safetensors"])
            )
        adapter_dir = args.adapter_dir or (args.output_dir / "adapter")
        audit["adapter"] = build_zero_lora(
            linear_shapes(bf16_path), adapter_dir, str(args.bf16), rank=args.rank
        )
        audit["adapter"]["path"] = str(adapter_dir)

    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(
        json.dumps(
            {k: audit[k] for k in ("model_type", "eos_token_id", "quant_method")}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Create a synthetic rank-r PEFT LoRA adapter for a checkpoint.

By default the adapter is behaviour-preserving: A is Kaiming-initialised
(seeded from the module name, so the adapter is reproducible) and B is zero,
so ``base(x) + x @ A^T @ B^T == base(x)`` exactly while every LoRA kernel
still runs. ``--random-b`` produces a non-trivial delta instead.

Target modules are discovered from the checkpoint's safetensors index: every
2-D ``<prefix>.layers.<i>.<...>.<name>.weight`` whose ``<name>`` is in
``--target-modules`` gets a pair. For Qwen3.5 (weights under
``model.language_model``) the keys are also emitted under the text-only
``model.`` prefix, matching the adapters used for the archived runs.

Example:
  python tools/rollout_lora/make_zero_lora.py \\
      --model /path/to/Qwen3.5-9B --output /tmp/qwen35_9b_zero
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    # Qwen3.5 GatedDeltaNet
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, help="checkpoint dir or HF id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=None, help="default: rank")
    parser.add_argument(
        "--target-modules",
        default=",".join(DEFAULT_TARGETS),
        help="comma-separated module suffixes",
    )
    parser.add_argument(
        "--random-b",
        action="store_true",
        help="random (non-zero) B instead of the behaviour-preserving zero B",
    )
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    parser.add_argument(
        "--exclude-prefixes",
        default="mtp.",
        help="comma-separated weight-name prefixes to skip (default: the MTP head)",
    )
    return parser.parse_args()


def resolve_snapshot(model: str) -> Path:
    path = Path(model)
    if path.exists():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, local_files_only=True))


def weight_shapes(snapshot: Path) -> dict[str, tuple[int, ...]]:
    index_paths = sorted(snapshot.glob("*.safetensors.index.json"))
    if index_paths:
        weight_map = json.loads(index_paths[0].read_text(encoding="utf-8"))[
            "weight_map"
        ]
    else:
        weight_map = {}
        for tensor_path in sorted(snapshot.glob("*.safetensors")):
            with safe_open(tensor_path, framework="pt", device="cpu") as handle:
                weight_map.update({key: tensor_path.name for key in handle})
    if not weight_map:
        raise FileNotFoundError(f"No safetensors found under {snapshot}")
    shapes: dict[str, tuple[int, ...]] = {}
    by_file: dict[str, list[str]] = {}
    for name, file in weight_map.items():
        by_file.setdefault(file, []).append(name)
    for file, names in by_file.items():
        with safe_open(snapshot / file, framework="pt", device="cpu") as handle:
            for name in names:
                shapes[name] = tuple(int(v) for v in handle.get_slice(name).get_shape())
    return shapes


def stable_seed(name: str) -> int:
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "little")


def make_pair(
    module_name: str,
    in_features: int,
    out_features: int,
    rank: int,
    dtype: torch.dtype,
    random_b: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(stable_seed(module_name))
    lora_a = torch.empty(rank, in_features, dtype=torch.float32)
    torch.nn.init.kaiming_uniform_(lora_a, a=5**0.5, generator=generator)
    if random_b:
        lora_b = (
            torch.rand(out_features, rank, dtype=torch.float32, generator=generator)
            - 0.5
        ) * 0.02
    else:
        lora_b = torch.zeros(out_features, rank, dtype=torch.float32)
    return lora_a.to(dtype), lora_b.to(dtype)


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    targets = {t for t in args.target_modules.split(",") if t}
    excluded = tuple(p for p in args.exclude_prefixes.split(",") if p)
    snapshot = resolve_snapshot(args.model)
    shapes = weight_shapes(snapshot)

    tensors: dict[str, torch.Tensor] = {}
    used_targets: set[str] = set()
    for name in sorted(shapes):
        if not name.endswith(".weight") or ".layers." not in name:
            continue
        if excluded and name.startswith(excluded):
            continue
        module_name = name.removesuffix(".weight")
        suffix = module_name.rsplit(".", 1)[-1]
        shape = shapes[name]
        if suffix not in targets or len(shape) != 2:
            continue
        out_features, in_features = shape
        lora_a, lora_b = make_pair(
            module_name, in_features, out_features, args.rank, dtype, args.random_b
        )
        prefixes = [module_name]
        lm_prefix = "model.language_model."
        if module_name.startswith(lm_prefix):
            prefixes.append("model." + module_name.removeprefix(lm_prefix))
        for i, prefix in enumerate(prefixes):
            # safetensors refuses aliased storage: clone the duplicate prefix.
            tensors[f"base_model.model.{prefix}.lora_A.weight"] = (
                lora_a if i == 0 else lora_a.clone()
            )
            tensors[f"base_model.model.{prefix}.lora_B.weight"] = (
                lora_b if i == 0 else lora_b.clone()
            )
        used_targets.add(suffix)
    if not tensors:
        raise RuntimeError(f"no target modules {sorted(targets)} found in {snapshot}")

    adapter_config = {
        "peft_type": "LORA",
        "auto_mapping": None,
        "base_model_name_or_path": args.model,
        "revision": None,
        "task_type": "CAUSAL_LM",
        "inference_mode": False,
        "r": args.rank,
        "lora_alpha": args.alpha if args.alpha is not None else args.rank,
        "lora_dropout": 0.0,
        "fan_in_fan_out": False,
        "bias": "none",
        "modules_to_save": None,
        "init_lora_weights": True,
        "layers_to_transform": None,
        "layers_pattern": None,
        "target_modules": sorted(used_targets),
        "exclude_modules": None,
        "use_rslora": False,
        "use_dora": False,
        "loftq_config": {},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output / "adapter_model.safetensors"))
    (args.output / "adapter_config.json").write_text(
        json.dumps(adapter_config, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"{args.output}: {len(tensors) // 2} LoRA pairs, rank {args.rank}, "
        f"targets {sorted(used_targets)}, zero_b={not args.random_b}"
    )


if __name__ == "__main__":
    main()

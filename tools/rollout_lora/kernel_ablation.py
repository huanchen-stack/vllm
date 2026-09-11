#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fixed-work decode benchmark for the rollout LoRA kernel ablation.

Two sub-commands:

  run    Load one model + one LoRA adapter with a given (base backend, LoRA
         kernel) configuration and measure decode TPOT on fixed cells
         (batch x context, N decode tokens) behind a synchronized prefill
         barrier. Appends one JSONL row per cell to --output.

  table  Read the per-cell JSONL rows of a results directory and print a
         markdown table with median TPOT per stage, the per-stage spread over
         repetitions, the incremental gain per stage and (optionally) the
         archived reference numbers.

The four LoRA kernel stages map onto the env knobs read by vllm/envs.py:

  punica            ROLLOUT_QLORA=0                          (vanilla)
  torch             ROLLOUT_QLORA=1 VLLM_ROLLOUT_LORA_FUSE_PACKED=0
  torch-fused       ROLLOUT_QLORA=1 VLLM_ROLLOUT_LORA_FUSE_PACKED=1
  torch-fused-dual  ... plus VLLM_LORA_ENABLE_DUAL_STREAM=1

Dual precision is never enabled here (design decision 1): the archived numbers
were taken with it off. The env vars must be set before vLLM is imported, so
`run` sets them from --lora-kernel first and imports vllm afterwards.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

LORA_KERNELS = ("punica", "torch", "torch-fused", "torch-fused-dual")
BASE_BACKENDS = ("bf16", "marlin", "bitsandbytes")


def kernel_env(lora_kernel: str) -> dict[str, str]:
    torch_path = lora_kernel != "punica"
    fused = lora_kernel in ("torch-fused", "torch-fused-dual")
    dual_stream = lora_kernel == "torch-fused-dual"
    return {
        "ROLLOUT_QLORA": "1" if torch_path else "0",
        "VLLM_ROLLOUT_LORA_FUSE_PACKED": "1" if fused else "0",
        "VLLM_LORA_ENABLE_DUAL_STREAM": "1" if dual_stream else "0",
        "VLLM_DUAL_PRECISION_ROLLOUT": "0",
    }


@dataclass(frozen=True)
class Cell:
    batch_size: int
    context_len: int
    decode_tokens: int


def csv_ints(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def _build_run_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("run", help="benchmark one (backend, kernel) configuration")
    p.add_argument("--model", required=True, help="HF id or local checkpoint path")
    p.add_argument("--adapter", type=Path, required=True, help="LoRA adapter dir")
    p.add_argument("--output", type=Path, required=True, help="JSONL to append to")
    p.add_argument("--base-backend", choices=BASE_BACKENDS, required=True)
    p.add_argument("--lora-kernel", choices=LORA_KERNELS, required=True)
    p.add_argument("--hf-config-path", default=None)
    p.add_argument("--label", default=None, help="free-form label stored in the row")
    p.add_argument("--cells", default="1x512", help="batch x context pairs")
    p.add_argument("--decode-tokens", default="64")
    p.add_argument("--warmups", type=int, default=2)
    p.add_argument("--repetitions", type=int, default=5)
    p.add_argument("--max-lora-rank", type=int, default=16)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument(
        "--no-synchronized-prefill-barrier",
        dest="synchronized_prefill_barrier",
        action="store_false",
        help="Measure from each request's own first token instead of the barrier.",
    )


def _run(args: argparse.Namespace) -> None:
    for key, value in kernel_env(args.lora_kernel).items():
        os.environ[key] = value
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    import torch

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from vllm.lora.request import LoRARequest
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.request import RequestStatus

    # The engine core runs in this process (VLLM_ENABLE_V1_MULTIPROCESSING=0)
    # so the scheduler subclass can report barrier events through this list.
    sync_events: list[dict[str, Any]] = []

    class SynchronizedPrefillScheduler(Scheduler):
        """Hold early requests after their first token until the batch is ready.

        With chunked prefill, requests that finish prefill early start decoding
        while later requests are still prefilling. Making them look caught up
        (zero scheduled tokens) keeps their KV blocks resident without decode
        progress; once every request has one token, normal batched decoding
        resumes and the measured decode phase is the same for every kernel.
        """

        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, **kw)
            self._sync_batch_ids: set[str] = set()
            self._sync_released = False

        def _begin_batch_if_needed(self) -> None:
            active_ids = set(self.requests)
            if not active_ids:
                return
            if not self._sync_batch_ids or self._sync_batch_ids.isdisjoint(active_ids):
                self._sync_batch_ids = active_ids
                self._sync_released = False
            elif not self._sync_released:
                self._sync_batch_ids.update(active_ids)

        def schedule(self):  # type: ignore[no-untyped-def]
            self._begin_batch_if_needed()
            if not self._sync_batch_ids or self._sync_released:
                return super().schedule()

            batch = [
                self.requests[rid]
                for rid in self._sync_batch_ids
                if rid in self.requests
            ]
            if batch and all(req.num_output_tokens >= 1 for req in batch):
                self._sync_released = True
                sync_events.append(
                    {
                        "event": "release",
                        "batch_size": len(batch),
                        "timestamp": time.monotonic(),
                    }
                )
                return super().schedule()

            held = [
                req
                for req in self.running
                if req.request_id in self._sync_batch_ids and req.num_output_tokens >= 1
            ]
            original = {req.request_id: req.num_computed_tokens for req in held}
            try:
                for req in held:
                    req.num_computed_tokens = max(
                        req.num_computed_tokens,
                        req.num_tokens_with_spec + req.num_output_placeholders,
                    )
                output = super().schedule()
            finally:
                for req in held:
                    if req.request_id in self.requests:
                        req.num_computed_tokens = original[req.request_id]

            if output.total_num_scheduled_tokens == 0 and held:
                sync_events.append(
                    {
                        "event": "oom",
                        "reason": "synchronized batch exceeds KV-cache capacity",
                        "timestamp": time.monotonic(),
                    }
                )
                aborted = set(self.requests).intersection(self._sync_batch_ids)
                self.finish_requests(aborted, RequestStatus.FINISHED_ABORTED)
                output.finished_req_ids.update(aborted)
                self._sync_released = True
            return output

    def make_prompts(tokenizer: Any, cell: Cell) -> list[TokensPrompt]:
        vocab_size = int(tokenizer.vocab_size)
        bos = tokenizer.bos_token_id
        safe_start = 1000 if vocab_size > 2000 else 1
        usable = max(1, vocab_size - safe_start - 1)
        prompts = []
        for r in range(cell.batch_size):
            toks = [
                safe_start + ((pos * 131 + r * 977) % usable)
                for pos in range(cell.context_len)
            ]
            if bos is not None:
                toks[0] = int(bos)
            prompts.append(TokensPrompt(prompt_token_ids=toks))
        return prompts

    def run_cell(llm: LLM, lora_request: LoRARequest, cell: Cell) -> dict[str, Any]:
        prompts = make_prompts(llm.get_tokenizer(), cell)
        params = SamplingParams(
            temperature=0.0,
            max_tokens=cell.decode_tokens,
            min_tokens=cell.decode_tokens,
            ignore_eos=True,
        )
        elapsed: list[float] = []
        decode_step_ms: list[float] = []
        generated_counts: list[int] = []
        for rep in range(args.warmups + args.repetitions):
            offset = len(sync_events)
            torch.cuda.synchronize()
            started = time.perf_counter()
            outputs = llm.generate(
                prompts, params, lora_request=lora_request, use_tqdm=False
            )
            torch.cuda.synchronize()
            duration = time.perf_counter() - started
            new_events = sync_events[offset:]
            if any(e["event"] == "oom" for e in new_events):
                return {**asdict(cell), "status": "oom", "oom": new_events[-1]}
            count = sum(len(o.outputs[0].token_ids) for o in outputs)
            expected = cell.batch_size * cell.decode_tokens
            if count != expected:
                raise RuntimeError(f"generated {count} tokens, expected {expected}")
            if args.synchronized_prefill_barrier:
                releases = [e for e in new_events if e["event"] == "release"]
                if len(releases) != 1:
                    raise RuntimeError(f"expected one barrier release: {new_events}")
            if rep < args.warmups:
                continue
            elapsed.append(duration)
            metrics = [o.metrics for o in outputs]
            if any(m is None for m in metrics):
                raise RuntimeError("request metrics unavailable")
            first = [m.first_token_ts for m in metrics]
            denom = max(1, cell.decode_tokens - 1)
            if args.synchronized_prefill_barrier:
                boundary = max(first)
                tpots = [(m.last_token_ts - boundary) * 1000.0 / denom for m in metrics]
            else:
                tpots = [
                    (m.last_token_ts - m.first_token_ts) * 1000.0 / denom
                    for m in metrics
                ]
            decode_step_ms.append(statistics.median(tpots))
            generated_counts.append(count)
        median_s = statistics.median(elapsed)
        return {
            **asdict(cell),
            "warmups": args.warmups,
            "repetitions": args.repetitions,
            "elapsed_s": elapsed,
            "median_elapsed_s": median_s,
            "p10_elapsed_s": percentile(elapsed, 0.10),
            "p90_elapsed_s": percentile(elapsed, 0.90),
            "tokens_per_s": cell.batch_size * cell.decode_tokens / median_s,
            "decode_step_ms": decode_step_ms,
            "tpot_ms": statistics.median(decode_step_ms),
            "tpot_spread_ms": max(decode_step_ms) - min(decode_step_ms),
            "generated_counts": generated_counts,
            "status": "ok",
            "synchronized_prefill_barrier": args.synchronized_prefill_barrier,
        }

    pairs = []
    for item in args.cells.split(","):
        b, c = item.lower().split("x", maxsplit=1)
        pairs.append((int(b), int(c)))
    cells = [Cell(b, c, d) for d in csv_ints(args.decode_tokens) for (b, c) in pairs]

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_num_seqs": max(b for b, _ in pairs),
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
        "language_model_only": True,
        "enable_lora": True,
        "max_loras": 1,
        "max_lora_rank": args.max_lora_rank,
        "enforce_eager": args.enforce_eager,
        "enable_prefix_caching": False,
        "disable_log_stats": False,
        "seed": 0,
    }
    if args.hf_config_path:
        llm_kwargs["hf_config_path"] = args.hf_config_path
    if args.base_backend == "bitsandbytes":
        llm_kwargs["quantization"] = "bitsandbytes"
        llm_kwargs["load_format"] = "bitsandbytes"
    if args.synchronized_prefill_barrier:
        llm_kwargs["scheduler_cls"] = SynchronizedPrefillScheduler
        # Synchronous scheduling makes the hold/release boundary exact.
        llm_kwargs["async_scheduling"] = False

    llm = LLM(**llm_kwargs)
    lora_request = LoRARequest("ablation_adapter", 1, str(args.adapter))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", encoding="utf-8") as handle:
        for cell in cells:
            row = run_cell(llm, lora_request, cell)
            row.update(
                {
                    "model": args.model,
                    "adapter": str(args.adapter),
                    "label": args.label,
                    "base_backend": args.base_backend,
                    "lora_kernel": args.lora_kernel,
                    "env": kernel_env(args.lora_kernel),
                    "gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "timestamp": time.time(),
                }
            )
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            print(json.dumps(row), flush=True)
            gc.collect()


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------


def load_rows(results_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") != "ok":
                continue
            rows[(row["base_backend"], row["lora_kernel"])] = row
    return rows


def load_archived(csv_path: Path | None) -> dict[tuple[str, str], float]:
    if csv_path is None or not csv_path.exists():
        return {}
    archived: dict[tuple[str, str], float] = {}
    with csv_path.open(encoding="utf-8") as handle:
        for rec in csv.DictReader(handle):
            archived[(rec["backend"], rec["stage"])] = float(rec["tpot_ms"])
    return archived


def render_table(
    rows: dict[tuple[str, str], dict[str, Any]],
    archived: dict[tuple[str, str], float],
) -> str:
    lines = [
        "| base | stage | TPOT ms (median) | spread ms | vs prev | vs punica | "
        "archived ms | archived vs punica |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for backend in BASE_BACKENDS:
        prev = None
        base_tpot = None
        arch_base = archived.get((backend, "punica"))
        for kernel in LORA_KERNELS:
            row = rows.get((backend, kernel))
            if row is None:
                continue
            tpot = float(row["tpot_ms"])
            steps = row.get("decode_step_ms", [])
            spread = (max(steps) - min(steps)) if steps else float("nan")
            if base_tpot is None:
                base_tpot = tpot
            vs_prev = "" if prev is None else f"{(prev - tpot) / prev * 100:+.1f}%"
            vs_base = f"{(base_tpot - tpot) / base_tpot * 100:+.1f}%"
            arch = archived.get((backend, kernel))
            arch_s = "" if arch is None else f"{arch:.2f}"
            arch_gain = (
                ""
                if arch is None or arch_base is None
                else f"{(arch_base - arch) / arch_base * 100:+.1f}%"
            )
            lines.append(
                f"| {backend} | {kernel} | {tpot:.2f} | {spread:.2f} | {vs_prev} | "
                f"{vs_base} | {arch_s} | {arch_gain} |"
            )
            prev = tpot
    return "\n".join(lines)


def _build_table_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("table", help="render a markdown table from JSONL rows")
    p.add_argument("--results", type=Path, required=True, help="dir of *.jsonl")
    p.add_argument(
        "--archived",
        type=Path,
        default=None,
        help="CSV with columns backend,stage,tpot_ms (archived reference numbers)",
    )
    p.add_argument("--output", type=Path, default=None, help="write markdown here")


def _table(args: argparse.Namespace) -> None:
    text = render_table(load_rows(args.results), load_archived(args.archived))
    if args.output is not None:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    _build_run_parser(sub)
    _build_table_parser(sub)
    args = parser.parse_args(argv)
    if args.command == "run":
        _run(args)
    else:
        _table(args)


if __name__ == "__main__":
    main(sys.argv[1:])

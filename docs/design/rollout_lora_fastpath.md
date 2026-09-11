# Rollout LoRA fast path (`ROLLOUT_QLORA`)

Component C1 of the rollout precision scheduler. Default off; with every knob
unset the LoRA code path is byte-for-byte vanilla.

## Purpose

RL rollout with a LoRA policy is a degenerate multi-LoRA workload: one adapter,
attached to every request, decoded at small batch sizes. vLLM's Punica
shrink/expand Triton kernels are built for the general case (many adapters,
per-token adapter ids, rank-sorted token groups) and pay for it three times
per linear layer per step:

1. two Triton kernels with per-token id lookups and an fp32 intermediate
   buffer instead of two plain GEMMs,
2. one shrink/expand pair per packed slice (q, k, v, or gate, up, or the four
   Qwen3.5 `in_proj` slices) instead of one pair per layer, and
3. `LoRAKernelMeta.prepare_tensors` on every scheduler step, which contains a
   `torch.all(...)` reduced to a host boolean (a 1-byte device-to-host copy
   that blocks the launch thread on stream order, observed at 15-20 ms per
   step in Nsight) plus a sort and a unique (`cub::DeviceRadixSort`,
   `cub::DeviceRunLengthEncode`). The copy prevents CUDA-graph launches from
   queueing ahead of the running step.

The fast path replaces all three for the single-adapter case and leaves every
other case on Punica.

## Mechanism

### Torch two-GEMM op (`vllm/lora/ops/torch_ops/rollout_lora_ops.py`)

`torch.ops.vllm.rollout_lora_matmul(x, A, B, output_size, scale)` computes
`((x.to(A.dtype) @ A.T) * scale) @ B.T`. The body is `torch.compile(dynamic=True)`
so token counts do not recompile; the custom-op wrapper keeps it opaque to
vLLM's piecewise inductor graph and lets it be captured in CUDA graphs. The
fake implementation returns `A.dtype` like the real one. The rank-16
intermediate stays in the LoRA dtype (bf16/fp16); Punica accumulates it in
fp32, so the two paths match to tolerance, not bitwise (tests use the
`tests/lora/test_layers.py` tolerances).

### Packed slices (`vllm/lora/layers/base_linear.py`)

When `ROLLOUT_QLORA=1`, `BaseLinearLayerWithLoRA.create_lora_weights` (and the
merged-column override) allocate two extra per-slot buffers next to the
vanilla `lora_a_stacked` / `lora_b_stacked` tuples:

```
rollout_lora_a_stacked[slot, 0] : (n_slices * R, input_size)   A_s stacked along rank
rollout_lora_b_stacked[slot, 0] : (sum(output_slices), n_slices * R)   block_diag(B_s)
```

`R = lora_config.max_lora_rank` is the rank *capacity* of the stacked buffers,
so slice `s` always occupies rank rows `[s*R, (s+1)*R)`; an adapter of lower
rank leaves zero padding that contributes nothing. `set_lora` / `reset_lora`
refresh the packed slot from the stacked buffers, so a merged layer (QKV,
gate-up, Qwen3.5 `in_proj_qkvz` with 4 unequal slices) is one GEMM pair.
`VLLM_ROLLOUT_LORA_FUSE_PACKED=0` keeps the per-slice GEMM pairs instead
(ablation stage 2); the packed buffers are still allocated. They are not
allocated on non-CUDA platforms or with `fully_sharded_loras=True`, where the
wrapper never takes the fast path.

### Slot selection and dispatch (`vllm/lora/punica_wrapper/punica_gpu.py`)

`update_metadata` runs `select_rollout_single_lora_index` once per step
(`max_loras` below is the configured slot count `lora_config.max_loras`; the
manager's `update_metadata` argument is `lora_slots + 1` and is not used):

| token id mapping                 | max_loras | result                          |
|----------------------------------|-----------|---------------------------------|
| empty (profile / dummy run)      | 1         | slot 0                          |
| empty                            | >1        | Punica                          |
| two or more distinct ids > 0     | any       | Punica                          |
| exactly one id > 0               | any       | its slot (Punica if not loaded) |
| only zeros                       | 1         | slot 0                          |
| only zeros                       | >1        | first loaded slot, else Punica  |

The all-zero rows keep CUDA-graph capture and replay in lockstep: capture
uses an all-zero dummy mapping for the `num_active_loras=0` variant, and with
`max_loras=1` (what verl configures) every batch picks slot 0, so every
captured graph contains the same kernels. The flip side is that a batch with
*no* LoRA requests would still have slot 0 applied; that never happens in RL
rollout, and it is why the path is opt-in.

`add_lora_linear` then dispatches to `_add_lora_linear_rollout` (fused or
per-slice), casting the op output to the activation dtype. Any Punica entry
point (`add_shrink`, `add_expand`, `add_lora_embedding`, the Punica branch of
`add_lora_linear`, `add_lora_logits`, `moe_lora_align_block_size`,
`add_lora_fused_moe`) first calls `_ensure_punica_metadata_prepared`, so the
metadata sync is paid lazily and only when an adapter targets `lm_head` /
`embed_tokens` / MoE experts or a batch carries several adapters. One warning
line is logged when the fast path first fires
(`Rollout QLoRA torch path active: fused_packed=..., lora_index=..., slices=...`)
and one when it first falls back, with the token-id set that caused it.

### Dual stream and the base-forward override

Vanilla `VLLM_LORA_ENABLE_DUAL_STREAM=1` runs the base GEMM on the current
stream and LoRA on an aux stream through the `lora_linear_async` custom op,
base first. With `ROLLOUT_QLORA=1` the order flips to LoRA first
(`_execute_lora_async(..., lora_first=True)` via `execute_in_parallel`), so
the two small LoRA GEMMs are queued before the large base GEMM starts.

`BaseLinearLayerWithLoRA.base_forward_override` /
`set_base_forward_override(fn)` is the hook for the dual-precision component
(C2): when installed, `apply()` computes the base output as `fn(x, bias)` and
applies LoRA synchronously on the same stream. **Design decision 1 (accepted
2026-09-11):** with an override installed the dual-stream op is not used. That
is how every archived dual-precision run behaved (the dual-precision branch
preceded the dual-stream branch in `apply()`), and the dual-stream gain below
was only ever measured with dual precision off. Making them compose is a
follow-up that needs re-measurement. `MergedColumnParallelLinearWithLoRA`
routes through the base `apply()` whenever the fast path or an override is
active (the vanilla `_mcp_apply` all-gather path is only needed for fully
sharded LoRA).

## Knobs

| Env var (vllm/envs.py)          | Default | Effect |
|---------------------------------|---------|--------|
| `ROLLOUT_QLORA`                 | 0       | Enable the fast path (packed buffers, torch dispatch, lazy metadata, LoRA-first stream order). |
| `VLLM_ROLLOUT_LORA_FUSE_PACKED` | 1       | Fused packed GEMM pair (1) or one pair per slice (0). Ablation only. |
| `VLLM_LORA_ENABLE_DUAL_STREAM`  | 0       | Vanilla knob; with `ROLLOUT_QLORA` the LoRA GEMMs launch first. |

Both new knobs are read once at layer / wrapper construction; nothing on the
forward path touches `os.environ`. Preconditions: `fully_sharded_loras=False`,
one adapter per batch (verl sets `max_loras=1`), TP=1 is the supported and
measured configuration.

## Contracts with neighbours

* C2 (dual precision) installs `set_base_forward_override` on each LoRA linear
  layer; C1 never imports `dual_precision` and holds no INT4 logic.
* verl forwards `ROLLOUT_QLORA`, `VLLM_LORA_ENABLE_DUAL_STREAM` and
  `VLLM_ROLLOUT_LORA_FUSE_PACKED` to the vLLM server actor from the
  `precision_scheduler` YAML block (C8); the env vars are the wire format.
* The log line `Rollout QLoRA torch path active: ...` is the runtime evidence
  that the fast path was live (headline runs show exactly one per worker and
  no fallback lines).

## Dropped from the experimental code, and why

| Item | Reason |
|------|--------|
| `_apply_base_forward_async` | Pre-custom-op duplicate of `_apply_async_impl`; unreachable for every LoRA target in Qwen3.5 (decision 1: the custom-op dual stream is the intended design). |
| `_mcp_apply` early return | Unreachable: `MergedColumnParallelLinearWithLoRA.apply` diverts first, and the sharded subclasses that still call `_mcp_apply` require `fully_sharded_loras=True`. |
| per-step `_rollout_lora_metadata_debug` f-string | Built every scheduler step for a warn-once message; the fallback diagnostics are now computed lazily in the fallback branch. |
| raw `os.getenv("VLLM_ROLLOUT_LORA_FUSE_PACKED")` per layer call | Registered in `envs.py`, read once at construction. |
| module-global `*_LOGGED` flags | Per-wrapper flags. |
| `dual_precision_base_linear` op, `register_dual_precision_lora_layer` | Moved to C2 behind `base_forward_override`. |
| `add_input` -> `add_inputs` rename in `_mcp_apply` | Harmless vanilla typo (kwarg swallowed by `**kwargs`); not carried as drive-by. |
| `vllm bench serve` sweeps, nsys drivers, 27B TP4 launcher | Superseded by the fixed-work bench; 27B and TP>1 are out of scope (decisions 7, 10). |

## Tests

`tests/lora/test_rollout_lora_fastpath.py` (39 tests, all run on one GPU):
op fake/real dtype, selection table (incl. the profile-run / capture dummy
mapping), lazy metadata with a mocked `prepare_tensors`, packed-buffer layout (incl. sub-rank and `None` slices),
fused and per-slice fast path vs the Punica reference for Column, Row,
Replicated, MergedColumn(2), QKV(1), MergedQKV(3) and the 4-slice
variable-slice layer in fp16 and bf16, mixed-batch fallback, zero-B adapter ==
base bit-exact, no host sync after warm-up under
`torch.cuda.set_sync_debug_mode("error")` (and the sync *is* raised on the
Punica path), kernel presence via `torch.profiler` (no `_lora_shrink_kernel` /
`_lora_expand_kernel`, GEMMs present), CUDA-graph replay == eager, the
override contract, and LoRA-first dual-stream ordering. The vanilla
`tests/lora/test_layers.py` linear/packed/replicated/variable-slice subset
passes unchanged with the flag unset (39 passed, 39 skipped for `stage=False`)
and with `ROLLOUT_QLORA=1` (39 passed, 39 skipped).

## Tooling

* `tools/rollout_lora/make_zero_lora.py`: rank-16 PEFT adapter from a
  checkpoint's safetensors index (Kaiming A seeded by module name, zero B by
  default so the model output is unchanged while every LoRA kernel runs).
* `tools/rollout_lora/kernel_ablation.py run`: fixed-work decode TPOT for one
  (base backend, LoRA kernel) cell behind a synchronized prefill barrier; one
  JSONL row per cell. `... table` renders the markdown table below.
* `tools/rollout_lora/run_kernel_ablation.sh`: runs the four stages per
  backend serially on one GPU; every path is an argument or env override.

## Measured numbers

### Archived (Qwen3.5-9B, TP1, B=1, context 512, 64 decode tokens, 2 warm-ups, 5 repetitions, median TPOT)

Source: `/data/huanchen/verl/evidence_bundle/kernel_backend_ablation_9b.csv`,
raw rows in
`/data/huanchen/vllm/.codex-reports/tail_w4_redo/qwen35_9b_kernel_ablation/results/*.jsonl`
(dirty vLLM 34e66a3, A100 80GB, dual precision off).

| base | punica | torch | torch-fused | torch-fused-dual | gain |
|---|---:|---:|---:|---:|---:|
| BF16 | 15.34 | 15.15 | 14.12 | 13.99 | 8.8% |
| Marlin INT4 (mssfj GPTQ) | 10.44 | 10.21 | 9.69 | 9.16 | 12.2% |
| BitsAndBytes | 17.38 | 13.51 | 12.62 | 11.68 | 32.8% |

### Clean branch rerun

Same cell (Qwen3.5-9B, TP1, B=1, context 512, 64 decode tokens, 2 warm-ups,
5 repetitions, synchronized prefill barrier, dual precision off), clean branch
at commit `8cf7b3f8e6` (bench rows for the first three BF16 cells were taken
at `e0bbe6a01a`, which differs only in logging), A100 80GB, GPU 2,
2026-09-11. Command:

```
tools/rollout_lora/run_kernel_ablation.sh \
  --model /data/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --int4-model /data/huggingface/hub/models--Intel--Qwen3.5-9B-int4-AutoRound/snapshots/29688b8959bebb6d019ddd8f174a5b4bfd670456 \
  --adapter <make_zero_lora.py output for Qwen3.5-9B> --backends bf16,marlin \
  --archived /data/huanchen/verl/evidence_bundle/kernel_backend_ablation_9b.csv --out <dir>
```

The Marlin column uses the Intel AutoRound INT4 checkpoint (the shadow model
of the headline runs; vanilla vLLM loads it with `MarlinLinearKernel`),
whereas the archived Marlin column used `mssfj/Qwen3.5-9B-GPTQ-INT4` with a
wrapped config; absolute values are therefore close but not identical, the
per-stage gains are what is compared. Spread is max-min of the five
repetition medians. Every fast-path row carries in-process evidence in its
JSONL (`rollout_single_lora_index=0`, `punica_metadata_prepared=False`,
no fallback) and the log line `Rollout QLoRA torch path active:
fused_packed=..., lora_index=0, tokens=8192`. The rows are committed under
`tools/rollout_lora/results/qwen35_9b_tp1_2026-09-11/`.

| base | stage | TPOT ms (median) | spread ms | vs prev | vs punica | archived ms | archived vs punica |
|---|---|---:|---:|---:|---:|---:|---:|
| bf16 | punica | 15.44 | 0.22 |  | +0.0% | 15.34 | +0.0% |
| bf16 | torch | 15.08 | 0.05 | +2.3% | +2.3% | 15.15 | +1.3% |
| bf16 | torch-fused | 14.12 | 0.04 | +6.4% | +8.6% | 14.12 | +7.9% |
| bf16 | torch-fused-dual | 13.78 | 0.22 | +2.4% | +10.7% | 13.99 | +8.8% |
| marlin | punica | 10.38 | 0.01 |  | +0.0% | 10.44 | +0.0% |
| marlin | torch | 10.23 | 0.02 | +1.5% | +1.5% | 10.21 | +2.2% |
| marlin | torch-fused | 9.33 | 0.02 | +8.8% | +10.1% | 9.69 | +7.1% |
| marlin | torch-fused-dual | 9.11 | 0.01 | +2.4% | +12.3% | 9.16 | +12.2% |

Pass criteria from the C1 card: every stage is no worse than the previous one
beyond the run's own spread (true for all six transitions; the smallest step,
BF16 torch-fused -> dual, is 0.34 ms against a 0.22 ms spread), and stage 4
beats stage 1 by at least half the archived gain (BF16 10.7% vs 8.8%
archived; Marlin 12.3% vs 12.2%). The BitsAndBytes column was not rerun
(time-boxed; not part of the project story).


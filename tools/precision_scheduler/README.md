# Decode TPOT heatmap for the rollout precision scheduler

`tpot_heatmap.py` measures steady-state decode time per output token (TPOT) for the BF16
base model and for its INT4 (W4) shadow over a batch-size x context-length grid. The
resulting `heatmap.json` is the `TPOT[p](ctx, live)` table that the verl policy builder
(`verl/experimental/precision_scheduler`) interpolates in log2 space when it prices
BF16-prefix / W4-suffix plans.

## Mechanism

* **One engine launch per precision row.** The base precision is selected through the
  decision-6 policy spec: the driver process spawns one child of itself per precision
  with `VLLM_DUAL_PRECISION_POLICY=uniform_w4` (plus the INT4 shadow checkpoint) for the
  INT4 row and the variable **unset** for the BF16 row, so the BF16 row is
  vanilla-equivalent vLLM. `precision_environment()` is the only place that knows the
  variable names. The older harness forced the precision per batch from a scheduler
  subclass fed by a stub `switch_thresholds` policy; that hack is gone.
* **Synthetic KV** (`synthetic_kv_connector.py`): a `KVConnectorBase_V1` loaded through
  `kv_connector_module_path`. Its scheduler side declares all but the last prompt token
  as externally computed; its worker side fills every allocated block (attention KV and
  recurrent/conv state alike, so hybrid models such as Qwen3.5 work) with non-zero
  constants. Long-context cells therefore never prefill. Toggle with
  `set_scheduler_enabled()`.
* **Decode barrier** (`sync_prefill.SynchronizedPrefillScheduler`): holds every request
  that already has its first token until all requests in the batch do, then releases
  the batch; `barrier_release_count` and `events` let the driver verify that every
  measured generation crossed exactly one barrier and that no request decoded ahead.
  A batch that cannot be resident at once is aborted (`oom` event) instead of hanging.
  The class is shared with the LoRA kernel-ablation bench under `tools/rollout_lora/`.
* **Timing.** Every synchronous `llm_engine.step()` after the barrier is bracketed by
  `torch.cuda.synchronize()`. `--warmup-steps` iterations are discarded inside the same
  live request (the prefill-to-decode transition is never amortised into TPOT),
  `--measurement-steps` are recorded, `--measurement-repetitions` times. Before the
  first cell an untimed `--initial-precision-warmup-steps` generation primes lazy
  first-request setup. Cells are scanned cheapest-KV first.
* **Outputs** in `--output-dir`: `cells.jsonl` (one fsync'd row per cell and precision,
  incl. every per-step TPOT), `heatmap.json` (`batch_sizes`, `seq_lens`, `bf16_tpot_ms`,
  `int4_tpot_ms`, `speedup_bf16_over_int4`, `required_kv_bytes`; missing cells are
  `null`, never interpolated), `manifest_<precision>.json`, `manifest.json`,
  `progress.json`, and (unless `--no-plots`) three PNGs from `heatmap_plots.py`.
* **Resume.** A rerun on the same directory skips completed cells. A run that died
  inside a cell is reported (exit 2) and left untouched unless
  `--retry-interrupted-cell`; complete (0) and capacity-exhausted (3) runs never start
  an engine.

## Knobs and defaults

| Argument | Default | Notes |
|---|---|---|
| `--model` | required | BF16 checkpoint |
| `--int4-model` | none | required for the `int4` row |
| `--adapter` | none | zero-delta LoRA (`tools/rollout_lora/make_zero_lora.py`); enables LoRA |
| `--batch-sizes`, `--seq-lens` | required | `1,2,4` or `start:stop:step` |
| `--precisions` | `bf16,int4` | subset |
| `--warmup-steps` / `--measurement-steps` | 2 / 9 | per repetition |
| `--measurement-repetitions` | 1 | the archived Phi/Gemma rep5 runs used 5, Qwen3.5-4B used 1 |
| `--initial-precision-warmup-steps` | 9 | untimed priming |
| `--gpu-memory-utilization` | 0.5 | archived runs used 0.5 |
| `--max-model-len` | `max(seq) + warmup + measure + 2` | |
| `--full-cudagraph-without-torch-compile` | off | Phi-4-mini and Gemma lanes set it; Qwen3.5 did not |
| `--synthetic-float-value` / `--synthetic-integer-value` | 0.015 / 1 | must be non-zero |
| `--no-language-model-only` | off | multimodal checkpoints load the text tower only by default |

Per-model argument sets that produced the archived heatmaps (provenance:
`/data/huanchen/verl/.codex-report/new-storyline-experiments/eos_hazard_extensibility/b32_16k_sensitivity/ema30_alpha020/heatmaps/*/manifest.json`):
batch `1,2,4,8,16,32`, seq `1024,2048,3072,4096,5120,6144,7168,8192,12288,16384`,
warmup 2, measure 9, initial warmup 9, gmem 0.5; Phi-4-mini / Gemma-E2B / Gemma-E4B with
`--measurement-repetitions 5 --full-cudagraph-without-torch-compile` (Gemma additionally
`VLLM_DUAL_PRECISION_INT4_MODULES=mlp_only`); Qwen3.5-4B with repetitions 1 and no
full-cudagraph flag.

## Validity guard

Heatmaps dated 2026-09-09 were produced before the precision-selector fix: both rows
ran BF16 and the median speedup is exactly 1.000. `manifest.json` now records
`median_speedup_bf16_over_int4`, and the verl side refuses a heatmap whose median is
within 2% of 1.0 (`tpot_grid.validate_heatmap`, override with `--skip-heatmap-guard`).

## Measured numbers

BF16-only 2x2 smoke on Qwen3.5-4B, A100-80GB, GPU 6, 2026-09-11 (evidence in
`fixtures/smoke_qwen35_4b_bf16_2x2.json`): batch 1 / 8 x context 1024 / 4096, warmup 1,
measure 3, repetitions 2: TPOT 9.20 / 8.65 / 9.84 / 10.39 ms; barrier released once per
repetition; first-token spread 0 ms; 1023 / 4095 synthetic KV tokens per request; a
second invocation starts no engine. Engine init 132 s cold (16 s with a warm compile
cache), 4 cells in 1.2 s.

Archived valid W4 heatmaps (post-fix, `*_rep5_20260910` and the Qwen3.5-4B
`corrected_selector` run): median BF16/INT4 speedup 1.20 (Phi-4-mini), and the Qwen3.5-9B
Gen1 grid shows 1.53 at batch 1 x 512 context.

## Tests

```bash
python -m pytest -p no:cacheprovider -q tools/precision_scheduler          # 14 CPU tests
run_gpu.sh --gpus 6 --timeout 1500 -- python -m pytest -m gpu_smoke tools/precision_scheduler
```

The INT4 row needs the dual-precision runtime (components C2/C4) and is therefore not
part of the smoke; the `uniform_w4` spec is the contract it will honour.

## Dropped from the report-dir harness and why

* stub `benchmark_dynamic_policy.json` with `switch_thresholds` and the
  `_dual_precision_base_precision` scheduler override: replaced by one engine per
  precision through the policy spec (decision 4 / 6).
* `model_specs.json` registry with absolute paths: every path is a CLI argument.
* the Gen1 `fixed_token_bench.py` (real-prefill, `LLM.generate` wall-clock TPOT): superseded;
  only its `SynchronizedPrefillScheduler` survives here and its kernel-ablation mode
  moves to `tools/rollout_lora/` (C1). The legacy CSV grid it produced can still be
  regenerated with `verl ... cli grid --kind gen1-csv`.
* per-model shell launchers and the hard-coded plot scripts.

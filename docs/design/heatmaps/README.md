# TPOT heatmaps (A100-80GB, gmem 0.5, LoRA rank 16, all layers INT4)

Produced by `tools/precision_scheduler/tpot_heatmap.py` (batch 1,2,4,8,16,32,64 x context
512..16384, 2 warm-up + 9 measured decode steps x 3 repetitions per cell, synthetic KV,
synchronized-prefill barrier, one child engine per precision). These are the grids the C6
policy builder reads (`--heatmap`).

| file | backend | LoRA path | INT4 gain over BF16, median per batch (1 / 8 / 32 / 64) | wall time |
|---|---|---|---|---|
| `qwen35_9b_ours_sync_2026-09-15.json` | ours (fused torch GEMM), dual precision attached | synchronous (dual stream silently disabled by the base override, decision 1) | 44.9 % / 34 % / 22.2 % / 3.7 % | ~45 min (harness hang on the first non-resident cell, reaped by the launcher timeout) |
| `qwen35_9b_ours_dualstream_2026-09-16.json` | ours + LoRA-first dual stream (commit 7fb5ccaa6c) | async, LoRA GEMMs on the aux stream | 47.6 % / 38.0 % / 27.5 % / 6.6 % | **~8 min** (`--kv-capacity-fraction 0.8`, commit 8b74b7efb5; engine init 40 s BF16 + 194 s INT4, 143 cells at ~1.2 s each) |

Both precisions gain the same ~0.6 ms per decode step from the dual stream; because the INT4
step is shorter, the INT4/BF16 ratio widens by 3-6 points at batch 2-32
(`qwen35_9b_ours_dualstream_2026-09-16.png`, right panel). Non-resident cells (batch 32 at 16k,
batch 64 at 8k+; INT4 also 32 x 12k and 64 x 6-7k, since the shadow takes weight memory) are
`null` and the policy builder extrapolates them from the nearest resident cell.

Every scheduler run before 2026-09-17 used the sync heatmap and the sync LoRA path; runs from
the dual-stream heatmap onwards should cite this file.

## All three models on the dual-stream backend (2026-09-20)

`qwen35_4b_ours_dualstream_2026-09-20.json` and `phi4_mini_ours_dualstream_2026-09-20.json` were
re-recorded with the same harness and backend (~8 min each); `speedup_3models_dualstream_2026-09-20.png`
shows the three speedup grids side by side. Median INT4 gain per batch size:

| model | b1 | b2 | b4 | b8 | b16 | b32 | b64 | BF16 -> INT4 TPOT at b1 / 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.5-9B | 48 % | 39 % | 39 % | 38 % | 31 % | 28 % | 7 % | 13.7 -> 9.4 ms |
| Qwen3.5-4B | 29 % | 25 % | 25 % | 24 % | 16 % | 13 % | 2 % | 9.8 -> 7.6 ms |
| Phi-4-mini-reasoning | 19 % | 15 % | 16 % | 15 % | 12 % | 7 % | 6 % | 9.2 -> 7.2 ms |

The Qwen grids are flat across context (weight-bandwidth bound); Phi's gain decays with context
(29 % -> 10 % at batch 1) as its attention dominates, so the late-tail switch buys less there.

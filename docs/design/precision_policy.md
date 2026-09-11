# Precision switching policy (torch-free)

Component C5 of the rollout precision scheduler. Module:
`vllm/v1/core/sched/precision_policy.py`. Tools: `tools/precision_policy/`.
Tests: `tests/v1/core/test_precision_policy.py`.

## Purpose

During an RL rollout the longest responses decode alone for most of the wall
clock. The dual-precision runtime (C2/C3) can flip the base weights of a
rollout from BF16 to INT4 once, and the scheduler (C4) decides *when*. This
module is the decision logic, kept free of torch so that the scheduler, the
GPU worker (which reads `capture_max_batch`) and the CPU-only policy toolkit
(which writes policy files, C6) all share one definition:

- the policy JSON contract (schema 2 through 6) and its validator,
- the dense lookup table `(response frontier, prompt bucket, live batch) ->
  committed switch frontier`,
- the commitment rules (`monotone`, `receding`) and the live-batch guard,
- the receding-horizon cost model that defines the calibrated tables, kept
  as a pure offline function and golden-tested against archived runs,
- revision-checked reload for online calibration.

## Mechanism

### One flag, four kinds (decision 6)

`PolicySpec.parse(spec)` turns the single policy flag into one of:

| spec | kind | runtime representation |
|---|---|---|
| `""` | disabled | no policy object |
| `fixed_frontier:<K>` | `fixed_frontier` | degenerate table, every cell `K`, monotone, guard = capture ceiling |
| `fixed_threshold:<t>` | `fixed_threshold` | degenerate table, every cell = first frontier (250), monotone, `max_switch_live_batch = t` |
| `uniform_w4` | `uniform_w4` | no table; the decider starts switched (INT4 from the first token) |
| `<path>.json` | `lookup` (or `fixed_frontier` via `fixed_switch_frontier`, or `cost_model` for schema 2/3) | the file's table |

`load_precision_policy(spec)` returns a `PrecisionPolicy` for every kind, so
C4 has one switcher and one cohort log format. `fixed_threshold` is expressed
exactly as the full-RL matrix emulated it (`fixed_t{2,4,8}.json`: constant
250 table plus `max_switch_live_batch`); the first frontier must be observed
before a drain can switch, so a drain inside the first 250 response tokens
switches at the 250 crossing.

### Lookup table

`LookupTable.committed_frontier(frontier_tokens, prompt_tokens, live)` is
constant time and reproduces the experimental `_lookup_table_frontier`:

- frontier index `(f - frontier_start) // frontier_step`, **not clamped**:
  outside the table -> `None`;
- prompt index `round((p - prompt_bucket_start) / prompt_bucket_step)`
  (Python `round`, half to even), clamped to `[0, count)`;
- live index `live - live_batch_start`, clamped;
- flat layout `frontier_major,prompt_bucket,live_batch` (validated);
- a stored `0` means "no planned switch" and is returned as `None`.

### Decider

`PolicyDecider(policy).observe(frontier_tokens, prompt_tokens_median,
decision_live, actual_live, max_response_tokens) -> Decision` is called by
the scheduler on every `schedule()` once the rollout cohort is armed (calls
between scan-grid crossings are O(1)). `frontier_tokens` is quantized to the
scan grid; each grid frontier is observed once (`last_frontier` gate).

```text
observe(f, prompt, decision_live, actual_live, max_resp):
  if switched: return already_switched
  if monotone and committed is not None and max_resp >= committed
     and actual_live <= switch_live_cap:            # evaluated every call
      switched = True; return switch
  f = floor(f / scan) * scan
  if f <= last_frontier: return not_observed | guard_blocked
  last_frontier = f
  candidate = table[f, bucket(prompt), decision_live]
  receding:  committed = candidate                  # None clears
  monotone:  committed = min(committed, candidate)  # None ignored
  if committed is None: return no_commitment
  if max_resp >= committed and actual_live <= switch_live_cap:
      switched = True; return switch
  return guard_blocked | committed
```

Two inputs are deliberately different: the table is indexed with
`decision_live` (live requests plus not-yet-arrived cohort members, so
asynchronous admission cannot make a B64 rollout look like B8 at the first
observation) while the guard uses `actual_live`.

The receding switch is **observation-gated**: it is evaluated only at a new
grid frontier against the freshly looked-up candidate, and a `None` candidate
clears the commitment. This is what the archived receding runs did (the
handoff document's per-tick pseudocode does not match the code), and the
1313-line receding golden depends on it. The monotone switch is evaluated on
every call so that a guard-blocked switch fires as soon as the batch drains.

`Decision` carries `committed_frontier`, `switch_now` (True exactly on the
flipping call), `reason` (`switch`, `committed`, `guard_blocked`,
`no_commitment`, `not_observed`, `already_switched`) and, for observations,
`frontier`, `candidate_frontier`, `previous_frontier`. C4 derives the base
precision from `decider.switched` / `decider.base_precision` (uniform_w4 is
switched from construction and never reports `switch_now`).

### Live-batch guard and capture ceiling

`PrecisionPolicy.switch_live_cap` is `max_switch_live_batch` when set, else
`capture_max_batch`. The loader rejects `max_switch_live_batch >
capture_max_batch`: switching above the INT4 CUDA-graph ceiling ran eager
INT4 steps in the hardmath run (switch at live 56, ceiling 32). No archived
switch happened above live 23, so defaulting the guard to the ceiling does
not change any golden.

### Cost model (offline only)

`CostModel` keeps the receding-horizon full-RL-step cost model as a pure
function with instance-level memoization (the experimental module-level
caches were keyed by `id(policy)` and alias after a reload). It is never
evaluated on the scheduler thread: one prediction costs 0.2-0.5 s in pure
Python.

`predict_receding_horizon(frontier, live, median_prompt)` reproduces the
archived online predictor: conditional archived response lengths
(`> frontier`), survival integrated in `integration_chunk_tokens` chunks up
to `response_cap`, log2-interpolated TPOT grid with non-finite cells skipped
per batch row, `E[TPOT | non-empty] * P(non-empty)` for the binomial number
of survivors, speed scales, tail-correction anchors, switch overhead, a
downstream `seconds_per_token` term; futures whose surviving batch exceeds
`capture_max_batch` are skipped; candidates are compared with strict `<`
from the current frontier upward so ties keep "now"; `switch_now` requires
the best plan to be the current frontier with gain strictly above
`required_gain_seconds`.

`predict(precision, fi, pj, live, alive)` and `plan_switch(...)` expose the
hazard-table builder's per-bin cost and global search (decision 2, the
global-search builder) with the same tie rule, so C6 can equivalence-test its
vectorized builder against the scalar definition:

```text
Inputs
  F[i]        frontier bins: 250, 500, ..., CAP-250 (STEP = 250 response tokens)
  P[j]        prompt buckets: 0, 128, ..., 2048
  B           initial rollout batch; live-batch axis is 1..B
  TPOT[p](ctx, live)   profiler heatmap, log2-interpolated in context and live batch, p in {bf16, w4}
  slope       downstream seconds per sampled token, fitted from RL step timings
  hazard tables H_bf16, H_w4: per bin, risk[i] = P(eligible and alive at bin start),
              event[i] = P(finish inside bin i, not by cap); h_i = event[i] / risk[i]

Online EMA (once per RL step, from that step's switch cohort: entry token and final length per request)
  new = components(entries, finals)                 # risk/event per bin, only bins with eligible requests
  H[k] = (1 - alpha) * H[k] + alpha * new[k]         # k in {risk, event}, only on observed bins
  rebuild table below; bump policy_revision; scheduler reloads before next rollout

survival(H, f)   # P(a request alive at frontier f is still alive at the START of each later bin)
  S[0] = 1;  S[k] = prod_{j < k} (1 - h_{f+j})

cost(p, fi, pj, live, alive[0..n))   # expected cost of decoding bins fi.. under precision p
  for each bin k:
    nonempty_k = 1 - (1 - alive_k)^live           # P(at least one of `live` requests still running)
    eff_k      = clip(round(live * alive_k / nonempty_k), 1, B)   # E[live | nonempty]
    tokens_k   = min(STEP, CAP - F[fi+k])
  rollout    = sum_k TPOT[p](P[pj] + F[fi+k], eff_k) * tokens_k / 1000 * nonempty_k
  downstream = slope * live * sum_k alive_k * tokens_k
  return rollout + downstream

build_policy   # dense table, every (frontier, prompt bucket, live) state
  for each observed frontier fi:
    S_bf = survival(H_bf16, F[fi])
    stay = cost(bf16, fi, ., ., S_bf)                            # never switch
    best = +inf
    for each candidate switch frontier fj >= fi:                 # global search over all later frontiers
      prefix = cost(bf16, fi, ., ., S_bf[0 : fj-fi])             # BF16 until fj
      reach  = S_bf[fj-fi]                                       # P(still alive when fj is reached)
      suffix = cost(w4, fj, ., ., survival(H_w4, F[fj]) * reach)  # W4 from fj on, conditioned on reaching it
      cand   = prefix + suffix
      if cand < best: best, best_fj = cand, fj                   # strict <: earliest frontier wins ties
    table[fi, pj, live] = F[best_fj] if best < stay else 0       # 0 = do not plan a switch

runtime (C4, receding mode; runs when the longest live request crosses a new 250-token frontier)
  state     = (f, decision_live = live + not-yet-arrived cohort members, prompt bucket of median prompt)
  committed = table[f, bucket, decision_live]                    # None if 0
  switch to INT4 (one-way, for the rest of the rollout) when
      committed is not None and max_response_tokens >= committed
      and actual_live <= max_switch_live_batch
```

Two properties are preserved: the comparison is plan against plan (BF16
prefix plus W4 suffix, downstream token cost included, against staying BF16),
not "is W4 faster right now"; and ties resolve to the earliest frontier
because the update uses strict less-than while iterating upward.

### Reload with revision check

`PolicyStore(spec).load()` reads the policy once; `reload()` re-reads the
file between rollouts. The installed policy is kept and `PolicyRevisionError`
is raised when the file is unreadable or invalid or its
`calibration.policy_revision` went backwards; with `require_advance=True` an
unchanged revision also raises (fail closed against a stale table). A
reload that advances the revision installs the new policy; inline specs
reload to themselves. `reload_count` / `last_reload_advanced` let C4 log
"Reloaded policy before rollout N: revision=R" once per boundary.

## Policy JSON schema

Top-level keys (schema 6 is what C6 writes; 2-5 are the archived ancestors):

| key | type | default | meaning |
|---|---|---|---|
| `schema_version` | int, required | - | one of 2, 3, 4, 5, 6 |
| `description` | str | `""` | free text |
| `scan_interval_tokens` | int > 0 | 250 | frontier grid; must equal `lookup_table.frontier_step` |
| `arm_min_requests` | int > 0 | 1 | cohort-free arming threshold (used only when `initial_rollout_batch` is absent) |
| `capture_max_batch` | int > 0 | 32 | INT4 CUDA-graph capture ceiling (read by C3); skips futures in the cost model |
| `commitment_enabled` | bool | false | requires a positive `initial_rollout_batch` |
| `receding_horizon_lookup` | bool | false | `true` = receding mode, `false` = monotone |
| `initial_rollout_batch` | int > 0 or null | null | rollout cohort size; pads `decision_live` |
| `max_switch_live_batch` | int > 0 or null | null | live-batch guard; must not exceed `capture_max_batch`; null = ceiling |
| `fixed_switch_frontier` | int > 0 | absent | profiler-forced switch at this frontier (degenerate table, cohort-free arming); exclusive with `lookup_table` |
| `calibration` | object | `{}` | provenance; only `policy_revision` (int, default 0) is read |
| `offline_cost_model` | object | absent | provenance (`response_cap`, `downstream_seconds_per_token`, `switch_overhead_seconds`, ...); `response_cap` sizes a `fixed_switch_frontier` table |
| `lookup_table` | object | absent | dense table, below |
| `cost_model` | object | absent | schema 2/3 offline cost model (`response_cap`, `downstream_seconds_per_token`, `tpot_batches`, `tpot_contexts`, `bf16_tpot_ms`, `w4_tpot_ms`, `bf16_lengths`, `w4_lengths`, `*_speed_scale`, `integration_chunk_tokens`=64, `min_conditional_samples`=5, `switch_overhead_seconds`=0, `required_gain_seconds`=0, `tail_correction_anchors`=[], `prompt_bucket_tokens`=128) |
| `switch_thresholds`, `lookup_hierarchy` | - | - | **rejected** (decision 4) |

`lookup_table`:

| key | meaning |
|---|---|
| `layout` | must be `frontier_major,prompt_bucket,live_batch` |
| `frontier_start`, `frontier_step`, `frontier_count` | response-token axis (headline tables: 250, 250, 65 or 98) |
| `prompt_bucket_start`, `prompt_bucket_step`, `prompt_bucket_count` | prompt axis (0, 128, 17) |
| `live_batch_start`, `live_batch_count` | live axis (1, B) |
| `committed_frontiers` | flat list of `frontier_count * prompt_bucket_count * live_batch_count` ints; 0 = no planned switch |

A policy must carry one of `lookup_table`, `cost_model` or
`fixed_switch_frontier`. A `cost_model`-only file (schema 2/3) is an offline
oracle: `PolicyDecider` refuses it.

## Knobs and defaults

This module reads no environment variables. The flag that carries the spec
(`VLLM_DUAL_PRECISION_POLICY`, decision 6) and the reload / observation
knobs are declared in `vllm/envs.py` by C4 and translated from the verl YAML
block by C9. Module constants: `DEFAULT_SCAN_INTERVAL_TOKENS = 250`,
`DEFAULT_CAPTURE_MAX_BATCH = 32`, `DEFAULT_PROMPT_BUCKET_TOKENS = 128`,
`DEFAULT_SPEC_RESPONSE_CAP = 2^20` (frontier span of inline-spec tables),
`UNBOUNDED_CAPTURE_MAX_BATCH = 2^30` (`uniform_w4`: no INT4 ceiling).

## Contracts with neighbors

- **C4 scheduler** owns cohort arming, the rollout index, the cumulative
  response watermark, the median prompt, decision-live padding, the
  switch-cohort JSONL and the reload hook; it calls `observe` every
  `schedule()` after arming, resets the decider at rollout boundaries, and
  publishes `decider.base_precision`. It must call `observe` every tick (not
  only at crossings) for the monotone guard-blocked switch to fire on drain.
- **C3 dispatcher** reads `policy.capture_max_batch` to bound INT4 graph
  capture; `switch_live_cap <= capture_max_batch` is guaranteed by the
  loader.
- **C6 toolkit** writes schema-6 files, bumps `calibration.policy_revision`
  atomically, and can validate its vectorized builder against
  `CostModel.predict` / `plan_switch`. `PrecisionPolicy.to_json()` emits a
  schema-6 file for any kind (used to materialize the degenerate baselines).
- **Tooling** may load the module from its file path
  (`tools/precision_policy/replay_policy_log.py::load_policy_module`) when
  the `vllm` package or torch is unavailable.

## Dropped from the experimental code, and why

| item | reason |
|---|---|
| `switch_thresholds` table mode and its scheduler loop | one superseded run; the profiler's forced 1K switch becomes `fixed_switch_frontier` |
| `lookup_hierarchy` (base + earlier-only refinement) | two policies built, replayed offline, never launched (no `lookup_level=refine_` line in 796 logs) |
| async `ProcessPoolExecutor` predictor, `VLLM_DUAL_PRECISION_ASYNC_*`, `PROFILE_NOOP_PREDICTION` | profiling runs only; the dense table already removes prediction work from the scheduler thread |
| in-scheduler online cost loop (`Dynamic cost prediction:` sync path) | superseded by lookup tables in every completed run; the math survives as `CostModel` |
| `commitment_enabled` + `cost_model` without a table | never launched |
| `functools.cache` + `cache_clear` reload | replaced by `PolicyStore` with an explicit revision check |
| module-level caches keyed by `id(policy)` | replaced by instance memoization |
| "commitment armed when peak batch >= initial_rollout_batch" gate | dead in both arming modes (cohort arming pre-sets the peak to the cohort size); C4 owns arming |

## Measured numbers and provenance

All goldens are CPU replays of archived runs under
`/data/huanchen/verl/.codex-report/new-storyline-experiments/` (absolute
paths; the tests skip when absent and always run on the trimmed fixtures).

| golden | oracle | result |
|---|---|---|
| static lookup replay | `dynamic_tail8k_heatmap_20260823/runs/{b64_cap16384,b32_cap24576,b128_cap16384}_tail8k_dynamic/logs/*.log` with `policies/*_tail8k_lookup250.json` (schema 4) | 53+49+54 = 156/156 commitment lines, 14+13+15 = 42/42 switches, 15 rollouts each |
| receding replay | `runs/{b64_cap16384,b64_cap24576,b128_cap16384,b128_cap24576}_online_hazard_warm5_receding_gpuval/logs/*.log` with `policies/*_online_hazard_warm5_receding_lookup250.json` (schema 5) | 291+389+375+258 = 1313/1313 update lines, 60/60 switches |
| logged sync predictions | `dynamic_full_cost_b64_20260823/runs/b64_cap24576_online_cost_dynamic/logs/*.log` (102 `Dynamic cost prediction:` lines) with `policies/b64_cap24576_online_full_cost.json` (schema 2) | every field to 1e-5 (tokens 1e-3); 16 lines at prompt bucket 0, 86 at bucket 128 (the line does not log the median prompt) |
| validation states | `dynamic_switch_rollout_20260822/policies/runtime_full_cost_lookup_committed_250_validation.json` (87 rows) with `runtime_full_cost_receding_1k.json` at scan 250 and `runtime_full_cost_lookup_committed_250.json` | 87/87 `direct == predict_receding_horizon` gated on `required_gain_seconds` (7 rows plan a sub-margin switch and are recorded as `None`), 87/87 `lookup == table` |
| commitment replay | `dynamic_full_cost_rollout_20260822/analysis/commitment_policy_replay.json` with `dynamic_switch_rollout_20260822/runs/bf16/traces/request_lifetimes_replica000_node000.jsonl` (43 MB, archive only) | grid 250 step 6 by default (switch 7500 @ live 15, ~12 s); all 30 grid/step rows with `PRECISION_POLICY_FULL_GOLDEN=1` |

Not a golden: `full_rl_policy_matrix_b64_cap24k_20260902/runs/ema` replays
19/23 update lines against the on-disk `ema.json` because the file is EMA
revision 1 while the logged rollout ran revision 0 (two lookups differ at
frontier 1750/live 28 and 5250/live 14); its `fixed_t{2,4,8}` lanes never
switched in the single completed rollout, so no completed run exercises a
binding guard. The guard is covered by unit tests only.

Fixture provenance: `tests/v1/core/fixtures/precision_policy/` holds the
policies above trimmed to the first 44 frontiers x 3 prompt buckets
(`tools/precision_policy/trim_policy.py`; every logged prompt is below 192
tokens and every logged frontier below 11000, so indexing is unchanged) and
the policy lines of each log (`replay_policy_log.py extract`), plus the two
schema-2 cost models, the 87-row validation file and the archived
commitment-replay summary verbatim. Schema fixtures (`schema/`) are 4-frontier
x 2-bucket trims of `fixed_t8.json`, `b32_cap16384_ema_pair128_a010_30step.json`,
`b64_cap16384_fixed_frontier8000_30step.json`,
`b128_cap24576_forced_tail8k_calibration.json` (schema 4 without
`calibration`), `b64_cap16384_tail8k_lookup250.json` and
`b64_cap16384_online_hazard_warm5_receding_lookup250.json`.

## Auditing a run

```bash
python tools/precision_policy/replay_policy_log.py --policy policy.json --log run.log
# rollouts=15 updates=291/291 switches=15/15 mismatches=0   (exit 0)
python tools/precision_policy/replay_policy_log.py trace --policy cost.json \
    --trace request_lifetimes.jsonl --batch 128 --steps 6-15 --action-grid 250 \
    --expect commitment_policy_replay.json
```

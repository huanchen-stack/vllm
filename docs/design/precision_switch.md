# Scheduler switching runtime

Component C4 of the rollout precision scheduler. Module:
`vllm/v1/core/sched/precision_switch.py`; scheduler integration in
`vllm/v1/core/sched/scheduler.py`; contract fields in
`vllm/v1/core/sched/output.py` and `vllm/v1/request.py`; knobs in
`vllm/envs.py`. Tests: `tests/v1/core/test_precision_switch.py` (CPU),
`tests/v1/core/test_precision_switch_gpu.py` (gpu-smoke), fixtures under
`tests/v1/core/golden/precision_switch/`. Tooling:
`tools/precision_policy/extract_switch_golden.py` (vLLM) and
`tools/precision_scheduling/policies/build_static_policy.py` (verl).

## Purpose

An RL rollout decodes B responses; the longest ones run alone for most of
the wall clock. The dual-precision runtime (C2 residency, C3 dispatch) can
flip the base weights of the *whole rollout* from BF16 to INT4 once. The
policy module (C5) decides, from a dense table, *at which response frontier*
that flip should happen for a given live batch and prompt length. This
component is the scheduler-side state machine around that decision: it knows
which requests form a rollout, how far the longest response has progressed,
how many requests are live, publishes one precision string per `schedule()`
in `SchedulerOutput.dual_precision_base_precision`, writes the switch-cohort
record the online calibrator (C6) learns from, and reloads the policy at
rollout boundaries.

The scheduler owns the precision because the decision needs response-length
history and a one-way latch; a worker that reconstructs it from the request
count of the current forward would flip back and forth as prefills and
finishes interleave.

## Mechanism

### One switcher, four policy kinds (decision 6)

`VLLM_DUAL_PRECISION_POLICY` selects the policy through `PolicyStore`
(C5): `fixed_threshold:<t>`, `fixed_frontier:<K>`, `uniform_w4`, or a path
to a policy JSON. All kinds run through the same `RolloutPrecisionSwitcher`,
the same `PolicyDecider.observe` call and the same cohort JSONL. The
experimental tree had a separate env-threshold latch (`_dual_precision_base_precision`)
that never wrote cohorts and a lookup runtime for JSON policies; both are
now the one path below. The old `VLLM_DUAL_PRECISION_THRESHOLD` /
`VLLM_DUAL_PRECISION_DYNAMIC_POLICY` variables do not exist on this branch.

### Hooks and tick

```text
add_request(new id)          -> switcher.on_new_request(id)
_update_request_with_output  -> switcher.on_request_output(id, num_cumulative_output_tokens)
_free_request                -> switcher.on_request_finished(id, num_cumulative_output_tokens)
schedule()                   -> precision = switcher.tick(SchedulerStep(live, unfinished))
                                SchedulerOutput(dual_precision_base_precision=precision,
                                                num_unfinished_requests=get_num_unfinished_requests())
```

`live` is every request that is unfinished and not
`WAITING_FOR_STREAMING_REQ`, with `num_prompt_tokens` and
`num_cumulative_output_tokens` (`streaming_output_token_offset +
num_output_tokens`; resumable streaming folds kept output into the prompt at
a chunk boundary and the offset keeps it as response progress). With the
policy flag empty the switcher is `None`, every hook is skipped and the
output field stays `None`; `num_unfinished_requests` is always populated
(it is an int the dispatcher may use for batch sizing).

### Rollouts as cohorts

Two arming modes, chosen by the policy:

* **Cohort arming** (`initial_rollout_batch = B`, every calibrated table and
  every file built by `build_static_policy.py`). The cohort is exactly B
  genuinely new request ids. The first id arms rollout 1; the first id after
  a complete cohort is the boundary of the next rollout (waiting for the full
  next cohort would let early arrivals decode past a frontier without a
  commitment). While the cohort is still arriving, `decision_live = live +
  (B - arrived)` indexes the table, so asynchronous admission cannot make a
  B64 rollout look like B8 at the first observation; the live-batch guard
  and the logged `applied_live_requests` use the actual live count. An
  empty scheduler never ends a cohort rollout (early finishers, streaming
  gaps), and a re-admitted resumable continuation carrying a known id is not
  a new member. The (B+1)th new id is by definition the first member of the
  next rollout (the experimental runtime behaved the same way; its
  "cohort exceeded" error was unreachable). Caveat: every rollout the engine
  sees, validation included, must submit exactly B new ids, otherwise cohort
  boundaries drift for the rest of the run; and with reload enabled a
  validation rollout is a boundary that consumes a policy reload. Size
  validation batches to B or run them on a separate engine.
* **Cohort-free arming** (inline specs; `initial_rollout_batch = None`,
  `arm_min_requests = 1`). The rollout arms at the first `tick` with at
  least `arm_min_requests` unfinished requests, `decision_live` is the
  actual live count, and the rollout ends when the scheduler drains (drain
  re-arm: the next batch switches again) or when a live set disjoint from
  the armed rollout appears (back-to-back batches). This is what the
  experimental env-threshold latch did.

### Watermark and decision

`tick` computes `max_response = max(watermark, max live output)`, the upper
median prompt of the live set (`sorted[n // 2]`), and calls
`decider.observe(max_response, median_prompt, decision_live, actual_live,
max_response)` on **every** tick after arming. The decider quantizes the
frontier to the 250-token scan grid, looks the table up once per new grid
frontier, applies the commitment mode (`monotone`: `min`; `receding`: replace)
and the switch predicate (`max_response >= committed and actual_live <=
switch_live_cap`); the monotone predicate is evaluated on every call so a
guard-blocked switch fires as soon as the batch drains. The precision is
`decider.base_precision` (one-way INT4 until the next rollout;
`uniform_w4` is INT4 from construction and never reports a switch).

The watermark is advanced from output processing, not only in `tick`: a
request can cross the frontier on its final token and be freed before the
next `schedule()`; without the watermark the rollout would never observe
that frontier (archived scenario "request finished between schedules").

### Switch cohort JSONL

When `VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS` is set, the switching tick
appends one record (`sort_keys`, one object per line, directory created):

```json
{"event": "switch_cohort", "rollout_index": 7, "policy_kind": "lookup", "policy_revision": 6,
 "trigger": {"committed_frontier": 9000, "applied_response_tokens": 9000, "applied_live_requests": 4,
             "decision_live_requests": 4, "median_prompt_tokens": 62.0, "reason": "switch"},
 "requests": [{"request_id": "<client>-<8hex>", "entry_output_tokens": 9000, "prompt_tokens": 59}, ...]}
```

`event`, `rollout_index`, `requests[].request_id`,
`requests[].entry_output_tokens` are the archived keys the calibrator reads
(`verl/experimental/precision_scheduler/traces.py`: `read_cohorts`,
`cohort_observation`, `resolve_request_id` strips the EngineCore `-<8 hex>`
suffix). The per-request key is `entry_output_tokens`; the final response
length is not in the record, the calibrator joins it from the request
lifetime traces (`generation_tokens`). `requests[].prompt_tokens`,
`trigger`, `policy_kind`, `policy_revision` and `policy_reload_lagged`
(the reload before this rollout saw an unchanged revision, see below) are
additive: they replace the experimental "Dynamic precision exact switch
request states" log line (parsed by three archived plot scripts) and let an
audit check which revision drove a switch. The same record is kept in memory
(`switcher.switches`, `last_switch`) for tests and tooling.

The log line `Lookup dynamic full-cost switch: rollout_index=%d,
committed_frontier=%d, applied_response_tokens=%d, applied_live_requests=%d`
is emitted verbatim for every kind (it is the `SWITCH_RE` contract of the
archived audits); `Reloaded dynamic precision policy before rollout %d:
revision=%d` and `Precision policy reload lagged before rollout %d: ...`
keep their formats too. These three contract lines are logged at
**WARNING** (one line per event, as the archived runs did): verl launches
vLLM with `VLLM_LOGGING_LEVEL=WARN`, and at INFO they were invisible to
`validate_rollout_run.py` from a recipe launch (integration defect 3). The
informational lines (`Precision rollout armed`, `... cohort complete`,
`Dynamic precision lookup commitment updated`, `... receding lookup
updated`) stay at INFO and the per-step trace at DEBUG. The switcher takes
`log` (informational) and `contract_log` callables; a single `log` override
captures both.

### Reload at rollout boundaries

With `VLLM_DUAL_PRECISION_RELOAD_POLICY_EACH_ROLLOUT=1` and a file policy,
`start_rollout` calls `PolicyStore.reload()` exactly once per boundary,
rollout 1 included (the archived runs log `Reloaded ... before rollout 1:
revision=0`). The reload is invoked from `on_new_request` before the
scheduler registers the request, so a raise leaves scheduler state
untouched. Three outcomes from rollout 2 on:

* revision advanced: the new table is installed for this rollout;
* revision unchanged (the calibrator lagged the boundary): by default a
  warning is logged once per boundary, `switcher.policy_reload_lag_count`
  is incremented and the rollout's cohort record carries
  `policy_reload_lagged: true`; the rollout runs on the previous table. The
  archived headline runs did lag occasionally (`b32_cap16384_ema_alpha_a000_30step`
  before rollout 6, `ema_pair128_a010` before rollouts 6 and 22), so a hard
  failure here would have aborted them;
* invalid file or revision went backwards: `PolicyRevisionError` always.

The primary lag protection is a verl-side barrier: the trainer waits for
the calibrator's new revision before submitting the next rollout
(config key `precision_scheduler.policy_barrier_timeout_s`, added by C8).
`VLLM_DUAL_PRECISION_REQUIRE_POLICY_ADVANCE=1` is the strict fallback
inside vLLM: an unchanged revision at a boundary from rollout 2 on raises
(the store exempts the reload before rollout 1, where the calibrator has
not run yet), so a stale table cannot silently drive a run the way the
B128/16K continuous-EMA run kept switching on a frozen revision-14 table
for 15 steps (handoff section 9.6). The experimental runtime reloaded twice
per boundary (59 reloads for 29 boundaries in every Sep-8 run) and only
logged exceptions. Inline specs never reload.

### Profiler seam

`switcher.set_forced_precision("bf16" | "int4" | None)` pins the published
precision while the state machine keeps running (cohort logs stay
meaningful). This replaces the private `_dual_precision_base_precision`
hook the efficiency-heatmap harness used to override, whose name mismatch
invalidated a run (`INVALID_PRECISION_SELECTOR.md`); the clean heatmap
harness (C6) selects the row precision with `uniform_w4` per engine launch
instead.

## Knobs and defaults

| env (vllm/envs.py) | default | meaning |
|---|---|---|
| `VLLM_DUAL_PRECISION_POLICY` | `""` | policy spec or JSON path; empty = switcher off, output field `None` |
| `VLLM_DUAL_PRECISION_ONLINE_OBSERVATIONS` | `""` | switch-cohort JSONL path; empty = no file (records still kept in memory) |
| `VLLM_DUAL_PRECISION_RELOAD_POLICY_EACH_ROLLOUT` | `0` | reload once per boundary; unchanged revision = logged lag, recorded in the cohort JSONL |
| `VLLM_DUAL_PRECISION_REQUIRE_POLICY_ADVANCE` | `0` | strict fallback: raise on an unchanged revision from rollout 2 on (verl's policy barrier is the primary mechanism) |

Policy JSON fields the runtime honours: `scan_interval_tokens`,
`initial_rollout_batch`, `arm_min_requests`, `receding_horizon_lookup`,
`max_switch_live_batch` / `capture_max_batch` (guard), `lookup_table`,
`calibration.policy_revision`. verl translates the YAML block
`actor_rollout_ref.rollout.precision_scheduler` into these variables
(decision 9; C8/C9), so recipes never set them by hand.

## Contracts with neighbors

- **C5 policy** (`precision_policy.py`): `PolicyStore` (load / reload with
  revision check), `PolicyDecider.observe` on every tick after arming,
  `decider.switched` / `base_precision`. `fixed_threshold:t` is the C5
  emulation (constant-250 table plus guard `t`), identical to the archived
  full-RL-matrix `fixed_t{2,4,8}.json` files.
- **C3 dispatch / C2 residency**: read
  `SchedulerOutput.dual_precision_base_precision` (`"bf16"`, `"int4"` or
  `None`) and `num_unfinished_requests`; when the field is `None` the
  worker must behave as vanilla. Neither is merged yet: the GPU smoke
  verifies the signal and the file, not a weight change.
- **C6 calibrator**: consumes the cohort JSONL (keys above), writes a new
  policy revision atomically before the next rollout's first request; the
  switcher reloads it once, records a lag when it is late, and (strict
  knob) refuses to run on a stale revision.
- **C8 verl harness**: `precision_scheduler.policy_barrier_timeout_s` waits
  for the calibrator's revision before submitting the next rollout.
- **C7 re-prefill** (`VLLM_DUAL_PRECISION_REPREFILL`, default off): reads
  `switcher.last_switch` right after `tick` in `schedule()` and preempts every
  survivor once per switch event (rollout index as the key). It does not
  change the response-length accounting: output tokens survive preemption,
  so `num_cumulative_output_tokens` is unaffected. See
  `dual_precision_residency.md`, "Re-prefill after the switch".
- **verl** `build_static_policy.py --kind frontier|live_threshold|forced_switch`
  builds the file-based baselines (cohort arming with explicit B); byte-equal
  to the archived `fixed_k*`, `fixed_t*`, `*_fixed_frontier*_30step` and
  `b128_cap24576_forced_tail8k_calibration` policies.

## Dropped from the experimental code, and why

| item | reason |
|---|---|
| in-scheduler online cost loop (`predict_receding_horizon` on the scheduler thread) and the `switch_thresholds` fallback | decision 4; no completed final run used them (0 `Dynamic full-cost switch` / `Dynamic dual precision switch` lines in any headline log) |
| async `ProcessPoolExecutor(spawn)` predictor, readiness future, shutdown hook, `VLLM_DUAL_PRECISION_ASYNC_*` | profiling runs only; spawning a child process from EngineCore at scheduler init is unsafe under Ray |
| `lookup_hierarchy` branch | two policies built, never launched |
| NVTX / nsys window code (`_nvtx_range_push/_pop`, `_dynamic_nsys_*`, three bridge blocks, the forced switch at iteration 3) | profiling scaffolding that altered switching; results archived under `nsight_*_20260823/` |
| separate env-threshold latch, `threshold >= max_num_seqs` uniform trick | decision 6: `fixed_threshold:t` and `uniform_w4` through the one switcher, cohorts logged for every kind |
| double reload per boundary, exception-swallowing reload | reload once; lag counted and recorded, strict fail-closed knob |
| `dynamic_precision_peak_requests`, `last_prediction`, `last_context`, `Dynamic cost policy no-switch` log | only fed cost-model log lines |
| "Dynamic precision exact switch request states" log line | superseded by `prompt_tokens` in the JSONL |

Behavioural difference to note: `fixed_threshold:t` with cohort-free arming
commits at the first 250-token frontier, so a batch that is already `<= t`
when its longest response reaches 250 tokens switches there, whereas the
experimental env latch required having *seen* `live > t` first. No archived
fixed-t configuration is affected (every arm had `B > t` and responses far
longer than 250 tokens); the equivalence test covers B64 with t in {2,4,8}.

## Measured numbers and provenance

CPU replays under `/data/huanchen/verl/.codex-report/new-storyline-experiments/dynamic_tail8k_heatmap_20260823/`
(fixtures trimmed by `extract_switch_golden.py`; tests run on the fixtures
and skip only if a fixture directory is missing):

| golden | oracle | result |
|---|---|---|
| lockstep replay, receding EMA table | `runs/b32_cap16384_ema_alpha_a000_30step` (`driver.log` 29 switch lines, `online_switch_cohorts.jsonl`, `traces/`; `policies/b32_cap16384_ema_alpha_a000_30step.json`, revision 29 == revision 0 at alpha 0) | 29/29 switches with identical `committed_frontier`, `applied_response_tokens`, `applied_live_requests`; 103/103 cohort members exact; entry tokens within 4 of the archive (admission skew) |
| lockstep replay, fixed frontier 8000 | `runs/b32_cap16384_fixed_frontier8000_30step` (28 switch lines, cohorts from the `exact switch request states` lines; the run never set the observations path) | 28/28 switches exact; 135/135 members; entries within 4 |

Lockstep replay (every live request advances one token per tick; a request
with `generation_tokens` G is live at ticks `0..G-1`) reproduces every
logged receding observation of rollouts 1-3 (`live_requests` and
`median_prompt_tokens` at all 66 frontiers) before the switcher is even
involved, which is why the trigger frontier and live count are exact.

Not reproduced here: the Aug-24/25 continuous-EMA30 runs (produced by an
earlier scheduler without cohort arming; only summary EMA history on disk,
no per-revision tables) and the B128/16K run (stale revision). The C5
golden `replay_policy_log.py` covers the decider on the static and receding
warm5 logs; this component's goldens cover the cohort / watermark / arming
logic on top of it.

verl: `tests/precision_scheduler/test_build_static_policy_on_cpu.py`
reproduces `full_rl_policy_matrix_b64_cap24k_20260902/policies/{fixed_k6000,fixed_k8000,fixed_k10000,fixed_t2,fixed_t4,fixed_t8}.json`,
three `dynamic_tail8k_heatmap_20260823/policies/*_fixed_frontier*_30step.json`
files and `b128_cap24576_forced_tail8k_calibration.json` byte for byte
(`fixed_k8000.json` and `b64_cap24576_fixed_frontier8000_30step.json` are
the same bytes in the archive).

GPU smoke (`test_precision_switch_gpu.py`, Qwen3.5-4B, `fixed_frontier:32`,
8 prompts x 128 tokens, eager): the signal is `bf16` until the step at which
the longest response reaches 32 tokens and `int4` afterwards; one cohort
record with 8 requests and all keys is written. See the component report
for the run under `run_gpu.sh --gpus 7`.

# Precision-keyed CUDA graphs and dispatch

Component C3 of the rollout precision scheduler. Files:
`vllm/v1/cudagraph_dispatcher.py` (key registration, dispatch),
`vllm/v1/worker/gpu_model_runner.py` (bind before execute and capture),
`vllm/v1/worker/gpu_worker.py` (PP+SP pre-dispatch, engine-feature guard),
`vllm/forward_context.py` (`BatchDescriptor.base_precision`, landed with C2),
`vllm/model_executor/dual_precision/loader.py`
(`check_dual_precision_model_runner`). Tests:
`tests/v1/cudagraph/test_precision_cudagraphs.py` (CPU),
`tests/v1/cudagraph/test_precision_cudagraphs_gpu.py` (gpu-smoke).

## Purpose

The scheduler (C4) flips a rollout from BF16 to INT4 base weights once,
late in the decode tail. The flip must not cost the CUDA-graph replay that
makes small-batch decode fast: both precisions need captured graphs, and
the step that runs after the flip must replay the INT4 graph of the same
shape. This component makes the base precision part of the CUDA-graph key,
captures an INT4 graph next to every BF16 LoRA graph up to a ceiling,
selects the graph from the scheduler's per-step precision, binds the weights
to match, and falls back to eager (loudly, once per key) when a requested
graph was never captured.

## Mechanism

### Precision in the key

`BatchDescriptor` is the frozen dataclass that `CUDAGraphWrapper` uses as
the dictionary key of its captured graphs (`concrete_cudagraph_entries`).
The field `base_precision: str = "bf16"` (C2) splits graph storage per
precision with no other change: `BatchDescriptor(num_tokens=8, ...,
base_precision="int4")` is a different key from its BF16 twin. Every
vanilla construction site keeps the default, so a tree with the feature off
builds exactly the vanilla keys.

The precision has to live in a host-side descriptor field rather than in a
device tensor read at runtime: graph launches must stay queued ahead of the
CPU (`.codex-reports/dual_precision_v3/BUGFIX_ROLLOUT_SUMMARY.md`, "CUDA
graph launch was not sufficiently async"), and the LoRA wrapper reads the
bound base layer through an opaque custom op (C2) so the compiled graph is
the same for both precisions; the selection is baked into each captured
graph, which is why replaying a BF16 graph while bound to INT4 still yields
the BF16 result (`test_wrapper_captures_one_graph_per_precision_for_the_same_shape`).

### Key registration

`CudagraphDispatcher.initialize_cudagraph_keys` computes
`int4_capture_max_batch = resolve_int4_capture_ceiling(max_capture_size)`:

```text
None                                   if VLLM_DUAL_PRECISION_ROLLOUT is off
None                                   if VLLM_DUAL_PRECISION_POLICY is empty
min(policy.capture_max_batch,          otherwise (policy loaded through C5's
    max_cudagraph_capture_size)        load_precision_policy; uniform_w4 reports 2^30)
```

Every key goes through `_add_precision_cudagraph_keys`: the BF16 key is
always added; an INT4 twin is added when the ceiling is set, the key has
`has_lora=True` (the shadow binds LoRA wrappers only, so a no-LoRA batch
has nothing to switch) and `num_tokens <= ceiling`. This applies to the
mixed-batch keys (PIECEWISE or FULL) and to the uniform-decode FULL keys
alike, for every policy kind: the experimental static-threshold path
captured one precision per size (INT4 below the threshold, BF16 above) and
the scheduler's BF16 override at `num_tokens=1` ran eager (SmolLM3
preflight log: `FULL=38` then `No matching dynamic-precision CUDA graph for
num_tokens=1 ... base_precision=bf16`); with both precisions registered the
same configuration captures 45 keys and never falls back.

With the headline configuration (LoRA cases `[0, max_loras+1]`, ceiling 32,
default ladder `min(2 * max_num_seqs, 512)`), the key counts are the archived
"Profiling CUDA graph memory" lines:

| `max_num_seqs` | ladder | PIECEWISE | FULL (decode) |
|---|---|---|---|
| 32 | 11 sizes to 64 | 11 x 2 + 7 = 29 | 7 x 2 + 7 = 21 |
| 64 | 19 sizes to 128 | 19 x 2 + 7 = 45 | 11 x 2 + 7 = 29 |
| 128 | 35 sizes to 256 | 35 x 2 + 7 = 77 | 19 x 2 + 7 = 45 |

(7 INT4 twins: sizes 1, 2, 4, 8, 16, 24, 32.) `get_capture_descs` lists the
INT4 keys with the others, sorted largest-first, so
`capture_model` and `profile_cudagraph_memory` capture them without change.

The ceiling is read once, when the keys are initialised. A policy reload at
a rollout boundary (C4) that raises `capture_max_batch` does not add keys;
INT4 steps above the original ceiling run eager with the warning below.
The C5 loader rejects `max_switch_live_batch > capture_max_batch` and
defaults the live-batch guard to the ceiling, so a policy cannot ask for a
switch the dispatcher has no graph for (the hardmath run switched at live
56 with ceiling 32 and ran eleven eager INT4 steps; that combination is
now refused at load).

### Dispatch

```python
dispatch(num_tokens, *, num_reqs=None, uniform_decode=False, has_lora=False,
         num_active_loras=0, base_precision=None, valid_modes=None, invalid_modes=None)
```

Everything after `num_tokens` is keyword-only (the spec-decode callers pass
`num_tokens` positionally and nothing else). `base_precision` is the
scheduler's `SchedulerOutput.dual_precision_base_precision`; `None` means
BF16. The dispatcher never derives a precision from a request count: the
experimental `precision_num_reqs` (live unfinished count re-run through the
fixed threshold on the worker) and the "static mismatch" guard that compared
that choice with the descriptor built from the same count are gone; the
scheduler is the single owner of the decision, for every policy kind.

Resolution order is vanilla's: the padded descriptor is stamped with the
precision, FULL is tried, then the relaxed PIECEWISE key. A precision other
than `"bf16"` / `"int4"` raises `ValueError`. When no key matches:

* bare eager path (keys not initialised, `num_tokens` above the capture
  list, or `valid_modes={NONE}`): `BatchDescriptor(num_tokens,
  num_reqs=num_reqs, base_precision=precision)`, no warning. With no
  `num_reqs` and no precision this is the vanilla `BatchDescriptor(num_tokens)`
  (the experimental version filled `num_reqs=min(num_tokens, max_num_seqs)`
  here and broke four vanilla dispatcher tests).
* a precision was requested but its key was never captured (INT4 above the
  ceiling): eager with the same descriptor, and one warning per
  `(num_tokens, num_reqs, precision)`:
  `No matching dynamic-precision CUDA graph for num_tokens=%d, num_reqs=%s,
  base_precision=%s; falling back to eager execution.` This is how the
  eleven eager INT4 steps of the hardmath run were noticed; the 52 headline
  runs under `dynamic_tail8k_heatmap_20260823/runs` contain zero such lines.

### Runner: bind before execute, bind before capture

`GPUModelRunner._determine_batch_execution_and_padding(...,
base_precision=None)` forwards `num_reqs` and `base_precision` to the
dispatcher (also through the DP re-dispatch closure). `execute_model` passes
`scheduler_output.dual_precision_base_precision` and, right before the
forward, calls `_bind_base_precision(precision)`:
`bind_dual_precision(model, precision, static_forward_context)` when the
feature is on and the precision is not `None`. The bind is idempotent and
cheap (C2), so calling it every step costs nothing after the first flip.

`_dummy_run(..., base_precision=None)` takes the same argument and binds
before its forward; `_warmup_and_capture` passes `desc.base_precision` to
both the warmup runs and the capture run, so every captured graph is
recorded under the binding its key names. Capture descriptors are sorted by
size, so BF16 and INT4 captures interleave and the model may end capture
bound to INT4; `capture_model` and `profile_cudagraph_memory` therefore
finish with `_restore_bf16_binding()`. The worker's PP+SP pre-dispatch
(`gpu_worker.execute_model`, pipeline parallel with sequence parallelism
only) passes the scheduler's precision too.

`None` never binds. With the feature on but no policy, the scheduler
publishes `None`, no INT4 key exists, and the model stays bound to BF16
from `attach_dual_precision` onward; with the feature off nothing on this
path is reached.

### Engine-feature guard

`check_dual_precision_model_runner(vllm_config)` runs in
`Worker.init_device` before the model runner is constructed and, with
`VLLM_DUAL_PRECISION_ROLLOUT=1`, raises `NotImplementedError` for:

* the V2 model runner (`VLLM_USE_V2_MODEL_RUNNER=1`, or the default for an
  unquantized dense `Qwen3ForCausalLM`): it has no attach, bind or
  precision-keyed capture and would silently serve BF16;
* speculative decoding and `kv_sharing_fast_prefill`: their extra dispatches
  build descriptors without a precision (implicitly BF16 graphs under an
  INT4 binding);
* data parallelism: the precision is chosen per DP rank's scheduler and
  `coordinate_batch_across_dp` does not synchronise it.

None of the project's models or archived runs use any of these.

## Knobs and defaults

This component declares no environment variable. It reads (through
`vllm/envs.py`) `VLLM_DUAL_PRECISION_ROLLOUT` (default `0`, C2) and
`VLLM_DUAL_PRECISION_POLICY` (default `""`, C4), and the policy field
`capture_max_batch` (default 32; `fixed_threshold:<t>` lifts it to `t` when
`t > 32`, `uniform_w4` is unbounded and clamps to the capture list; C5).
With both variables at their defaults the dispatcher, runner and worker
behave as vanilla: the vanilla `tests/v1/cudagraph/test_cudagraph_dispatch.py`
passes unchanged (15 tests on GPU), key sets are 10 / 2, and
`test_vanilla_equivalence_when_disabled` pins it with a policy string
present but the feature off.

## Contracts with neighbours

* **C2 residency**: `bind_dual_precision(model, precision,
  no_compile_layers)`, `get_active_precision(model)`,
  `BASE_PRECISION_BF16/INT4`, `BASE_PRECISIONS`,
  `dual_precision_rollout_enabled()`; `attach_dual_precision` has run in
  `load_model` before any capture. The bind call sites listed in C2's
  "known gaps" are this component.
* **C4 scheduler**: `SchedulerOutput.dual_precision_base_precision`
  (`"bf16"`, `"int4"`, `None`) is the only precision source;
  `num_unfinished_requests` is not consumed (the experimental
  `precision_num_reqs` path is dropped).
* **C5 policy**: `load_precision_policy(spec).capture_max_batch` bounds the
  INT4 keys; the loader guarantees `switch_live_cap <= capture_max_batch`.
* **C6 profiler**: the efficiency heatmap needs INT4 graphs for every
  profiled batch size; it launches one engine per row with `uniform_w4`
  (ceiling = whole capture list).

## Dropped from the experimental code, and why

| item | reason |
|---|---|
| `precision_num_reqs` and the worker-side `select_base_precision` | the scheduler already computes the choice for every kind (decision 6); two implementations of one knob (`run_dynamic.sh` set the threshold to -1 and only the scheduler handled it) |
| static-mismatch guard (`base_precision is None and descriptor precision != select_base_precision(count)`) | dead once the scheduler always supplies the precision; with it `None` it compared a descriptor with the count it was built from |
| `_create_eager_batch_descriptor` filling `num_reqs=min(num_tokens, max_num_seqs)` and its range `ValueError` | broke the vanilla bare-eager contract (4 vanilla tests); the eager descriptor now carries only what the caller gave |
| INT4 twins only when a dynamic policy is loaded; one precision per size on the static path | both precisions for every kind, see "Key registration" |
| V2 runner hunks (`gpu/cudagraph_utils.py`, `gpu/dp_utils.py`, `gpu/model_runner.py`) | fixed-threshold only, never received the scheduler override, never selected by a project model; replaced by the guard |
| `_dummy_run`'s `replace(batch_desc, base_precision=...)` after dispatch | the dispatcher now returns the precision-stamped key directly |
| CUDA decode profiler and routed-experts ring buffer hunks in the same files | C6 / decision 12 |

## Measured numbers and provenance

Archived (Qwen3.5-9B TP1, LoRA, `capture_max_batch=32`, vLLM default
ladder), `/data/huanchen/verl/.codex-report/new-storyline-experiments/`:

| run | `Profiling CUDA graph memory` | `Graph capturing finished` |
|---|---|---|
| `dynamic_tail8k_heatmap_20260823/runs/b32_*` | PIECEWISE=29 (largest=64), FULL=21 (largest=32) | 16 s, 1.24 GiB |
| `.../runs/b64_*`, `hardmath_lora_accuracy_100step_20260825/.../cap9k` | PIECEWISE=45 (largest=128), FULL=29 (largest=64) | 20-23 s, 1.86 GiB |
| `.../runs/b128_*` | PIECEWISE=77 (largest=256), FULL=45 (largest=128) | 31-34 s, 3.15 GiB |
| `eos_hazard_extensibility/.../smollm3_3b_gsm8k_tail_w4_t8_cg.log` (static path, SmolLM3) | FULL=38 (largest=128) + 1 eager-fallback warning | 11 s, 0.57 GiB |

Clean branch, this component's GPU smoke (`test_engine_switch_replays_captured_int4_graphs`,
GPU 3, Qwen3.5-9B BF16 + Intel AutoRound INT4 shadow, zero rank-16 LoRA
adapter from `tools/rollout_lora/make_zero_lora.py`, `ROLLOUT_QLORA=1`,
`BF16_LAYERS=none`, `fixed_frontier:32`, `max_num_seqs=64`, 8 prompts x 128
tokens, 2026-09-11, 224 s wall clock):

| quantity | value |
|---|---|
| `Profiling CUDA graph memory` | `PIECEWISE=45 (largest=128), FULL=29 (largest=64)` (archived b64 counts) |
| `Graph capturing finished` | `16 secs, took 0.68 GiB` (archived 1.86 GiB at `max_model_len` 16-24k; here 1024, and the vanilla base now profiles graph memory in a shared pool first) |
| bind lines during capture | `precision=bf16 ... int4_shadow_active=0` then `precision=int4, lora_base_layers=176, rebound_layers=176, int4_shadow_active=152` |
| switch | `Lookup dynamic full-cost switch: rollout_index=1, committed_frontier=32, applied_response_tokens=32, applied_live_requests=8`; one cohort record with 8 requests |
| dispatches | 128 (1 prefill + 127 decode); BF16 for the first 33, INT4 for the remaining 95, every INT4 step `FULL` at `num_tokens=8, uniform, has_lora, int4` |
| eager fallback | 0 `No matching dynamic-precision CUDA graph` lines; `_missing_precision_keys_logged` empty |
| text | coherent for all 8 prompts (Rayleigh-scattering explanations / the model's `<think>` prelude), 128 tokens each |

The wrapper smoke (`test_wrapper_captures_one_graph_per_precision_for_the_same_shape`)
captures two `CUDAGraphWrapper` entries for `num_tokens=10` under the two
bindings of C2's fake wrappers, replays each against its eager reference
(rtol/atol 5e-2) and shows the BF16 entry keeps producing the BF16 result
while the model is bound to INT4.

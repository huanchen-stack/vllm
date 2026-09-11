# Dual-precision residency and binding

Component C2 of the rollout precision scheduler. Package:
`vllm/model_executor/dual_precision/` (`policy_layers`, `loader`, `binding`,
`validation`, `__init__`), plus one attach call in
`vllm/v1/worker/gpu_model_runner.py`, the `BatchDescriptor.base_precision`
field in `vllm/forward_context.py`, and six knobs in `vllm/envs.py`.
The verl half lives in `verl/workers/config/rollout.py`,
`verl/workers/rollout/vllm_rollout/{vllm_async_server,utils}.py` and
`verl/workers/engine_workers.py`.

## Purpose

Keep two copies of the base model resident on one GPU, a BF16 copy (the one
LoRA trains against) and a GPTQ INT4 shadow, and let every LoRA wrapper pick
one of them per forward without changing module topology. The scheduler
(C3/C4) decides *when* to use INT4 (long-tail decode with few live requests);
this component only makes both bases available and switchable.

## Mechanism

1. **Load** (`loader.attach_dual_precision`, called once from
   `GPUModelRunner.load_model` right after `load_lora_model`, inside the
   worker's CuMem `weights` pool). The INT4 checkpoint named by
   `VLLM_DUAL_PRECISION_INT4_MODEL` is loaded through the normal model loader
   with a cloned `VllmConfig` (`make_int4_vllm_config`: same everything except
   `model`/`hf_config_path`, `model_weights=""`, `quantization=None` so the
   shadow's own quant config is auto-detected, and a fresh
   `CompilationConfig` so the shadow's attention layers register in their own
   static forward context). The format is validated first
   (`validate_shadow_quantization`): Intel AutoRound `auto_round:auto_gptq`,
   plain GPTQ, or compressed-tensors `pack-quantized` are accepted; AWQ in any
   form raises `ValueError`.
2. **Match** (`policy_layers`). INT4 `LinearBase` modules carrying a packed
   weight (`qweight`, `weight_packed`, ...) are matched by module name onto the
   BF16 model's LoRA wrappers (the wrapper sits at the linear's original
   name). `VLLM_DUAL_PRECISION_BF16_LAYERS` (default `first:3,last:3`) keeps
   whole transformer blocks BF16; `VLLM_DUAL_PRECISION_INT4_MODULES`
   (`all` | `mlp_only`) restricts to gate/up/down projections. Counts in the
   log line follow the archived semantics over every BF16 `LinearBase`:
   *attached* / *kept BF16 by policy* / *left BF16 (no shadow)*; a fourth
   count, *unwrapped* (quantized and eligible but not LoRA-wrapped, so not
   switchable), is logged as a warning when non-zero.
3. **Store**. Only the attached INT4 linears survive, owned by an
   `Int4ShadowLayerStore` registered on the model as
   `SHADOW_MODULE_NAME = "_vllm_dual_precision_int4_model"` *after* LoRA
   wrapping (the LoRA manager never sees the shadow). The temporary INT4 model
   is dropped (`gc.collect`, `empty_cache`). Because the store is a submodule
   allocated in the `weights` pool, sleep level 1 offloads and restores it
   with the BF16 weights; level 2 would discard it with no way to re-sync
   (hence verl's level-1 rule).
4. **Bind** (`binding`). Each wrapper gets a `DualPrecisionBinding`
   (`bf16`, `int4_or_fallback`, `layer_index`, mutable `active`) stored in
   its `__dict__`, registered in `compilation_config.static_forward_context`
   under `base_layer.prefix + ".dual_precision_base_linear"`, and a closure
   installed through C1's `set_base_forward_override`. The closure calls the
   opaque custom op `torch.ops.vllm.dual_precision_base_linear(layer_name,
   output_size, x, bias)`, whose body looks the binding up in the forward
   context and runs `active.quant_method.apply(active, x, bias)`. Outside a
   forward context (multimodal tower modules) the closure uses the BF16 base
   directly, as the plain sync path would. `bind_dual_precision(model,
   precision)` flips `active` for every binding; it never touches
   `_modules`, `_parameters` or `_buffers` (GEMMA4 audit invariant 1), is
   idempotent per (precision, analysis mask), and logs once per precision.
   The compiled graph sees one stable op, so one Dynamo graph serves both
   precisions and each precision captures its own CUDA graph
   (`BatchDescriptor.base_precision` is part of the graph key; C3/C4 wire the
   bind calls before capture/replay/eager forwards).
5. **Validation** (`validation`, both off by default).
   `VLLM_DUAL_PRECISION_VALIDATE_SHADOW=1` compares every attached INT4 linear
   with its BF16 twin on a random input and logs the ten worst cosines.
   `VLLM_DUAL_PRECISION_VALIDATE_LIFECYCLE=1` records fixed-input probes for
   the first six attached linears at load and re-runs them at the first INT4
   bind, logging `exact=True/False` per probe.

## Knobs (all registered in `vllm/envs.py`; defaults keep vanilla behavior)

| Env var | Default | Meaning |
|---|---|---|
| `VLLM_DUAL_PRECISION_ROLLOUT` | `0` | enable residency (`dual_precision_rollout_enabled()`) |
| `VLLM_DUAL_PRECISION_INT4_MODEL` | `""` | GPTQ INT4 checkpoint; required when enabled |
| `VLLM_DUAL_PRECISION_BF16_LAYERS` | `first:3,last:3` | blocks kept BF16 (`none`, `first:N`, `last:N`, `i`, `a-b`); every final run used `none` |
| `VLLM_DUAL_PRECISION_INT4_MODULES` | `all` | `all` or `mlp_only` (Gemma4 E2B/E4B QAT runs) |
| `VLLM_DUAL_PRECISION_VALIDATE_SHADOW` | `0` | numerical check at load |
| `VLLM_DUAL_PRECISION_VALIDATE_LIFECYCLE` | `0` | probes at load, re-check at first INT4 bind |

verl: `actor_rollout_ref.rollout.model_path` (null: reuse the actor path) and
`actor_rollout_ref.rollout.sleep_level` (null: engine default; env fallback
`VERL_FORCE_VLLM_SLEEP_LEVEL`, read only in the config layer). With
`VLLM_DUAL_PRECISION_ROLLOUT=1` in the environment the config forces level 1
and refuses level 2. C8's `precision_scheduler` block will translate YAML
into these env vars; until then they are set by hand.

## Contracts with neighbours

* **C1 (`base_linear.py`)**: `BaseLinearLayerWithLoRA.base_forward_override`
  / `set_base_forward_override(fn)`; when set, `apply()` computes the base
  output through `fn(x, bias)` and applies LoRA synchronously afterwards
  (the dual-stream op is not used, decision 1). All dual-precision state
  lives in this package, keyed by module.
* **C3/C4**: consume `BASE_PRECISION_BF16/INT4`, `bind_dual_precision(model,
  precision, no_compile_layers)` (call before every capture, replay and eager
  forward with the batch's `base_precision`), `get_active_precision(model)`
  and `BatchDescriptor.base_precision`.
* **verl**: `SHADOW_MODULE_NAME` is imported by
  `_hide_dual_precision_shadow_model` (string fallback), which pops the store
  around `process_weights_after_loading` during weight sync: the Marlin
  repack is not idempotent and, on this vLLM base, asserts on a second visit.
  `aggressive_empty_cache` runs before `rollout.resume(tags=["weights"])`
  so the colocated trainer's allocator cache does not collide with the
  remapped BF16 + shadow weights.

## Dropped from the experimental tree, and why

* **AWQ block-state pairing** (paired norm parameters swapped in
  `_parameters`/`_buffers` per bind) and AWQ shadows altogether. GPTQ only:
  on the Qwen3.5-9B AutoRound shadow 152 of 176 shared non-linear block
  tensors are bit-identical to BF16 and the remaining 24 (128-dim
  gated-deltanet norm vectors) differ at bf16 rounding level (max abs 0.004),
  so the pairing was a near no-op on the headline path; the INT4 path now
  uses the BF16 model's norms. Removing it also removes the only bind-time
  mutation of module dicts.
* **`_modules['base_layer']` swap** (commit 34e66a3): root cause of the
  `KeyError('weight')` under `@support_torch_compile` on Gemma4; replaced by
  the opaque op. `test_committed_modules_swap_violates_the_invariant` keeps
  it out.
* **Module-global caches** (`_BIND_LOGGED`, `_LIFECYCLE_VALIDATED`,
  wrapper discovery cache) and 15 `object.__setattr__` side attributes:
  replaced by one `DualPrecisionState` per model and one
  `DualPrecisionBinding` per wrapper.
* **Two `if self.lora_config` blocks in the runner**: one
  `attach_dual_precision()` call after LoRA load; discovery happens there,
  so `bind_dual_precision` no longer needs `no_compile_layers` (accepted for
  call-site symmetry, ignored).
* **`ROLLOUT_QLORA` gating of the feature flag**: `dual_precision_rollout_
  enabled()` reads only `VLLM_DUAL_PRECISION_ROLLOUT`; the LoRA fast path is
  C1's concern and the override contract works with or without it.
* **Analysis BF16 mask**: kept as `set_analysis_bf16_layers` (eager-only
  diagnostic used through `llm.apply_model`), now a field of the per-model
  state rather than extra attributes; the layer-sensitivity scripts that used
  it are not ported.
* **Marlin K-padding** for Nemotron stays with C9; the Nemotron golden skips
  until it merges.

## Measured numbers (2026-09-11, GPU 3, this branch)

* Qwen3.5-9B BF16 (`/data/huggingface/hub/models--Qwen--Qwen3.5-9B/...c2022362`)
  + Intel AutoRound INT4 (`models--Intel--Qwen3.5-9B-int4-AutoRound/...29688b89`),
  `BF16_LAYERS=none`: `Loaded 286 GPTQ shadow linear layers; attached 152 ...
  kept 0 ... left 134` (identical to the 697 archived runs under
  `/data/huanchen/verl/.codex-report/**`); shadow store 3.32 GiB; worst
  per-layer cosine 0.9856 (`layers.30.linear_attn.in_proj_ba`, rel-RMSE
  0.169). With the default `first:3,last:3`: 123 attached / 29 by policy
  (archived line matches). Lifecycle probes after `sleep(1)`, `wake_up` and a
  weight-sync repack: six probes `exact=True, max_abs=0`.
* Nemotron-Nano-9B-v2 + RedHatAI w4a16: 139 linears, 112 attached / 27
  fallback (the 27 mamba `conv1d` projections), from the module-name fixture;
  archived greedy audit in
  `.codex-report/precision-scheduling-validation/stage0/weight_audit/`
  (dual-resident W4 bit-identical across two runs; prefix agreement with
  standalone W4 of 671 and 215 tokens).
* Gemma4 E2B QAT, `mlp_only`: 213 linears, 70 attached / 141 by policy / 2
  fallback (the vision/audio embedding projections, `LinearBase` on this
  base; the archived log reported 0).
* Fixtures: `tests/model_executor/dual_precision/fixtures/*_modules.json`,
  derived from the checkpoints' safetensors headers by
  `derive_module_names.py`; the Qwen3.5-9B and Gemma4 lists were verified
  equal to a meta-device `initialize_model` of the real vLLM models.

## Known gaps

* `ReplicatedLinearWithLoRA.apply` calls `self.base_layer(x)` directly and
  never reaches the override, so a `ReplicatedLinear` with a quantized peer
  (Gemma4 `per_layer_input_gate` / `per_layer_projection` under
  `INT4_MODULES=all`, MoE routers) is counted as attached but keeps running
  BF16. `mlp_only`, used for every Gemma4 run, excludes them. Fixing this is
  a C1 change (route `ReplicatedLinearWithLoRA.apply` through the base class
  when an override is set).
* Bind call sites in `execute_model`, `_dummy_run` and CUDA-graph capture
  are C3/C4's; until they land the engine attaches the shadow but always
  serves BF16.

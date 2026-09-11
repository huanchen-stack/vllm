# Model support for the rollout precision scheduler (vLLM side)

Component C9 of the `rollout-precision-scheduler-clean` branch. The verl side
of the same component is documented in
`verl/docs/precision_scheduler/model_support.md`.

## Purpose

The precision scheduler (dual-precision BF16/INT4 rollout with a LoRA fast
path) was measured on four model families. Most of the work is configuration
(checkpoint pairs, LoRA target names) and lives in verl; this document covers
the three vLLM-side code changes that some of those models need, and states for
every model which tier it is in.

## Tiers

| Model | Tier | vLLM code needed | Notes |
|---|---|---|---|
| Qwen3.5-9B, Qwen3.5-4B (+ Intel AutoRound INT4) | first-class | none | GatedDeltaNet `in_proj_*`/`out_proj` LoRA targets; vLLM merges them into `in_proj_qkvz`/`in_proj_ba` |
| Phi-4-mini-reasoning (+ llm-compressor W4A16) | first-class, config only | none | `Phi3ForCausalLM`; K=8192 is Marlin-aligned. verl must use `rollout.load_format: auto` (see verl doc). Phi-4-mini-flash-reasoning (`Phi4FlashForCausalLM`) is unsupported by vLLM V1 |
| Gemma-4 E2B/E4B (QAT pair) | flagged | `gemma4.py` k_norm fix | training needs the dense FFPA path on A100 (verl) |
| Gemma-4 12B/31B (`gemma4_unified`) | flagged | `gemma4_unified` registry glue + `gemma4.py` fixes | text-only serving of the unified checkpoints |
| Nemotron-Nano-9B-v2 (`NemotronHForCausalLM`, + RedHatAI W4A16) | flagged | `VLLM_MARLIN_INPUT_PADDING` | without padding `mixer.down_proj` (K=15680) runs on Triton W4A16 |
| Qwen3.5-27B, Falcon-H1, DeepSeek-R1 distills, SmolLM3, GLM-Z1, Kimi-VL, Granite, MiniCPM | dropped | — | screening only; no code in either repo. Granite/MiniCPM/Kimi failed W4 load/output validation in the archived screen |

## Mechanisms

### 1. `gemma4_unified` text-only registry glue

Gemma-4 12B/31B checkpoints were published with the pre-release model type
`gemma4_unified` (architecture `Gemma4UnifiedForConditionalGeneration`, nested
`text_config.model_type = gemma4_unified_text`). Transformers 5.x does not
register the top-level type, but the nested text config is wire-compatible
with `Gemma4Config`.

Files:

- `vllm/transformers_utils/configs/gemma4_unified.py` — `Gemma4UnifiedConfig(Gemma4Config)` with `model_type = "gemma4_unified"`.
- `vllm/transformers_utils/config.py` — `_CONFIG_REGISTRY["gemma4_unified"]`.
- `vllm/transformers_utils/configs/__init__.py` — lazy export.
- `vllm/transformers_utils/model_arch_config_convertor.py` — `gemma4_unified` and `gemma4_unified_text` use `Gemma4ModelArchConfigConvertor`.
- `vllm/model_executor/models/config.py` — `MODELS_CONFIG_MAP["Gemma4UnifiedForConditionalGeneration"] = Gemma4Config`.
- `vllm/model_executor/models/registry.py` — `Gemma4UnifiedForConditionalGeneration -> ("gemma4", "Gemma4ForCausalLM")`: the text-only class; vision/audio embedders are deliberately not constructed.

Inert unless such a checkpoint is loaded.

### 2. `Gemma4ForCausalLM.load_weights` fixes

- `vision_embedder.` is added to the skip list: encoder-free unified
  checkpoints store a raw-patch projection under that prefix.
- YOCO KV-shared layers (the last `num_kv_shared_layers`: 20 for E2B, 18 for
  E4B, 0 for 12B/31B) carry no `k_norm` in the official checkpoints because
  their K/V come from an earlier layer. Vanilla `Gemma4ForConditionalGeneration`
  could not load the official E2B checkpoint at all (`ValueError: Following
  weights were not initialized from checkpoint` for
  `model.layers.15..34.self_attn.k_norm.weight`); this is not a behavior change
  but a load fix. `Gemma4Attention` keeps the module for a uniform
  implementation and never evaluates it when `is_kv_shared_layer`, so
  `load_weights` marks those never-read parameters as loaded. The multimodal
  wrapper delegates to the same method, so it is fixed too.
  `tests/models/test_gemma4_unified_config.py` asserts the invariant by making
  `k_norm.forward` raise on a KV-shared layer.

### 3. Marlin input padding (`VLLM_MARLIN_INPUT_PADDING`)

`vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_wNa16.py`

Marlin needs K to be a multiple of 128. With the flag on, a symmetric,
groupwise, non-actorder WNA16 layer whose full K equals its partition K and
`K % 128 != 0` is padded to the next tile: the packed int32 weight gets
`padding / pack_factor` zero columns (packed zeros decode to zero), the scales
get `padding / group_size` columns of 1.0 (unit scale keeps the all-zero group
finite), `weight_shape[1]` is rewritten, and `apply_weights` zero-pads the
activation by the same amount. The result is mathematically the original GEMM.

Guards and contracts:

- Loaders are attached per parameter (`weight_packed`, `weight_scale`,
  `weight_shape`) by identity, never chosen by tensor shape. A `weight_shape`
  whose K matches neither the original nor the padded size raises.
- `padding % group_size != 0` raises `ValueError` (a partial group cannot be
  represented). Nemotron: padding 64, group 64.
- **TP > 1 row-parallel fallthrough**: when `input_size != input_size_per_partition`
  (K sharded across ranks) the layer is *not* padded and keeps the vanilla kernel
  choice (Triton when unaligned), because padding only the last shard would
  misalign scales across ranks. Column-parallel layers keep the full K per rank
  and are padded normally. This branch only runs TP=1.
- Asymmetric or `g_idx` (actorder) layers are never padded.
- One `logger.info_once` line per padded layer.

Default off; with the flag unset the scheme is byte-for-byte vanilla
(`tests/quantization/test_wna16_input_padding.py::test_padding_off_by_default_keeps_vanilla_shapes`).

## Knobs

| Knob | Where | Default | Effect |
|---|---|---|---|
| `VLLM_MARLIN_INPUT_PADDING` | `vllm/envs.py` (`"1"` to enable) | off | Pad unaligned WNA16 K to a Marlin tile (TP=1 only) |

Everything else in this component is data (checkpoint pairs and LoRA target
names in verl's `examples/precision_scheduler/models/*.yaml`). Whether verl
exposes the padding flag as a `rollout.precision_scheduler` key is an open item
(the C8 block does not carry it yet; until then it is set in the vLLM server
environment).

## Tools

`tools/precision_scheduler/models/`

- `validate_nemotron_marlin_padding.py --snapshot <dir> [--layer ...] [--output json]` — Triton (K=15680) vs padded Marlin (K=15744) on one real Nemotron layer; exit 1 if cosine < 0.99999.
- `prepare_phi4mini.py --output-dir <dir> [--bf16 ...] [--w4 ...] [--prompts-jsonl ...]` — BF16/W4 pair audit (quant method, vocab/EOS/chat-template identity, tokenizer fingerprints), optional prompt rendering with the `<|assistant|>` marker check, and a zero-output rank-16 LoRA adapter with the model's native fused target names.

## Tests

| Test | Tier | What it checks |
|---|---|---|
| `tests/models/test_gemma4_unified_config.py` | unit | `get_config` on the real 12B `config.json` -> `Gemma4UnifiedConfig`; registry/convertor/config-map entries; tiny `Gemma4ForCausalLM` on CPU: `load_weights` complete without KV-shared `k_norm`, `vision_embedder.*` skipped, `k_norm` never read on a KV-shared layer |
| `tests/quantization/test_wna16_input_padding.py` | unit | `create_weights` shapes with the flag off/on (Nemotron K=15680 -> 15744; 8192/10240/15360/21504 untouched), whole-group `ValueError`, TP>1 fallthrough, asymmetric/g_idx untouched, the three loaders, activation padding |
| `tests/quantization/test_wna16_input_padding_gpu.py` | gpu-smoke | real `backbone.layers.1.mixer.down_proj`: padded Marlin vs Triton, cosine >= 0.99999 (skips without the snapshot) |
| `tests/tools/test_prepare_phi4mini.py` | unit | pair audit accept/reject, zero-LoRA builder, language-tower restriction |

## Measured numbers (provenance)

- Nemotron padded Marlin vs Triton, GPU 5, 2026-09-11: cosine 1.0,
  max_abs 0.03125, relative_rmse 2.856e-4 — identical to the archived oracle
  `/data/huanchen/verl/.codex-report/new-storyline-experiments/eos_hazard_extensibility/manifests/nemotron_padded_marlin_numerical_validation.json`.
- The verl-side FFPA gpu-smoke test
  (`verl/tests/models/test_gemma4_ffpa_dense_on_gpu.py`) passes only with the
  archived `ffpa_attn` site
  (`/data/huanchen/verl/.codex-report/new-storyline-experiments/eos_hazard_extensibility/tools/ffpa_site`)
  on `PYTHONPATH`; without `ffpa_attn` importable it skips.
- Gemma-4 E2B (`google/gemma-4-E2B-it-qat-q4_0-unquantized`) loads through
  `Gemma4ForCausalLM` (via `hf_overrides={"architectures": ["Gemma4ForCausalLM"]}`)
  in 15.8 s on GPU 5 and produces the same greedy continuation as the
  multimodal `Gemma4ForConditionalGeneration` path on the same prompt.
- Phi-4-mini-reasoning W4A16 (`alishafique/...gptq-w4a16-llmcompressor`) loads
  with `MarlinLinearKernel` for every WNA16 layer (K=8192, no padding needed)
  in 10.5 s on GPU 5.
- Archived timing context (not re-measured here): Nemotron W4 arms in
  `.codex-report/.../reports/ALL_MODEL_DATASET_TIMING_WORK_MASTER.md` "mix
  Marlin/Triton" without padding; the padded arms are a supplement
  (`run_supplement_arm.sh`, `marlin_padding=1`).

## Dropped from the experimental tree and why

- `tests/model_executor/test_dual_precision.py::test_wna16_input_padding_*` —
  moved to `tests/quantization/` (they test the scheme, not dual precision).
- Shape-heuristic loader (`numel()==2` / `ndim==2`) — replaced by per-parameter loaders.
- Qwen3.5-27B TP4 configs and its kernel-ablation launcher — multi-GPU out of scope.
- The extensibility zoo (Falcon-H1, DeepSeek distills, SmolLM3, GLM-Z1, Kimi-VL,
  Granite, MiniCPM, LFM2.5) — never had code; screening runs only.
- `prepare_*.py` manifest mutation and hard-coded snapshot hashes — the tool now
  takes paths and writes a single `audit.json`.

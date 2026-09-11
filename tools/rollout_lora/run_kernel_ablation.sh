#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Rollout LoRA kernel ablation: runs the four LoRA kernel stages
#   punica -> torch -> torch-fused -> torch-fused-dual
# on the same fixed-work cell for each requested base backend, one after
# another on a single GPU, then renders the markdown table.
#
# Usage:
#   tools/rollout_lora/run_kernel_ablation.sh --model PATH --adapter DIR --out DIR \
#       [--int4-model PATH] [--backends bf16,marlin] [--python BIN] \
#       [--cells 1x512] [--decode-tokens 64] [--warmups 2] [--repetitions 5] \
#       [--archived CSV] [--hf-config-path PATH] [--stages punica,torch,...]
#
# Every knob is also overridable through the environment (MODEL, INT4_MODEL,
# ADAPTER, OUT, PYTHON, BACKENDS, ...). CUDA_VISIBLE_DEVICES is inherited: run
# this under your GPU launcher. Dual precision is never enabled (decision 1).
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODEL=${MODEL:-}
INT4_MODEL=${INT4_MODEL:-}
ADAPTER=${ADAPTER:-}
OUT=${OUT:-}
PYTHON=${PYTHON:-python3}
BACKENDS=${BACKENDS:-bf16}
STAGES=${STAGES:-punica,torch,torch-fused,torch-fused-dual}
CELLS=${CELLS:-1x512}
DECODE_TOKENS=${DECODE_TOKENS:-64}
WARMUPS=${WARMUPS:-2}
REPETITIONS=${REPETITIONS:-5}
ARCHIVED=${ARCHIVED:-}
HF_CONFIG_PATH=${HF_CONFIG_PATH:-}
EXTRA_ARGS=${EXTRA_ARGS:-}

while [ $# -gt 0 ]; do
  case "$1" in
    --model) MODEL=$2; shift 2 ;;
    --int4-model) INT4_MODEL=$2; shift 2 ;;
    --adapter) ADAPTER=$2; shift 2 ;;
    --out) OUT=$2; shift 2 ;;
    --python) PYTHON=$2; shift 2 ;;
    --backends) BACKENDS=$2; shift 2 ;;
    --stages) STAGES=$2; shift 2 ;;
    --cells) CELLS=$2; shift 2 ;;
    --decode-tokens) DECODE_TOKENS=$2; shift 2 ;;
    --warmups) WARMUPS=$2; shift 2 ;;
    --repetitions) REPETITIONS=$2; shift 2 ;;
    --archived) ARCHIVED=$2; shift 2 ;;
    --hf-config-path) HF_CONFIG_PATH=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
[ -n "$MODEL" ] && [ -n "$ADAPTER" ] && [ -n "$OUT" ] || {
  echo "usage: $0 --model PATH --adapter DIR --out DIR [--int4-model PATH] [--backends bf16,marlin]" >&2
  exit 2
}

mkdir -p "$OUT/results" "$OUT/logs"
status=0
for backend in ${BACKENDS//,/ }; do
  model=$MODEL
  if [ "$backend" = marlin ]; then
    [ -n "$INT4_MODEL" ] || { echo "--int4-model is required for the marlin backend" >&2; exit 2; }
    model=$INT4_MODEL
  fi
  for kernel in ${STAGES//,/ }; do
    label="${backend}_${kernel}"
    if [ -s "$OUT/results/$label.jsonl" ]; then
      echo "[skip] $label already has a row"
      continue
    fi
    echo "[run ] $label  model=$model"
    args=(run --model "$model" --adapter "$ADAPTER" --output "$OUT/results/$label.jsonl"
          --base-backend "$backend" --lora-kernel "$kernel" --label "$label"
          --cells "$CELLS" --decode-tokens "$DECODE_TOKENS"
          --warmups "$WARMUPS" --repetitions "$REPETITIONS")
    [ -n "$HF_CONFIG_PATH" ] && [ "$backend" = marlin ] && args+=(--hf-config-path "$HF_CONFIG_PATH")
    # shellcheck disable=SC2086
    if ! "$PYTHON" "$HERE/kernel_ablation.py" "${args[@]}" $EXTRA_ARGS >"$OUT/logs/$label.log" 2>&1; then
      echo "[fail] $label (see $OUT/logs/$label.log)"
      rm -f "$OUT/results/$label.jsonl"
      status=1
    else
      echo "[ ok ] $label: $(grep -o '"tpot_ms": [0-9.]*' "$OUT/results/$label.jsonl" | tail -1)"
    fi
  done
done

table_args=(table --results "$OUT/results" --output "$OUT/kernel_ablation.md")
[ -n "$ARCHIVED" ] && table_args+=(--archived "$ARCHIVED")
"$PYTHON" "$HERE/kernel_ablation.py" "${table_args[@]}"
exit $status

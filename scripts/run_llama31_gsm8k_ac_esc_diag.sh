#!/usr/bin/env bash
# 诊断：AC τ=0.99 + PC C=0；ESC 连续两窗或满 8 才停。
# 不覆盖 Table 2 的 AC τ=0.95 / ESC 单窗 4 条。未满 1319 不填表。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/26xyw/.conda/envs/PC/bin/python"
MODEL="/home/26xyw/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
DATASET="/home/26xyw/data/Datasets/gsm8k_test.jsonl"
OUT="/home/26xyw/data/path-consistency-results/gsm8k"
PAIR="${PAIR:-both}"
GPU_AC="${GPU_AC:-1}"
GPU_ESC="${GPU_ESC:-3}"

export HF_HOME="${HOME}/data/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT/logs"
cd "$ROOT"

launch() {
  local gpu="$1"
  local methods="$2"
  local tag="$3"
  shift 3
  local log="$OUT/logs/${tag}.log"
  if pgrep -f "/.conda/envs/PC/bin/python -u run_math.py .*--tag ${tag}" >/dev/null; then
    echo "already running: ${tag}" >&2
    return 0
  fi
  echo "launch gpu=${gpu} methods=${methods} tag=${tag} -> ${log}"
  setsid nohup "$PY" -u run_math.py \
    --gpus "$gpu" \
    --tensor-parallel-size 1 \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --outdir "$OUT" \
    --tag "$tag" \
    --methods "$methods" \
    --max-branches 32 \
    --window-size 4 \
    --temperature 0.6 \
    --top-p 0.9 \
    --top-k -1 \
    --max-tokens 2048 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --logprobs 20 \
    --seed 0 \
    --allow-cudagraph \
    --answer-fallback gsm8k \
    "$@" \
    < /dev/null >> "$log" 2>&1 &
  echo "  pid $!"
}

case "$PAIR" in
  ac)
    launch "$GPU_AC" "ac,pc_ac" "gsm8k_llama31_b32_mt2048_ac099_beta0_seed0" \
      --beta-threshold 0.0 \
      --early-stop-threshold 0.99
    ;;
  esc)
    launch "$GPU_ESC" "esc,pc_esc" "gsm8k_llama31_b32_mt2048_esc2w8_seed0" \
      --beta-threshold 0.8 \
      --early-stop-threshold 0.95 \
      --esc-stop-rule two_window_or_8
    ;;
  both)
    launch "$GPU_AC" "ac,pc_ac" "gsm8k_llama31_b32_mt2048_ac099_beta0_seed0" \
      --beta-threshold 0.0 \
      --early-stop-threshold 0.99
    launch "$GPU_ESC" "esc,pc_esc" "gsm8k_llama31_b32_mt2048_esc2w8_seed0" \
      --beta-threshold 0.8 \
      --early-stop-threshold 0.95 \
      --esc-stop-rule two_window_or_8
    ;;
  *)
    echo "PAIR must be ac, esc, or both" >&2
    exit 1
    ;;
esac
echo "logs: $OUT/logs"

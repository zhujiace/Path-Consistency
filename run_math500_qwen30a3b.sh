#!/usr/bin/env bash
# Qwen3-30B-A3B MATH-500 490 paired SC/PC (exclude engineering 10).
#
# 4090 lab default: GPUs 1-4, TP=4, V1 CUDA-graph, KV blocks=8192.
# New / larger GPU server: override the env vars below. Do NOT copy the
# 8192-block cap onto 80 GB cards — that *limits* KV. See handoff.md.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="${PC_PYTHON:-/home/26xyw/.conda/envs/PC/bin/python}"
MODEL="${QWEN_A3B_MODEL:-${HOME}/data/Qwen3-30B-A3B}"
DATA="${MATH500_JSONL:-${HOME}/data/Datasets/math500_test.jsonl}"
OUT="${PC_RESULTS_MATH500:-${HOME}/data/path-consistency-results/math500}"
GPUS="${QWEN_GPUS:-1,2,3,4}"
TP="${QWEN_TP:-4}"
TAG="${QWEN_TAG:-math500_qwen3_30b_a3b_b32_mt32768_c08_seed0}"
# Unset → 8192 (4090 recipe). Empty string → omit the flag (preferred on large VRAM).
QWEN_GPU_BLOCKS="${QWEN_GPU_BLOCKS-8192}"
LOGDIR="$OUT/logs"
LOG="$LOGDIR/${TAG}.log"

mkdir -p "$LOGDIR"
export HF_HOME="${HF_HOME:-${HOME}/data/huggingface}"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
unset VLLM_USE_V1

EXTRA=()
if [[ -n "${QWEN_GPU_BLOCKS}" ]]; then
  EXTRA+=(--num-gpu-blocks-override "${QWEN_GPU_BLOCKS}")
fi
if [[ -n "${QWEN_MAX_QUESTIONS:-}" ]]; then
  EXTRA+=(--max-questions "${QWEN_MAX_QUESTIONS}")
fi
if [[ -n "${QWEN_SOURCE_ROWS:-}" ]]; then
  EXTRA+=(--source-rows "${QWEN_SOURCE_ROWS}")
fi

cd "$ROOT"
CMD=(
  "$PY" -u run_math.py
  --gpus "$GPUS"
  --tensor-parallel-size "$TP"
  --model "$MODEL"
  --dataset "$DATA"
  --outdir "$OUT"
  --tag "$TAG"
  --methods sc,pc
  --exclude-engineering-split
  --max-branches 32
  --window-size 4
  --physical-window-size 4
  --beta-threshold 0.8
  --temperature 0.6
  --top-p 0.95
  --top-k 20
  --max-tokens 32768
  --max-model-len 34816
  --gpu-memory-utilization 0.90
  --max-num-seqs 8
  --max-num-batched-tokens 256
  --cudagraph-capture-sizes 256,128,64,32,16,8,4,2,1
  --vllm-v1
  --allow-cudagraph
  --logprobs 20
  --seed 0
  "${EXTRA[@]}"
)

echo "launch ${TAG} gpus=${GPUS} tp=${TP} model=${MODEL}"
if [[ "${QWEN_FOREGROUND:-0}" == "1" ]]; then
  exec "${CMD[@]}"
fi
echo "log ${LOG}"
setsid nohup "${CMD[@]}" < /dev/null > "$LOG" 2>&1 &
echo "pid $!"
echo "log $LOG"

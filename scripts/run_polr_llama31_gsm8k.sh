#!/usr/bin/env bash
# Llama-3.1-8B-Instruct × GSM8K test 1319, PoLR (arXiv 2601.21494).
# 默认 GPU 2；CPT 已结束后也可用 GPU 4。不要占 GPU 0。
#
# PREFIX_LEN=64：本机 SC-32 轨迹 mean 260 / p10 160，64 才是真前缀。
# PREFIX_LEN=128：64 与 256 之间的插值。
# PREFIX_LEN=256：论文默认，对多数题接近全文。
# N=32 对齐 SC B=32。采样对齐 Llama GSM8K SC/PC：T=0.6, top-p=0.9, max_tokens=2048。
# 未完成 1319 题之前不要填新 L_p 行。
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/26xyw/.conda/envs/PC/bin/python"
MODEL="/home/26xyw/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
DATASET="/home/26xyw/data/Datasets/gsm8k_test.jsonl"
OUT="/home/26xyw/data/path-consistency-results/gsm8k"
GPU="${GPU:-2}"
PREFIX_LEN="${PREFIX_LEN:-64}"
TAG="${TAG:-polr_llama31_gsm8k_n32_lp${PREFIX_LEN}_seed0}"

export HF_HOME="${HOME}/data/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
unset VLLM_USE_V1

mkdir -p "$OUT/logs"
cd "$ROOT"

if ps -eo pid,cmd | grep -E '[/]conda/envs/PC/bin/python -u run_polr.py' >/dev/null; then
  echo "a run_polr.py is already active; not starting a second copy" >&2
  ps -eo pid,cmd | grep -E '[/]conda/envs/PC/bin/python -u run_polr.py' || true
  exit 1
fi

LOG="$OUT/logs/${TAG}.log"
echo "launch gpu=${GPU} tag=${TAG} L_p=${PREFIX_LEN} -> ${LOG}"
setsid nohup "$PY" -u run_polr.py \
  --gpus "$GPU" \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --outdir "$OUT" \
  --tag "$TAG" \
  --n-prefixes 32 \
  --prefix-len "$PREFIX_LEN" \
  --distance-threshold 1.0 \
  --feature-dim 10 \
  --temperature 0.6 \
  --top-p 0.9 \
  --top-k -1 \
  --max-tokens 2048 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.88 \
  --physical-window-size 16 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 2048 \
  --cudagraph-capture-sizes 32,16,8,4,2,1 \
  --tensor-parallel-size 1 \
  --vllm-v1 \
  --allow-cudagraph \
  --seed 0 \
  < /dev/null > "$LOG" 2>&1 &
echo "  pid $!"
echo "  log $LOG"

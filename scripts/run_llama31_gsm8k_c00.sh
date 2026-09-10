#!/usr/bin/env bash
# Llama-3.1-8B-Instruct GSM8K PC C=0 ablation on idle GPUs 3 and 4.
# Ablation only: unconditional majority promote. Not Table 1 / not C*.
# Do not touch GPUs 1–2 (MATH-500 / GPQA) or GPU 0 (shared).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/26xyw/.conda/envs/PC/bin/python"
MODEL="/home/26xyw/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
DATA="/home/26xyw/data/Datasets/gsm8k_test.jsonl"
OUT="/home/26xyw/data/path-consistency-results/gsm8k"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

export HF_HOME="${HOME}/data/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
unset VLLM_USE_V1

cd "$ROOT"

# shard 0 -> GPU 3, shard 1 -> GPU 4
for spec in "0:3" "1:4"; do
  shard="${spec%%:*}"
  gpu="${spec##*:}"
  tag="gsm8k_llama31_b32_mt2048_c00_seed0_shard${shard}of2"
  log="$LOGDIR/${tag}.log"
  echo "launch shard=${shard} gpu=${gpu} -> ${log}"
  setsid nohup "$PY" -u run_math.py \
    --gpus "$gpu" \
    --tensor-parallel-size 1 \
    --model "$MODEL" \
    --dataset "$DATA" \
    --outdir "$OUT" \
    --tag "$tag" \
    --methods sc,pc \
    --max-branches 32 \
    --window-size 4 \
    --physical-window-size 4 \
    --beta-threshold 0.0 \
    --temperature 0.6 \
    --top-p 0.9 \
    --top-k -1 \
    --max-tokens 2048 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 2048 \
    --cudagraph-capture-sizes 32,16,8,4,2,1 \
    --vllm-v1 \
    --allow-cudagraph \
    --logprobs 20 \
    --seed 0 \
    --store-text \
    --num-shards 2 \
    --shard-index "$shard" \
    --answer-fallback gsm8k \
    < /dev/null > "$log" 2>&1 &
  echo "  pid $!"
done
echo "logs: $LOGDIR"

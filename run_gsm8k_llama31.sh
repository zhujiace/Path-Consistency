#!/usr/bin/env bash
# Paired SC/PC on GSM8K test with the frozen new-paper PC protocol.
# Shards 1319 questions across GPUs 1-4 (GPU 0 is usually occupied).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
PY="/home/26xyw/.conda/envs/PC/bin/python"
MODEL="/home/26xyw/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
DATA="/home/26xyw/data/Datasets/gsm8k_test.jsonl"
OUT="/home/26xyw/data/path-consistency-results/gsm8k"
LOGDIR="$OUT/logs"
mkdir -p "$LOGDIR"

export HF_HOME="${HOME}/data/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false

cd "$ROOT"
for shard in 0 1 2 3; do
  gpu=$((shard + 1))
  tag="gsm8k_llama31_b32_mt2048_c08_seed0_shard${shard}of4"
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
    --beta-threshold 0.8 \
    --temperature 0.6 \
    --top-p 0.9 \
    --top-k -1 \
    --max-tokens 2048 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --logprobs 20 \
    --seed 0 \
    --store-text \
    --allow-cudagraph \
    --num-shards 4 \
    --shard-index "$shard" \
    --answer-fallback gsm8k \
    < /dev/null > "$log" 2>&1 &
  echo "  pid $!"
done
echo "logs: $LOGDIR"

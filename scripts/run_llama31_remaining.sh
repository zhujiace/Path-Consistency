#!/usr/bin/env bash
# Llama-3.1-8B-Instruct paired SC/PC on remaining main datasets.
# One TP=1 V1 CUDA-graph engine per card on GPUs 1-3 (GPU 4 left alone).
#   GPU 1: MATH-500 490 (exclude engineering 10)
#   GPU 2: GPQA-Diamond 198
#   GPU 3: AIME 2024 then AIME 2025
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/26xyw/.conda/envs/PC/bin/python"
MODEL="/home/26xyw/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
MATH_OUT="/home/26xyw/data/path-consistency-results/math500"
AIME_OUT="/home/26xyw/data/path-consistency-results/aime"
GPQA_OUT="/home/26xyw/data/path-consistency-results/gpqa"

export HF_HOME="${HOME}/data/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
unset VLLM_USE_V1

mkdir -p "$MATH_OUT/logs" "$AIME_OUT/logs" "$GPQA_OUT/logs"
cd "$ROOT"

COMMON=(
  --tensor-parallel-size 1
  --model "$MODEL"
  --methods sc,pc
  --max-branches 32
  --window-size 4
  --physical-window-size 4
  --beta-threshold 0.8
  --temperature 0.6
  --top-p 0.9
  --top-k -1
  --max-tokens 8192
  --max-model-len 10240
  --gpu-memory-utilization 0.92
  --max-num-seqs 16
  --max-num-batched-tokens 2048
  --cudagraph-capture-sizes 32,16,8,4,2,1
  --vllm-v1
  --allow-cudagraph
  --logprobs 20
  --seed 0
)

launch_one() {
  local gpu="$1"
  local tag="$2"
  local dataset="$3"
  local outdir="$4"
  shift 4
  local log="$outdir/logs/${tag}.log"
  echo "launch gpu=${gpu} tag=${tag} -> ${log}"
  setsid nohup "$PY" -u run_math.py \
    --gpus "$gpu" \
    --dataset "$dataset" \
    --outdir "$outdir" \
    --tag "$tag" \
    "${COMMON[@]}" \
    "$@" \
    < /dev/null > "$log" 2>&1 &
  echo "  pid $!"
}

launch_one 1 \
  math500_llama31_b32_mt8192_c08_seed0 \
  /home/26xyw/data/Datasets/math500_test.jsonl \
  "$MATH_OUT" \
  --exclude-engineering-split

launch_one 2 \
  gpqa_diamond_llama31_b32_mt8192_c08_seed0 \
  /home/26xyw/data/Datasets/gpqa_diamond.jsonl \
  "$GPQA_OUT"

# AIME 24 then 25 on GPU 3 so the card stays occupied after year 2024 finishes.
AIME_CHAIN_LOG="$AIME_OUT/logs/aime_llama31_b32_mt8192_c08_seed0_chain.log"
echo "launch gpu=3 AIME24 then AIME25 -> ${AIME_CHAIN_LOG}"
setsid nohup bash -c '
  set -euo pipefail
  PY="$1"
  ROOT="$2"
  MODEL="$3"
  AIME_OUT="$4"
  shift 4
  cd "$ROOT"
  "$PY" -u run_math.py \
    --gpus 3 \
    --dataset /home/26xyw/data/Datasets/aime_2024.jsonl \
    --outdir "$AIME_OUT" \
    --tag aime_2024_llama31_b32_mt8192_c08_seed0 \
    --tensor-parallel-size 1 \
    --model "$MODEL" \
    --methods sc,pc \
    --max-branches 32 \
    --window-size 4 \
    --physical-window-size 4 \
    --beta-threshold 0.8 \
    --temperature 0.6 \
    --top-p 0.9 \
    --top-k -1 \
    --max-tokens 8192 \
    --max-model-len 10240 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 2048 \
    --cudagraph-capture-sizes 32,16,8,4,2,1 \
    --vllm-v1 \
    --allow-cudagraph \
    --logprobs 20 \
    --seed 0
  "$PY" -u run_math.py \
    --gpus 3 \
    --dataset /home/26xyw/data/Datasets/aime_2025.jsonl \
    --outdir "$AIME_OUT" \
    --tag aime_2025_llama31_b32_mt8192_c08_seed0 \
    --tensor-parallel-size 1 \
    --model "$MODEL" \
    --methods sc,pc \
    --max-branches 32 \
    --window-size 4 \
    --physical-window-size 4 \
    --beta-threshold 0.8 \
    --temperature 0.6 \
    --top-p 0.9 \
    --top-k -1 \
    --max-tokens 8192 \
    --max-model-len 10240 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 16 \
    --max-num-batched-tokens 2048 \
    --cudagraph-capture-sizes 32,16,8,4,2,1 \
    --vllm-v1 \
    --allow-cudagraph \
    --logprobs 20 \
    --seed 0
' bash "$PY" "$ROOT" "$MODEL" "$AIME_OUT" < /dev/null > "$AIME_CHAIN_LOG" 2>&1 &
echo "  chain pid $!"
echo "logs: $MATH_OUT/logs $GPQA_OUT/logs $AIME_OUT/logs"

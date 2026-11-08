#!/usr/bin/env bash
# Llama-3.1-8B-Instruct × GSM8K test 1319, new-protocol AC / ESC ± PC.
#
# Table 2 组合块：截断型 parent（AC、ESC）及其 +PC。
# 采样对齐 GSM8K SC/PC：T=0.6, top-p=0.9, max_tokens=2048, B=32, m=4。
# AC 停止阈值 τ=0.95（Adaptive-Consistency 论文默认），不是 PC 前缀门控 C=0.8。
# ESC：当前窗口 4 条答案全部解析且一致则停。
# 默认 GPU 1 跑 ac,pc_ac；GPU 3 跑 esc,pc_esc。
# 不要占 GPU 2（PoLR L_p=256）或 GPU 4（CPT N=32）。未满 1319 不填 Table 2。
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

launch_pair() {
  local gpu="$1"
  local methods="$2"
  local tag="$3"
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
    --beta-threshold 0.8 \
    --early-stop-threshold 0.95 \
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
    < /dev/null > "$log" 2>&1 &
  echo "  pid $!"
}

case "$PAIR" in
  ac)
    launch_pair "$GPU_AC" "ac,pc_ac" "gsm8k_llama31_b32_mt2048_ac_seed0"
    ;;
  esc)
    launch_pair "$GPU_ESC" "esc,pc_esc" "gsm8k_llama31_b32_mt2048_esc_seed0"
    ;;
  both)
    launch_pair "$GPU_AC" "ac,pc_ac" "gsm8k_llama31_b32_mt2048_ac_seed0"
    launch_pair "$GPU_ESC" "esc,pc_esc" "gsm8k_llama31_b32_mt2048_esc_seed0"
    ;;
  *)
    echo "PAIR must be ac, esc, or both" >&2
    exit 1
    ;;
esac
echo "logs: $OUT/logs"

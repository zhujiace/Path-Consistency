#!/usr/bin/env bash
# Launch DeepSeek-V4-Flash paired SC/PC on AIME 2024, AIME 2025, and GPQA-Diamond.
# API only; no GPU. C=0.8 is not Dev-Mix C*.
set -euo pipefail

ROOT="/home/26xyw/projects/path-consistency"
PYTHON="/home/26xyw/.conda/envs/PC/bin/python"
SMOKE="/home/26xyw/data/path-consistency-results/math500/reports/deepseek_smoke.json"
AIME_OUT="/home/26xyw/data/path-consistency-results/aime"
GPQA_OUT="/home/26xyw/data/path-consistency-results/gpqa"

export HF_HOME="${HF_HOME:-$HOME/data/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

mkdir -p "$AIME_OUT/logs" "$GPQA_OUT/logs"

cd "$ROOT"

launch() {
  local tag="$1"
  local dataset="$2"
  local outdir="$3"
  local user="$4"
  local logfile="$5"
  setsid nohup "$PYTHON" run_math_deepseek.py \
    --dataset "$dataset" \
    --outdir "$outdir" \
    --tag "$tag" \
    --methods sc,pc \
    --max-branches 32 \
    --window-size 4 \
    --beta-threshold 0.8 \
    --max-tokens 32768 \
    --reasoning-effort high \
    --question-concurrency 4 \
    --max-inflight 16 \
    --user-id "$user" \
    --smoke-path "$SMOKE" \
    < /dev/null >> "$logfile" 2>&1 &
  echo "started $tag pid=$! log=$logfile"
}

launch \
  aime_2024_deepseek_v4_flash_b32_c08_rep0 \
  /home/26xyw/data/Datasets/aime_2024.jsonl \
  "$AIME_OUT" \
  pc-aime2024-rep0 \
  "$AIME_OUT/logs/aime_2024_deepseek_v4_flash_b32_c08_rep0.log"

launch \
  aime_2025_deepseek_v4_flash_b32_c08_rep0 \
  /home/26xyw/data/Datasets/aime_2025.jsonl \
  "$AIME_OUT" \
  pc-aime2025-rep0 \
  "$AIME_OUT/logs/aime_2025_deepseek_v4_flash_b32_c08_rep0.log"

launch \
  gpqa_diamond_deepseek_v4_flash_b32_c08_rep0 \
  /home/26xyw/data/Datasets/gpqa_diamond.jsonl \
  "$GPQA_OUT" \
  pc-gpqa-rep0 \
  "$GPQA_OUT/logs/gpqa_diamond_deepseek_v4_flash_b32_c08_rep0.log"

#!/usr/bin/env bash
# Re-run DeepSeek-V4-Flash GSM8K PC at C=0.8 with vote keys aligned to the
# numeric regrade (currency/units/duration). Reuses existing C=0.8 SC roots.
# Does not regenerate SC. Does not kill the C=0.7 ablation.
# Not Table 1 until 1319 PC rows exist.
set -euo pipefail

ROOT="/home/26xyw/projects/path-consistency"
PYTHON="/home/26xyw/.conda/envs/PC/bin/python"
OUT="/home/26xyw/data/path-consistency-results/gsm8k"
SMOKE="/home/26xyw/data/path-consistency-results/math500/reports/deepseek_smoke.json"
TAG="gsm8k_deepseek_v4_flash_b32_c08_pc_vote_v2"

CACHE_FIRST100="$OUT/gsm8k_deepseek_v4_flash_b32_c08_first100_rep0_4829b0ab9213_root_cache"
CACHE_NEXT500="$OUT/gsm8k_deepseek_v4_flash_b32_c08_next500_rep0_7028521973e2_root_cache"
CACHE_REST719="$OUT/gsm8k_deepseek_v4_flash_b32_c08_rest719_rep0_455c760d2e21_root_cache"

export HF_HOME="${HF_HOME:-$HOME/data/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

mkdir -p "$OUT/logs"
cd "$ROOT"

if pgrep -f "python -u run_math_deepseek.py .*${TAG}" >/dev/null; then
  echo "this PC vote-v2 tag is already running" >&2
  pgrep -af "python -u run_math_deepseek.py .*${TAG}" || true
  exit 1
fi

LOG="$OUT/logs/${TAG}.log"
echo "launch $TAG -> $LOG"
setsid nohup "$PYTHON" -u run_math_deepseek.py \
  --dataset /home/26xyw/data/Datasets/gsm8k_test.jsonl \
  --outdir "$OUT" \
  --tag "$TAG" \
  --methods pc \
  --max-branches 32 \
  --window-size 4 \
  --beta-threshold 0.8 \
  --max-tokens 32768 \
  --reasoning-effort high \
  --question-concurrency 2 \
  --max-inflight 8 \
  --user-id pc-gsm8k-c08-pc-vote-v2 \
  --smoke-path "$SMOKE" \
  --reuse-root-cache "$CACHE_FIRST100" \
  --reuse-root-cache "$CACHE_NEXT500" \
  --reuse-root-cache "$CACHE_REST719" \
  < /dev/null >> "$LOG" 2>&1 &
echo "started $TAG pid=$! log=$LOG"

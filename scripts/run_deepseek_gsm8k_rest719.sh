#!/usr/bin/env bash
# DeepSeek-V4-Flash GSM8K remaining 719 (source_row 600–1318). API only.
# C=0.8 is the a priori default, not C*. Do not start a second copy.
set -euo pipefail

ROOT="/home/26xyw/projects/path-consistency"
PYTHON="/home/26xyw/.conda/envs/PC/bin/python"
OUT="/home/26xyw/data/path-consistency-results/gsm8k"
SMOKE="/home/26xyw/data/path-consistency-results/math500/reports/deepseek_smoke.json"
TAG="gsm8k_deepseek_v4_flash_b32_c08_rest719_rep0"
ROWS="$(seq -s, 600 1318)"

export HF_HOME="${HF_HOME:-$HOME/data/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1

mkdir -p "$OUT/logs"
cd "$ROOT"

if pgrep -f '[r]un_math_deepseek.py' >/dev/null; then
  echo "a DeepSeek run is already active; not starting a second copy" >&2
  pgrep -a -f '[r]un_math_deepseek.py' || true
  exit 1
fi

setsid nohup "$PYTHON" -u run_math_deepseek.py \
  --dataset /home/26xyw/data/Datasets/gsm8k_test.jsonl \
  --outdir "$OUT" \
  --tag "$TAG" \
  --methods sc,pc \
  --source-rows "$ROWS" \
  --max-branches 32 \
  --window-size 4 \
  --beta-threshold 0.8 \
  --max-tokens 32768 \
  --reasoning-effort high \
  --question-concurrency 4 \
  --max-inflight 16 \
  --user-id pc-gsm8k-rest719-rep0 \
  --smoke-path "$SMOKE" \
  < /dev/null >> "$OUT/logs/${TAG}.log" 2>&1 &
echo "started $TAG pid=$! log=$OUT/logs/${TAG}.log"

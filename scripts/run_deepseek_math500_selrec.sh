#!/usr/bin/env bash
# DeepSeek MATH-500 选择 × recovery 消融。复用主表 SC 根，不覆盖主表 PC。
# B: uniform + recovery on
# C: BLC + recovery off
# D: uniform + recovery off
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/26xyw/.conda/envs/PC/bin/python"
OUT="/home/26xyw/data/path-consistency-results/math500"
SMOKE="/home/26xyw/data/path-consistency-results/math500/reports/deepseek_smoke.json"
CACHE="$OUT/math500_deepseek_v4_flash_b32_c08_rep0_4856d4ef5673_root_cache"
DATASET="/home/26xyw/data/Datasets/math500_test.jsonl"
ARM="${ARM:-both}"

export HF_HOME="${HF_HOME:-$HOME/data/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT/logs"
cd "$ROOT"

if [[ ! -d "$CACHE" ]]; then
  echo "missing SC root cache: $CACHE" >&2
  exit 1
fi

launch() {
  local tag="$1"
  shift
  local log="$OUT/logs/${tag}.log"
  if pgrep -f "/.conda/envs/PC/bin/python -u run_math_deepseek.py .*--tag ${tag}" >/dev/null; then
    echo "already running: ${tag}" >&2
    return 0
  fi
  echo "launch ${tag} -> ${log}"
  setsid nohup "$PY" -u run_math_deepseek.py \
    --dataset "$DATASET" \
    --outdir "$OUT" \
    --tag "$tag" \
    --methods pc \
    --max-branches 32 \
    --window-size 4 \
    --beta-threshold 0.8 \
    --max-tokens 32768 \
    --reasoning-effort high \
    --logprobs 20 \
    --question-concurrency 2 \
    --max-inflight 8 \
    --seed 0 \
    --smoke-path "$SMOKE" \
    --reuse-root-cache "$CACHE" \
    "$@" \
    < /dev/null >> "$log" 2>&1 &
  echo "  pid $!"
}

case "$ARM" in
  b|B)
    launch "math500_deepseek_selrec_rand_on_rep0" \
      --selector uniform \
      --recovery on \
      --selection-seed 0 \
      --user-id pc-math500-selrec-b
    ;;
  c|C)
    launch "math500_deepseek_selrec_blc_off_rep0" \
      --selector blc \
      --recovery off \
      --user-id pc-math500-selrec-c
    ;;
  d|D)
    launch "math500_deepseek_selrec_rand_off_rep0" \
      --selector uniform \
      --recovery off \
      --selection-seed 0 \
      --user-id pc-math500-selrec-d
    ;;
  both)
    launch "math500_deepseek_selrec_rand_on_rep0" \
      --selector uniform \
      --recovery on \
      --selection-seed 0 \
      --user-id pc-math500-selrec-b
    launch "math500_deepseek_selrec_blc_off_rep0" \
      --selector blc \
      --recovery off \
      --user-id pc-math500-selrec-c
    ;;
  *)
    echo "ARM must be b, c, d, or both" >&2
    exit 1
    ;;
esac
echo "logs: $OUT/logs"

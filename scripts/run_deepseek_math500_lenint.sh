#!/usr/bin/env bash
# DeepSeek MATH-500 matched-history 长度干预。
# 冻 4856d4ef5673 window-0 BLC source；r∈{0,0.2,0.4,0.6,0.8}；尾巴用 SC cache w2–7。
# 不是 PVC 控制器，不覆盖主表 JSONL。
# MODE=snapshot|smoke|full
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/26xyw/.conda/envs/PC/bin/python"
OUT="/home/26xyw/data/path-consistency-results/math500"
SMOKE="/home/26xyw/data/path-consistency-results/math500/reports/deepseek_smoke.json"
CACHE="$OUT/math500_deepseek_v4_flash_b32_c08_rep0_4856d4ef5673_root_cache"
DATASET="/home/26xyw/data/Datasets/math500_test.jsonl"
MODE="${MODE:-full}"
TAG_FULL="math500_deepseek_lenint_r8_seed0"
TAG_SMOKE="math500_deepseek_lenint_smoke"

export HF_HOME="${HF_HOME:-$HOME/data/huggingface}"
export HF_HUB_DISABLE_XET=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

mkdir -p "$OUT/logs" "$OUT/reports"
cd "$ROOT"

if [[ ! -d "$CACHE" ]]; then
  echo "missing SC root cache: $CACHE" >&2
  exit 1
fi

run_lenint() {
  local tag="$1"
  shift
  "$PY" -u run_math_deepseek_lenint.py \
    --dataset "$DATASET" \
    --outdir "$OUT" \
    --tag "$tag" \
    --window-size 4 \
    --beta-threshold 0.8 \
    --max-tokens 32768 \
    --reasoning-effort high \
    --logprobs 20 \
    --selection-seed 0 \
    --seed 0 \
    --smoke-path "$SMOKE" \
    --reuse-root-cache "$CACHE" \
    --common-support-only \
    "$@"
}

case "$MODE" in
  snapshot)
    run_lenint "$TAG_FULL" --snapshot-only --user-id pc-math500-lenint-e0
    ;;
  smoke)
    e0="$(ls -t "$OUT"/reports/${TAG_FULL}_*_e0.json 2>/dev/null | head -1 || true)"
    if [[ -z "$e0" ]]; then
      run_lenint "$TAG_FULL" --snapshot-only --user-id pc-math500-lenint-e0
      e0="$(ls -t "$OUT"/reports/${TAG_FULL}_*_e0.json | head -1)"
    fi
    row="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))['first_common_source_rows'][0])" "$e0")"
    echo "smoke source_row=$row from $e0"
    run_lenint "$TAG_SMOKE" \
      --source-rows "$row" \
      --replicates 1 \
      --question-concurrency 1 \
      --max-inflight 4 \
      --user-id pc-math500-lenint-smoke
    ;;
  full)
    tag="$TAG_FULL"
    log="$OUT/logs/${tag}.log"
    if pgrep -f "/.conda/envs/PC/bin/python -u run_math_deepseek_lenint.py .*--tag ${tag}" >/dev/null; then
      echo "already running: ${tag}" >&2
      exit 0
    fi
    echo "launch ${tag} -> ${log}"
    setsid nohup "$PY" -u run_math_deepseek_lenint.py \
      --dataset "$DATASET" \
      --outdir "$OUT" \
      --tag "$tag" \
      --replicates 8 \
      --window-size 4 \
      --beta-threshold 0.8 \
      --max-tokens 32768 \
      --reasoning-effort high \
      --logprobs 20 \
      --selection-seed 0 \
      --seed 0 \
      --question-concurrency 2 \
      --max-inflight 8 \
      --smoke-path "$SMOKE" \
      --reuse-root-cache "$CACHE" \
      --common-support-only \
      --user-id pc-math500-lenint \
      < /dev/null >> "$log" 2>&1 &
    echo "  pid $!"
    echo "log: $log"
    ;;
  *)
    echo "MODE must be snapshot, smoke, or full" >&2
    exit 1
    ;;
esac

#!/usr/bin/env bash
# Wait for rest719 to finish, then merge first100 + next500 + rest719 into 1319.
set -euo pipefail

ROOT="/home/26xyw/projects/path-consistency"
PYTHON="/home/26xyw/.conda/envs/PC/bin/python"
OUT="/home/26xyw/data/path-consistency-results/gsm8k"
LOG="$OUT/logs/merge_deepseek_gsm8k_full1319.log"

cd "$ROOT"
mkdir -p "$OUT/logs"

status_of() {
  "$PYTHON" - <<'PY'
import json
from pathlib import Path
out = Path("/home/26xyw/data/path-consistency-results/gsm8k")
paths = sorted(out.glob("gsm8k_deepseek_v4_flash_b32_c08_rest719_rep0_*.run.json"))
if not paths:
    print("missing")
else:
    rec = json.loads(paths[-1].read_text())
    print(rec.get("status") or "unknown")
PY
}

while true; do
  st="$(status_of)"
  echo "$(date '+%F %T') rest719 status=$st"
  case "$st" in
    complete)
      "$PYTHON" -u scripts/merge_deepseek_gsm8k.py | tee -a "$LOG"
      exit 0
      ;;
    failed|error)
      echo "rest719 $st; not merging" >&2
      exit 1
      ;;
  esac
  sleep 60
done

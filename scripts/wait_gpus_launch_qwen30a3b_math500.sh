#!/usr/bin/env bash
# Every 5 minutes, check GPUs 1-4. When all are free (used ≤ 3500 MiB),
# launch Qwen3-30B-A3B MATH-500 490 paired SC/PC. Does not kill other users.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="/home/26xyw/.conda/envs/PC/bin/python"
OUT="/home/26xyw/data/path-consistency-results/math500"
TAG="math500_qwen3_30b_a3b_b32_mt32768_c08_seed0"
LOGDIR="$OUT/logs"
LOG="$LOGDIR/wait_gpus_launch_qwen30a3b.log"
LOCK="$LOGDIR/wait_gpus_launch_qwen30a3b.lock"
INTERVAL="${INTERVAL:-300}"
GPUS="1,2,3,4"

mkdir -p "$LOGDIR"
exec >>"$LOG" 2>&1
exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date '+%F %T %Z') another waiter holds $LOCK; exit"
  exit 0
fi

log() { echo "$(date '+%F %T %Z') $*"; }

already_running() {
  pgrep -af "run_math.py" | grep -F -- "$TAG" | grep -v grep >/dev/null
}

already_complete() {
  "$PY" - "$OUT" "$TAG" <<'PY'
import json, sys
from pathlib import Path
out, tag = Path(sys.argv[1]), sys.argv[2]
for path in out.glob(f"{tag}_*.run.json"):
    try:
        rec = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    if rec.get("status") == "complete":
        raise SystemExit(0)
    summary = path.with_name(path.name.replace(".run.json", ".summary.json"))
    if summary.exists():
        try:
            payload = json.loads(summary.read_text())
        except json.JSONDecodeError:
            payload = {}
        n = (payload.get("methods") or {}).get("SC", {}).get("questions")
        if n == 490:
            raise SystemExit(0)
raise SystemExit(1)
PY
}

gpus_free() {
  local json
  json="$("$PY" "$ROOT/check_gpus_a3b.py")"
  log "gpu_check $json"
  printf '%s' "$json" | "$PY" -c 'import json,sys; d=json.load(sys.stdin); raise SystemExit(0 if d.get("free_enough") else 1)'
}

if already_complete; then
  log "MATH-500 $TAG already complete; nothing to launch"
  exit 0
fi
if already_running; then
  log "MATH-500 $TAG already running; nothing to launch"
  pgrep -af "run_math.py" | grep -F -- "$TAG" | grep -v grep || true
  exit 0
fi

log "waiter start interval=${INTERVAL}s gpus=$GPUS tag=$TAG"

while true; do
  if already_complete; then
    log "MATH-500 $TAG completed while waiting; exit"
    exit 0
  fi
  if already_running; then
    log "MATH-500 $TAG started elsewhere; exit"
    exit 0
  fi
  if gpus_free; then
    log "GPUs $GPUS free; re-check then launch"
    sleep 5
    if ! gpus_free; then
      log "lost the window; keep waiting"
    elif already_running; then
      log "MATH-500 $TAG started during re-check; exit"
      exit 0
    else
      cd "$ROOT"
      launch_out="$(bash "$ROOT/run_math500_qwen30a3b.sh")"
      log "launch_output $launch_out"
      sleep 8
      if already_running; then
        log "MATH-500 launched on GPUs $GPUS"
        exit 0
      fi
      log "launch did not leave a living run_math.py; keep waiting"
    fi
  else
    log "GPUs $GPUS not free; sleep ${INTERVAL}s"
  fi
  sleep "$INTERVAL"
done

#!/usr/bin/env bash
# Sequential single-GPU DeepConf then CPT smoke. Do not occupy GPUs 1-4 together.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PC_PY="/home/26xyw/.conda/envs/PC/bin/python"
SMOKE_PY="${SMOKE_PY:-/home/26xyw/data/envs/deepconf-cpt-smoke/bin/python}"
MODEL="/home/26xyw/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
DATA="/home/26xyw/data/Datasets/gsm8k_test.jsonl"
OUT="/home/26xyw/data/path-consistency-results/deepconf-cpt"
CPT_ROOT="/home/26xyw/data/path-consistency-baselines/CPT"
EMBED="${EMBED:-sentence-transformers/all-MiniLM-L6-v2}"
GPU="${GPU:-2}"
N="${N:-10}"
TAG_DC="${TAG_DC:-deepconf_llama31_gsm8k_smoke10}"
CPT_OUT="${OUT}/cpt_llama31_gsm8k_smoke10"

export HF_HOME="${HOME}/data/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU}"
export VLLM_USE_V1=0

mkdir -p "$OUT/logs"
cd "$ROOT"

echo "[smoke] GPU=${GPU} n=${N} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

echo "[smoke] DeepConf..."
"$PC_PY" -u "$ROOT/run_deepconf_gsm8k_smoke.py" \
  --gpu "$GPU" \
  --model "$MODEL" \
  --dataset "$DATA" \
  --outdir "$OUT" \
  --tag "$TAG_DC" \
  --max-questions "$N" \
  --sample-seed 2027 \
  --seed 0 \
  --mode online \
  --warmup-traces 8 \
  --total-budget 16 \
  --budget 16 \
  --confidence-percentile 10 \
  --window-size 2048 \
  --temperature 0.6 \
  --top-p 0.95 \
  --max-tokens 1024 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --logprobs 20

SUBSET="$OUT/${TAG_DC}_subset.jsonl"
if [[ ! -f "$SUBSET" ]]; then
  echo "DeepConf subset JSONL missing: $SUBSET" >&2
  exit 1
fi

if [[ ! -x "$SMOKE_PY" ]]; then
  echo "[smoke] CPT isolated python missing: $SMOKE_PY" >&2
  echo "[smoke] writing CPT blocked note; DeepConf finished" >&2
  "$PC_PY" - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
(out / "cpt_blocked.json").write_text(json.dumps({
    "status": "blocked",
    "reason": "isolated env python missing",
    "fair_comparison": False,
}, ensure_ascii=False, indent=2))
PY
  exit 0
fi

echo "[smoke] CPT official qwen/math.py on Llama (NOT a fair comparison)..."
export PYTHONPATH="${CPT_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"
export http_proxy="${http_proxy:-http://127.0.0.1:27890}"
export https_proxy="${https_proxy:-http://127.0.0.1:27890}"
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:27890}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:27890}"
# Official python code/qwen/math.py shadows stdlib math. Llama-3.1 also
# defaults to max_model_len=131072, which exceeds one 4090 KV. The adapter
# runs the official module after those two engine/import fixes only.
set +e
"$SMOKE_PY" -u "$ROOT/scripts/run_cpt_llama_adapter.py" \
  --gpu "$GPU" \
  --model "$MODEL" \
  --input "$SUBSET" \
  --output "${CPT_OUT}/results" \
  --embedding-model-path "$EMBED" \
  --embedding-device cpu \
  --n-completions 4 \
  --num-workers 4 \
  --batch-size 1 \
  --tensor-parallel-size 1 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --max-tokens 1024 \
  --chunk-tokens 256 \
  --chunk-dynamic-mode fixed \
  --chunk-tokens-fixed 256 \
  --write-tokens 128 \
  --bb-max-items 256 \
  --bb-random-seed 0 \
  --bb-sim-threshold 0.75 \
  --bb-broadcast-select-mode randomk \
  --bb-broadcast-select-k 32 \
  --start-idx 0 \
  --end-idx "$N"
CPT_RC=$?
set -e
echo "[smoke] CPT exit=${CPT_RC}"
echo "$CPT_RC" > "${OUT}/cpt_exit_code.txt"

echo "[smoke] summarize..."
"$PC_PY" -u "$ROOT/scripts/summarize_deepconf_cpt_smoke.py"
echo "[smoke] done"

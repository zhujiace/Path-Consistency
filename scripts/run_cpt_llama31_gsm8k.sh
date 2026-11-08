#!/usr/bin/env bash
# Llama-3.1-8B-Instruct × GSM8K test 1319, official CPT qwen/math.py.
# GPU 4 only. Table 2 CPT is N=32, matching PC B=32 and PoLR N=32.
# SC (B=20) is the matched-token SC baseline for PC, not CPT's worker count.
# N=20 (cpt_llama31_gsm8k_n20) is complete but not Table 2.
# Do not fill Table 2 from the 10-question smoke.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PC_PY="/home/26xyw/.conda/envs/PC/bin/python"
CPT_PY="${CPT_PY:-/home/26xyw/data/envs/deepconf-cpt-smoke/bin/python}"
MODEL="/home/26xyw/data/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
RAW="/home/26xyw/data/Datasets/gsm8k_test.jsonl"
INPUT="/home/26xyw/data/Datasets/gsm8k_test_cpt.jsonl"
OUT="/home/26xyw/data/path-consistency-results/deepconf-cpt"
CPT_ROOT="/home/26xyw/data/path-consistency-baselines/CPT"
EMBED="${EMBED:-sentence-transformers/all-MiniLM-L6-v2}"
GPU="${GPU:-4}"
N_WORKERS="${N_WORKERS:-32}"
TAG="${TAG:-cpt_llama31_gsm8k_n32}"
CPT_OUT="${OUT}/${TAG}"

export HF_HOME="${HOME}/data/huggingface"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${GPU}"
export VLLM_USE_V1=0
export PYTHONUNBUFFERED=1
export HTTP_PROXY="${HTTP_PROXY:-http://127.0.0.1:27890}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://127.0.0.1:27890}"
export http_proxy="${http_proxy:-http://127.0.0.1:27890}"
export https_proxy="${https_proxy:-http://127.0.0.1:27890}"
# 2048 matches SC decode budget but CPT prepends blackboard; Q2 died at prompt 2050.
export PC_CPT_MAX_MODEL_LEN="${PC_CPT_MAX_MODEL_LEN:-4096}"
export PC_CPT_GPU_MEM="${PC_CPT_GPU_MEM:-0.90}"
export PC_CPT_MAX_NUM_SEQS="${PC_CPT_MAX_NUM_SEQS:-32}"
export PC_CPT_ENFORCE_EAGER="${PC_CPT_ENFORCE_EAGER:-1}"

mkdir -p "$OUT/logs" "$CPT_OUT"
cd "$ROOT"

if pgrep -f '/python.*run_cpt_llama_adapter.py' >/dev/null; then
  echo "a CPT Llama adapter is already active; not starting a second copy" >&2
  pgrep -a -f '/python.*run_cpt_llama_adapter.py' || true
  exit 1
fi

if [[ ! -x "$CPT_PY" ]]; then
  echo "CPT python missing: $CPT_PY" >&2
  exit 1
fi

if [[ ! -f "$INPUT" ]]; then
  echo "[cpt] writing $INPUT from $RAW"
  "$PC_PY" - "$RAW" "$INPUT" <<'PY'
import json, sys
from pathlib import Path
from path_consistency.math_eval import format_gsm8k_gold
src, dest = Path(sys.argv[1]), Path(sys.argv[2])
with src.open() as handle, dest.open("w") as out:
    for index, line in enumerate(handle):
        if not line.strip():
            continue
        row = json.loads(line)
        out.write(json.dumps({
            "question_id": f"gsm8k_{index:04d}",
            "question": row["question"],
            "answer": format_gsm8k_gold(row["target"]),
        }, ensure_ascii=False) + "\n")
print(f"wrote {dest}")
PY
fi

echo "[cpt] GPU=${GPU} N=${N_WORKERS} tag=${TAG} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv

export PYTHONPATH="${CPT_ROOT}/code${PYTHONPATH:+:${PYTHONPATH}}"

setsid nohup "$CPT_PY" -u "$ROOT/scripts/run_cpt_llama_adapter.py" \
  --gpu "$GPU" \
  --model "$MODEL" \
  --input "$INPUT" \
  --output "${CPT_OUT}/results" \
  --embedding-model-path "$EMBED" \
  --embedding-device cpu \
  --n-completions "$N_WORKERS" \
  --num-workers "$N_WORKERS" \
  --batch-size 1 \
  --tensor-parallel-size 1 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --max-tokens 2048 \
  --chunk-tokens 256 \
  --chunk-dynamic-mode fixed \
  --chunk-tokens-fixed 256 \
  --write-tokens 128 \
  --enable-dynamic-broadcast-trend \
  --tau-start 0.40 \
  --tau-stop 0.10 \
  --bb-max-items 256 \
  --bb-random-seed 0 \
  --bb-sim-threshold 0.75 \
  --bb-broadcast-select-mode randomk \
  --bb-broadcast-select-k 32 \
  --start-idx 0 \
  --end-idx 1319 \
  < /dev/null >> "$OUT/logs/${TAG}.log" 2>&1 &
echo "started $TAG pid=$! log=$OUT/logs/${TAG}.log"
echo $! > "$OUT/logs/${TAG}.pid"

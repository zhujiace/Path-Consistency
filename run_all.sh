#!/bin/bash
# Run the 5 GSM8K configs in parallel, one per GPU.
set -e
PY=/home/26xyw/.conda/envs/PC/bin/python
export HF_HOME=$HOME/data/huggingface
export HF_HUB_DISABLE_XET=1
export CUDA_CACHE_DIR=$HOME/.cache/cuda

cd "$(dirname "$0")"
mkdir -p results logs

run() {
    local gpu=$1; local tag=$2; shift 2
    $PY run_gsm8k.py --gpu "$gpu" "$@" > "logs/$tag.log" 2>&1
}

run 0 sc    --method sc --seed 0 &
run 1 pc0   --method pc --confidence 0.0 --seed 0 &
run 2 pc07  --method pc --confidence 0.7 --seed 0 &
run 3 pc08  --method pc --confidence 0.8 --seed 0 &
run 4 pc09  --method pc --confidence 0.9 --seed 0 &

wait
echo "All configs done."
$PY analyze_results.py

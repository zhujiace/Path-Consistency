# Path-consistency API experiments

Requires Python 3.10 or later. No GPU is required. Includes the DeepSeek API client, self-consistency and path-consistency, prefix selection and recovery, paired evaluation, and answer grading. Python comments and docstrings have been removed.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export DEEPSEEK_API_KEY='your-api-key'
export DEEPSEEK_MODEL='deepseek-v4-flash'
```

The API must support thinking and assistant prefix completion; an arbitrary chat-compatible endpoint is not sufficient. The client uses the DeepSeek beta endpoint by default. Credentials can also be stored in a local `.env` file. Never share or commit that file.

## Data

Provide a JSONL file containing one problem per line, using one task format per file. These are synthetic format examples, not benchmark data.

| Task | JSON record |
|---|---|
| GSM8K | `{"question":"What is 2 + 2?","target":4}` |
| MATH | `{"problem":"Compute 2 + 2.","answer":"4"}` |
| AIME | `{"task":"aime","year":2025,"problem":"Compute 2 + 2.","answer":"4"}` |
| GPQA | `{"task":"gpqa","question":"What is 2 + 2?","choices":{"A":"1","B":"2","C":"3","D":"4"},"answer":"D"}` |

Convert the official GSM8K `answer` field to a numeric `target` before running. For GPQA, prepare the choice ordering and corresponding correct letter. Benchmark data, credentials, and experimental results are not included.

## Running experiments

First check prefix-completion support, then run paired evaluation. Both commands below make real API requests and incur charges.

```bash
python run_math_deepseek.py --smoke --outdir results
python run_math_deepseek.py --dataset data.jsonl --outdir results --smoke-path results/reports/deepseek_smoke.json --methods sc,pc --selector blc --tag example
```

Defaults are 32 trajectories, a sampling window of 4, agreement threshold 0.8, and fixed prefix levels of 0%, 20%, 40%, 60%, and 80%. Thinking and recovery are enabled. `blc` is the implementation name for local confidence selection. If the API does not return log probabilities, explicitly use `--selector uniform` for random selection. The original `--selector auto` option may select token-medoid when log probabilities are unavailable; this is not the paper's default setting.

For a small initial run, add `--max-questions 1 --question-concurrency 1 --max-inflight 4`. MATH-format runs exclude the paper's ten engineering-debugging questions by default. Add `--include-engineering-split` when using your own non-MATH-500 data.

Outputs in `results/` include per-question JSONL files, summary JSON files, run configurations, and root-window caches. Restarting with the same configuration skips completed questions. Use a new `--tag` when changing data or parameters. Add `--store-text` to retain generated text. For timing comparisons, use `generation_seconds`, which includes reused root-window costs, not `actual_generation_seconds`. The API does not expose a controllable sampling seed; `--seed` primarily identifies repetitions.

## Implementation coverage

The main path-consistency logic is included in `path_consistency/math_protocol.py`: windowed sampling, the top-1-versus-all-rest agreement gate, continuous prefix extension, local confidence selection, trial validation and commitment, one-level rollback, disabling prefix reuse after two consecutive failures, and final voting over all valid answers. Failed-window answers are excluded from subsequent agreement checks, but remain in the final vote.

`deepseek_inference.py` connects the controller to API generations and token log probabilities. `deepseek_api.py` handles requests and prefix completion. `run_math_deepseek.py` provides paired root reuse, resumable evaluation, grading, and result export.

This is the API implementation, not the complete experimental repository. It does not include the local vLLM backend, all comparison-method implementations, plotting scripts, or model-assisted answer-format regrading. The included programmatic grader may therefore differ from separately regraded paper results.

## Offline checks

```bash
python run_math_deepseek.py --help
python -m unittest discover -s tests -v
```

Tests use simulated responses and do not call the real API.

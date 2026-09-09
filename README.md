# Path-Consistency

Code for reproducing and extending **Path-Consistency**: reuse a high-confidence reasoning prefix so later self-consistency branches spend fewer tokens.

## Paper

Jiace Zhu, Yuanzhe Huang, Yingtao Shen, Jie Zhao, An Zou.  
**Path-Consistency with Prefix Enhancement for Efficient Inference in LLMs.**  
[[arXiv](https://arxiv.org/abs/2409.01281)] [[pdf](https://arxiv.org/pdf/2409.01281)]

```bibtex
@misc{zhu2025pathconsistencyprefixenhancementefficient,
      title={Path-Consistency with Prefix Enhancement for Efficient Inference in LLMs},
      author={Jiace Zhu and Yuanzhe Huang and Yingtao Shen and Jie Zhao and An Zou},
      year={2025},
      eprint={2409.01281},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2409.01281},
}
```

Agent notes: [`AGENTS.md`](AGENTS.md). **Qwen3 new-server handoff:** [`handoff.md`](handoff.md).

Two machines (2026-09-09): the 4090 lab keeps Llama-3.1-8B and DeepSeek API runs; the new GPU server runs **Qwen3-30B-A3B** only. Do not mix old Sentence-PC (`results/SC_seed0.json`, B=20) with the frozen B=32 paired-root v2 protocol. C=0.8 is provisional, not \(C^\*\).

---

Reproduce the original GSM8K case study using Llama-3-8B with vLLM. That table is **appendix continuity**, not the ICLR 2027 main table.

## Setup

```bash
conda create -n PC python=3.10 -y
conda activate PC
pip install torch==2.6.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install vllm==0.8.5+cu118 -f https://vllm-wheels.s3.us-west-2.amazonaws.com/0.8.5/cu118/vllm/index.html
pip install transformers==4.51.3 datasets scipy
```

Environment variables:

```bash
export HF_HOME=~/data/huggingface          # models & datasets live under ~/data
export HF_HUB_DISABLE_XET=1                # proxy-friendly downloads
```

## Experiment config (from the paper)

- Model: `meta-llama/Meta-Llama-3-8B` (base, few-shot CoT)
- Sampling: temperature 0.6, top-p 0.9, 20 branches per question
- Path-consistency: max prefix level 3  =>  4 windows of 5 branches each
- Beta confidence criterion (Aggarwal et al. 2023), thresholds C = 0 / 0.7 / 0.8 / 0.9
- Prefix extraction every 5 branches; the prefix (first `level` sentences of a
  majority-answer path) is prepended to the prompt for the next window
- vLLM parallelism: all branches of one window share the same prompt and are
  generated in one batched call (n = window size)

## Run

```bash
# SC baseline (equivalent to PC with threshold 1.0)
python run_gsm8k.py --method sc   --seed 0 --gpu 0

# PC with different confidence thresholds
python run_gsm8k.py --method pc --confidence 0.0  --seed 0 --gpu 1
python run_gsm8k.py --method pc --confidence 0.7  --seed 0 --gpu 2
python run_gsm8k.py --method pc --confidence 0.8  --seed 0 --gpu 3
python run_gsm8k.py --method pc --confidence 0.9  --seed 0 --gpu 4

# or all 5 configs in parallel on 5 GPUs:
bash run_all.sh
```

Results (JSON + CSV) are written to `results/`.

## Results (GSM8K test set, 1319 questions, Llama-3-8B, seed 0)

Reproduction of the paper's Table 1 (Speedup/Decrease per prefix level):

| Method | Acc (%) | TotSpd (%) | L1 (%) | L2 (%) | L3 (%) | Tokens (#) | TotDec (%) | L1Dec (%) | L2Dec (%) | L3Dec (%) |
|---|---|---|---|---|---|---|---|---|---|---|
| SC     | 67.0    | 0.0        | 0.0    | 0.0    | 0.0    | 2035.3     | 0.0        | 0.0       | 0.0       | 0.0       |
| PC (C=0)   | 65.6 | +18.6 | +14.0 | +24.2 | +35.8 | 1616.3 | -20.6 | -15.3 | -27.9 | -38.9 |
| PC (C=0.7) | 66.6 | +12.2 | +8.5  | +14.7 | +24.0 | 1754.3 | -13.8 | -8.6  | -18.7 | -27.7 |
| PC (C=0.8) | 67.2 | +12.0 | +9.2  | +14.1 | +24.0 | 1775.3 | -12.8 | -8.9  | -16.8 | -25.4 |
| PC (C=0.9) | 68.3 | +7.8  | +3.8  | +10.8 | +18.0 | 1851.6 | -9.0  | -4.6  | -12.1 | -20.3 |

Paper reference values: SC 64.1% / 1932 tokens; PC C=0: 66.6% / -24.0% tokens,
C=0.7: 67.1% / -16.1%, C=0.8: 67.8% / -16.2%, C=0.9: 66.6% / -9.9%.

Findings match the paper qualitatively: (1) prefix reuse reduces tokens and
latency monotonically with prefix level (L1 < L2 < L3) and with aggressiveness
(C=0 > 0.7 ~ 0.8 > 0.9); (2) accuracy is maintained or slightly improved at
conservative thresholds. Absolute latency speedups are lower than the paper's
because vLLM batches the 5 branches of each window in parallel (latency is
then dominated by the longest generation in the batch), while the paper
measured sequential generation.

## Notes on reproducibility

- Models/datasets are stored under `~/data` (`HF_HOME=~/data/huggingface`,
  `~/data/Datasets/*.jsonl`).
- The beta confidence gate, prefix extraction and "prefix correction"
  (discarding the prefix when the gate fails) follow the paper's appendix code
  and case-study behaviour exactly.
- `prepare_data.py` pre-converts datasets to JSONL so the GPU process never
  imports pyarrow: pyarrow 25's jemalloc background thread segfaults when
  racing with CUDA (observed repeatedly).
- Run with `setsid nohup bash run_all.sh` so the experiment survives the
  launching shell session.

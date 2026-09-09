# 当前试验阶段交接（Qwen3 新服务器）

日期：2026-09-09。本文件只服务 **Qwen3-30B-A3B** 实验机。Llama 与 DeepSeek 仍在 4090 实验室跑，不要在这边重做。协议与禁令以 [`AGENTS.md`](AGENTS.md) 为准；方法细节以 `paper/plan_2weeks_ICLR2027.md` 为准。

若你按口头文件名找 `handsoff.md`，正文就是这份 `handoff.md`。

---

## 1. 这一阶段要完成什么

**P0：Qwen3-30B-A3B × MATH-500 主统计 490 题，paired SC/PC，thinking on，C=0.8（临时，不是 \(C^\*\)）。**

成功标准：490 题 JSONL 齐、SC 与 PC 配对、有 `generation_seconds` 与 decode token、manifest 标 complete。准确率相对 SC 的 paired 单侧 95% 下界用来对照 plan 的 −1.0 pp 界，但 **不得**用这 490 题回写选阈值。

做完 MATH-500 再开同一模型的 AIME 2024/2025 与 GPQA-Diamond（同一协议）。公平 DeepConf / DeepConf+PC / 官方 CPT 排在 A3B 主 SC/PC 之后。

不要在这台机器上：

- 跑 Llama-3.1-8B 分片或 `scripts/run_llama31_remaining.sh`
- 配置 DeepSeek API
- 续跑已暂停的 Qwen3-32B V0 eager（fingerprint `9f54d1c4ba0d`）
- 关 thinking、把 C=0.8 写成 \(C^\*\)、把 32B 当主模型
- 把 `actual_generation_seconds` 当 speedup
- 把 GPQA 题目写进 git

---

## 2. 克隆代码

仓库：[github.com/zhujiace/Path-Consistency](https://github.com/zhujiace/Path-Consistency)（public）。论文：[arXiv:2409.01281](https://arxiv.org/abs/2409.01281)。

```bash
git clone git@github.com:zhujiace/Path-Consistency.git
cd Path-Consistency
```

权重、JSONL、运行结果都不在 git 里。旧 Sentence-PC 的 `results/*.json` 只是附录连续性，与本阶段无关。

---

## 3. Python / vLLM 环境

与实验室对齐，避免版本漂移：

```bash
conda create -n PC python=3.10 -y
conda activate PC
pip install torch==2.6.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install vllm==0.8.5+cu118 -f https://vllm-wheels.s3.us-west-2.amazonaws.com/0.8.5/cu118/vllm/index.html
pip install transformers==4.51.3 datasets scipy
# 评分；不要让这套 SymPy 盖住 torch 的 1.13.1
pip install math-verify==0.9.0
export MATH_VERIFY_PATH="$(python -c 'import math_verify, pathlib; print(pathlib.Path(math_verify.__file__).parent.parent)')"
```

若该机 CUDA 是 12.x 且已有匹配的 torch/vLLM wheel，可以用对应 CUDA 的官方 wheel，但 **不要升级到 vLLM 0.10+**：DeepConf 适配写在 0.8.5 的 V0 logits-processor 上。

强制：

```bash
export HF_HOME="${HOME}/data/huggingface"   # 或本机大数据盘
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
# CUDA_VISIBLE_DEVICES 由 run_math.py --gpus 在 import torch 之前设置
```

**禁止**在已经 `import torch` 之后调用 `datasets.load_dataset`（pyarrow jemalloc + CUDA 会随机 segfault）。JSONL 必须在启动引擎前写好。

后台任务用 `setsid nohup … < /dev/null &`。启动脚本默认已经这样包了一层。

---

## 4. 模型

主模型：**Qwen3-30B-A3B**（`Qwen3MoeForCausalLM`，BF16），不是 Qwen3-32B。

```bash
# 示例；把目标路径换成本机大盘
huggingface-cli download Qwen/Qwen3-30B-A3B --local-dir "${HOME}/data/Qwen3-30B-A3B"
export QWEN_A3B_MODEL="${HOME}/data/Qwen3-30B-A3B"
```

架构约束（`config.json`）：32 query heads / 4 KV heads。

| TP | 是否允许 |
|---|---|
| 1, 2, 4 | 允许 |
| 3, 8, 5… | **禁止**（不能整除 4 个 KV head 或 32 个 Q head） |

24 GB 级卡片上 A3B BF16 放不下 TP=2 的 KV+权重，实验室用 **4×4090 TP=4**。新机器显存够就用 **TP=1 或 TP=2**，更少通信。

Serving：

- `unset VLLM_USE_V1`，再传 `--vllm-v1 --allow-cudagraph`
- thinking **必须开**（不要 `--disable-thinking`）
- 采样冻结：T=0.6，top-p=0.95，top-k=20，max_tokens=32768，max_model_len=34816，logprobs=20
- 若 CUDA graph 在 TP warmup 报 `NCCL invalid usage`：stock cu118 NCCL 是 CUDA 11.0 build。设 `PC_NCCL_SO` 指向用 CUDA 12.2 编的 `libnccl.so.2`，`PC_CUDA_LIB` 指向对应 `lib64`（`path_consistency/vllm_nccl.py` 在 import torch 前执行）。能 capture 就不要设。
- **不要**把实验室的 `--num-gpu-blocks-override 8192` 抄到 80 GB 卡上。那个 cap 是 4090 给 graph 留显存用的，大卡上会限制 KV。启动脚本：`export QWEN_GPU_BLOCKS=`（空字符串）即可省略该 flag。

---

## 5. MATH-500 JSONL（不进 git）

需要 500 行、带 `unique_id` 的 JSONL，字段至少 `problem` / `answer`（或 `solution`）。实验室文件：`~/data/Datasets/math500_test.jsonl`，源是 HuggingFaceH4/MATH-500 顺序；**`source_row` = 文件中的 0-index 行号**。工程排除依赖这个顺序，不要 shuffle。

在 **import torch 之前** 转换：

```bash
export HF_HOME="${HOME}/data/huggingface" HF_HUB_DISABLE_XET=1
mkdir -p "${HOME}/data/Datasets"
python - <<'PY'
import json, os
from pathlib import Path
from datasets import load_dataset
out = Path(os.path.expanduser("~/data/Datasets/math500_test.jsonl"))
ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
assert len(ds) == 500, len(ds)
with out.open("w") as f:
    for row in ds:
        f.write(json.dumps({
            "problem": row["problem"],
            "solution": row.get("solution", ""),
            "answer": row["answer"],
            "subject": row.get("subject", ""),
            "level": row.get("level", ""),
            "unique_id": row["unique_id"],
        }, ensure_ascii=False) + "\n")
print("wrote", out, "n=500")
PY
export MATH500_JSONL="${HOME}/data/Datasets/math500_test.jsonl"
```

主统计必须加 `--exclude-engineering-split`。排除的 `source_row`：

```
303, 457, 359, 315, 316, 415, 407, 370, 151, 332
```

剩下 **490** 题。这 10 题曾在 Llama 工程调试中看过，不能当作未见测试集，也不能用来选 \(C^\*\)。

AIME / GPQA JSONL 用 `prepare_eval_datasets.py`（同样必须在 CUDA 之前）。GPQA 选项顺序冻结 `seed=2027`。题目全文不要提交到 git。

---

## 6. 第一次怎么跑

把路径换成这台机器的 python / 模型 / GPU。检查 `nvidia-smi` 空闲显存后再启动。

```bash
export PC_PYTHON="$(command -v python)"          # 或 conda env 的绝对路径
export QWEN_A3B_MODEL="${HOME}/data/Qwen3-30B-A3B"
export MATH500_JSONL="${HOME}/data/Datasets/math500_test.jsonl"
export PC_RESULTS_MATH500="${HOME}/data/path-consistency-results/math500"
export HF_HOME="${HOME}/data/huggingface"
export HF_HUB_DISABLE_XET=1

# 按实际卡号改。TP 只能是 1、2 或 4。
export QWEN_GPUS=0,1,2,3
export QWEN_TP=4

# 大显存：省略 4090 的 KV cap
export QWEN_GPU_BLOCKS=

# 1) smoke 1 题，前台，确认 thinking / boxed / 配对
export QWEN_MAX_QUESTIONS=1
export QWEN_FOREGROUND=1
export QWEN_TAG=math500_qwen3_30b_a3b_smoke1
bash run_math500_qwen30a3b.sh

# 2) 全量 490。删掉 MAX_QUESTIONS；换回正式 tag；后台
unset QWEN_MAX_QUESTIONS
export QWEN_FOREGROUND=0
export QWEN_TAG=math500_qwen3_30b_a3b_b32_mt32768_c08_seed0
bash run_math500_qwen30a3b.sh
```

等价 CLI 见 `run_math500_qwen30a3b.sh` 与 `configs/math500_qwen3_30b_a3b_b32_c08_seed0.json`。JSON 里的 `/home/26xyw/...` 只是实验室记录，**以环境变量 / CLI 为准**。

冻结开关（不要改）：

- `--methods sc,pc`
- `--exclude-engineering-split`
- `--max-branches 32 --window-size 4 --physical-window-size 4`
- `--beta-threshold 0.8`
- `--vllm-v1 --allow-cudagraph --logprobs 20 --seed 0`
- 不要 `--store-text`（32k thinking 文本极大）
- 不要 `--disable-thinking`

JSONL **按题 id 追加，可断点续跑**。同一 tag 改 flag 重跑必须先删旧 `.jsonl`，否则已完成的 id 会被 skip。

日志：`$PC_RESULTS_MATH500/logs/${QWEN_TAG}.log`。进程被杀后先 `nvidia-smi`，必要时只杀本仓库的 `run_math.py` / `EngineCore`。

---

## 7. 报数口径

- 准确率：SC 与 PC 各 490；差值与 paired 单侧 95% 下界。
- Token：decode tokens / 题，PC 相对 SC 的百分比。
- 墙钟：**`generation_seconds`，含 PC 复用的 SC root**。禁止用 `actual_generation_seconds` 报加速。
- 标签写 **PC (C=0.8, provisional)**，不要写 PC (\(C^\*\))。
- 不要和旧 `results/SC_seed0.json`（Sentence-PC，Llama-3-8B，B=20）混表。
- 不要和 DeepSeek MATH-500 490（已完成：SC 87.76% / PC 87.35% / decode −34.5%）混成同一个模型格子。

实验室已完成、仅供对照（不要在新机器上重跑）：

| 模型 × 数据 | SC | PC | decode |
|---|---|---|---|
| Llama-3.1-8B × GSM8K 1319 | 92.12% | 91.74%（−0.38 pp） | −39.4% |
| DeepSeek-V4-Flash × MATH-500 490 | 87.76% | 87.35%（−0.41 pp） | −34.5% |

---

## 8. MATH-500 之后（仍在这台 Qwen 机器）

保持同一套采样与门控，仍用 C=0.8 直到 Dev-Mix 选出 \(C^\*\)：

1. AIME 2024（30）→ AIME 2025（30）→ GPQA-Diamond（198）
2. 官方 DeepConf（不要用仓库里仅离线的 `deepconf_offline.py` 冒充 in-the-loop）
3. DeepConf+PC
4. 官方 CPT **Qwen 入口**（`code/qwen/math.py`）。禁止把 `scripts/run_cpt_llama_adapter.py` 的数字写进主表。

Dev-Mix（GSM8K train 100 + MATH train 100，seed=2027）与 PVC 查找表都还没做；**测试集上的 C=0.8 不能当 \(C^\*\)。**

---

## 9. 实验室还在做什么（不要抢）

4090 机（2026-09-09 13:00 CST 起）GPU 1–3：Llama-3.1-8B MATH-500 / GPQA / AIME，`max_tokens=8192`。DeepSeek AIME/GPQA API 也在跑。结果在实验室 `~/data/path-consistency-results/`，不通过 git 同步。需要对照数字时向实验室拷 JSON summary，不要把带题目的 GPQA JSONL 推进 GitHub。

控制台 http://127.0.0.1:8787/ 只部署在实验室。

---

## 10. 检查清单

- [ ] conda `PC`：torch 2.6.0 + vLLM 0.8.5，`MATH_VERIFY_PATH` 可用
- [ ] `Qwen3-30B-A3B` 权重完整；`HF_HOME` 在大盘
- [ ] `math500_test.jsonl` 正好 500 行且 **未 shuffle**；工程 10 行号对得上
- [ ] `QWEN_GPUS` / `QWEN_TP∈{1,2,4}` / 空的 `QWEN_GPU_BLOCKS` 已按显存设好
- [ ] smoke 1 题：thinking 非空、有 `\boxed{}`、SC 与 PC 写出配对记录
- [ ] 全量 tag `math500_qwen3_30b_a3b_b32_mt32768_c08_seed0`，thinking on，C=0.8 未标成 \(C^\*\)
- [ ] 日志与 JSONL 在数据盘，不在 home 小配额盘

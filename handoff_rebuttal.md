# ICLR 审稿补实验交接

日期：2026-11-08。本文件给**其他服务器**并行开工用。协议以 [`AGENTS.md`](AGENTS.md) 和 [`handoff.md`](handoff.md) 为准。审稿编号：R1 = rS6y，R2 = uwQR，R3 = zwLH。

本机是 4090 实验室（`10.10.60.106`）。**不要在这台机器上再启动 Qwen3-30B-A3B**（要 TP=4，且 GPU 1–3 可能仍被 Llama 占用）。DeepSeek 重复不占卡，可以在任何能访问 API 的机器上跑。

C=0.8 是先验操作点。这些重复和对照**不得**用来重选 C，也不得把结果最好的一次换进表。

---

## 0. 四项实验怎么分机器

| 实验 | 覆盖 | 放到哪台机器 | 占什么 |
|---|---|---|---|
| 1 满并行墙钟 | R1-W2，R1-Q1，R3-W2 | 能同时解码 32 条且吞吐未饱和的大显存机 | Qwen，1 张或 TP∈{1,2,4} |
| 2 长链近期基线 | R1-W1，R1-Q3，R2-W1 的比较部分 | 同一台 Qwen 机，与实验 1 错开 GPU | Qwen |
| 3 多次重复 | R1-W3，R1-Q2 | DeepSeek 部分：API 机。Qwen 部分：Qwen 机 | API 不占卡；Qwen 占卡 |
| 4 相同 token 的树搜索 | R3-W1，R3-Q1 | GSM8K 全量放有空闲 4090 的机器；MATH-500 的 50 题放 Qwen 机 | Llama 单卡 + Qwen |

不需要新跑、不要在别的机器重做：

- **R1-Q4** 配对翻转。DeepSeek 已有：MATH-500（`4856d4ef5673`）490 题里 2 题由对变错、0 题由错变对；AIME 2024（`e4cd6dcd6672`）、AIME 2025（`771c0b2351c6`）、GPQA-D（`a8a29bd70ad3`）翻转数为 0。Qwen 只对已有 JSONL 做同样统计。
- **R2-W2 / R2-Q1** 多样性。用已有轨迹算每题 unique answers，以及投票是否改掉第一条。DeepSeek AIME 2025 上 SC 比 Pass@1 高 13.3 个百分点，PC 与 SC 同为 100%。不必再找新基准。
- **R2-W1** 里除长链比较以外的部分：写法上区分连续因果前缀；用已有 32 条画 SC 的准确率–token 曲线。

相关工作里的 AC / ESC 已经在 Llama/GSM8K 上做过组合，不单开新实验。

---

## 1. 所有新跑共用的协议

- 方法名 PC。Beta 门控是 top-1 vs all-rest。主结果写 **PC (C=0.8)**。
- B=32，窗口 m=4，物理窗口 4。长度网格 `0,0.2,0.4,0.6,0.8`。trial/commit，单层回退，两击停止。有 logprobs 时用 BLC。
- Qwen / DeepSeek：**thinking on**。Llama 没有 thinking 开关，只用 chat template + `\boxed{}`。
- Qwen 采样：T=0.6，top-p=0.95，top-k=20，max_tokens=32768，max_model_len=34816，logprobs=20。
- DeepSeek：`reasoning_effort=high`，max_tokens=32768，logprobs=20。API **没有可依赖的 seed**。重复实验是新的独立请求，用不同 tag，不要把 `--seed` 理解成可复现抽样。
- Llama/GSM8K：T=0.6，top-p=0.9，max_tokens=2048。
- MATH-500 必须 `--exclude-engineering-split`。工程 `source_row`：`303,457,359,315,316,415,407,370,151,332`。准确率口径仍是 raw 490 + 这 10 题记对 / 500。Tok 用 490 的均值。
- 速度：实验 1 单独测满并行墙钟和吞吐。其余实验若报时间，只用 `generation_seconds`（含复用 root）。不要用 `actual_generation_seconds`。
- JSONL 按题追加。改 flag 重跑必须先删该 tag 的旧 jsonl。已完成 tag 不要删、不要续跑进新 tag。
- 不要 `--store-text`，除非离线分析明确缺答案文本。
- 结果放 `~/data/path-consistency-results/`，不进 git。

已完成、不要重跑的 Qwen MATH-500 分片：

| 分片 | fingerprint | source_row | n |
|---|---|---|---|
| first100 | `630825247308` | 0–99 | 100 |
| next100 | `664997212910` | 100–199，跳过 151 | 99 |

本机打包：`~/data/path-consistency-results/math500/qwen_math500_n199_for_aggregate.tgz`。后续 MATH-500 全量若还没齐，剩余是 `source_row` 200–499 再排除工程 9 行，预期 291 题。那是主表补齐，不是下面四项里的新问题。

---

## 2. 实验 1：满并行墙钟

**回答：** R1-W2，R1-Q1，R3-W2。

**问题：** 现有 `generation_seconds` 是每题 8 个窗口串行。DeepSeek MATH-500 上 SC 148 秒、PC 110 秒，只说明两边都串行时 PC 更快。审稿人要的是：SC 的 32 条可以一批并行，PC 必须 8 轮 × 4 条。

**机器：** 一张能同时解码 32 条、且 32 路还没有把吞吐打满的 GPU。24 GB 的 4090 上 A3B 做不到，不要在实验室 4090 上跑这项。Qwen3-30B-A3B，TP 只能是 1、2 或 4。大显存用 TP=1 或 2。`export QWEN_GPU_BLOCKS=`（空）去掉 8192 的 KV cap。

**数据：** MATH-500，抽 32 题即可，不必 490。建议固定 `source_row` 0–31 中去掉不在其中的工程行（这 32 行没有工程行），写死清单，两臂用同一批题。

**两臂，必须同一模型、同一批题、prefix cache 打开：**

1. **SC-parallel：** 32 条轨迹一次批进去，不按窗口串行。测的是这 32 条同时解码的墙钟。
2. **PC-serial：** 维持现在的调度：8 个窗口顺序执行，每窗 4 条，后一窗等前一窗的前缀决定。prefix cache 开。

若现有 `run_math.py` 的 SC 也是 8 窗串行，这项不能直接拿它的 `generation_seconds` 交差。需要一段专门的计时脚本，或临时把 SC 收成一次 `max_num_seqs>=32` 的批。PC 臂保持 8×4。

**记录：**

- 每题墙钟（秒）
- 吞吐：题/小时，以及生成 token/秒
- 生成 token（SC 与 PC）
- GPU 利用率、显存
- 确认这张卡在 batch 32 时没有打满。若利用率已经顶满，换更大的卡或更少的并发，并在记录里写明

**完成标准：** 32 题两臂都有墙钟和吞吐。结论允许是「token 更少，但满并行墙钟不一定更快」。不要把串行的 25% 时间差写成满并行加速。

**建议 tag：**

- `latency_qwen3_30b_a3b_math500_n32_sc_batch32`
- `latency_qwen3_30b_a3b_math500_n32_pc_serial8x4`

---

## 3. 实验 2：长链上的 PoLR、CPT、DeepConf

**回答：** R1-W1，R1-Q3，以及 R2-W1 里「比较只在 Llama/GSM8K」的那一部分。

**不要用** `scripts/run_cpt_llama_adapter.py` 或 Llama 上的 CPT 数字填这张表。Llama/GSM8K 的 PoLR、CPT、AC、ESC 已经有了，只留在短链表。

**模型：** Qwen3-30B-A3B，thinking on。采样与第 1 节相同，C=0.8，B=32，m=4。

**数据（两套都跑）：**

- AIME 2024（30）和 AIME 2025（30）。这里 SC 明显高于 Pass@1，能同时回答「为什么还要 SC」。
- MATH-500 490（`--exclude-engineering-split`）。

**四个系统，预算和采样对齐：**

| 系统 | 要求 |
|---|---|
| SC | 同预算 B=32，作为父方法 |
| PC (C=0.8) | 标准协议。若该 tag 的 JSONL 已完整，不要重跑，直接引用 |
| DeepConf-offline | 官方语义：从同一批完整轨迹做 confidence filtering / weighted vote。`N_init=16`，B=32。不要在拿不到可靠 usage 时把提前截断叫 online DeepConf |
| DeepConf+PC | warmup、预算、采样、聚合与上面的 DeepConf 相同，只是后续轨迹走 PC 前缀 |
| CPT | **官方 Qwen 入口**（`code/qwen/math.py` 那条），32 workers。共享的是去重后的笔记，不是一条连续前缀 |
| PoLR | 仅当官方或当前 `run_polr.py` 能接到 thinking 模型时才跑。N=32，`L_p=128` 与 Llama 表一致。接不上就在记录里写明，不要用短前缀聚类冒充 |

**建议 tag：**

- `aime_2024_qwen3_30b_a3b_b32_c08_seed0`（SC/PC，已有则跳过）
- `aime_2025_qwen3_30b_a3b_b32_c08_seed0`
- `math500_qwen3_30b_a3b_b32_mt32768_c08_seed0`（490，已有分片不要删）
- `aime_2024_qwen3_30b_a3b_deepconf_offline_b32`
- `aime_2024_qwen3_30b_a3b_deepconf_pc_b32`
- `aime_2025_...` 与 `math500_...` 同样各一套 DeepConf / DeepConf+PC
- `aime_2024_qwen3_30b_a3b_cpt_official_n32`，AIME 2025 与 MATH-500 同样
- `aime_2024_qwen3_30b_a3b_polr_n32_lp128`（仅当能接上）

**完成标准：** 每个数据集上 SC、PC、DeepConf、DeepConf+PC、官方 CPT 都有准确率、生成 token、`generation_seconds`。PoLR 有数字或有「无法在 thinking 模型上运行」的记录。

Qwen 启动骨架（路径按那台机器改）：

```bash
export PC_PYTHON="$(command -v python)"
export QWEN_A3B_MODEL="${HOME}/data/Qwen3-30B-A3B"
export MATH500_JSONL="${HOME}/data/Datasets/math500_test.jsonl"
export PC_RESULTS_MATH500="${HOME}/data/path-consistency-results/math500"
export QWEN_GPUS=0
export QWEN_TP=1
export QWEN_GPU_BLOCKS=
export QWEN_TAG=math500_qwen3_30b_a3b_b32_mt32768_c08_seed0
bash run_math500_qwen30a3b.sh
```

AIME 不要用这个 MATH-500 脚本硬跑。用 `run_math.py` 的同一组采样 flag，换数据集和 tag。TP 只能是 1、2、4。

---

## 4. 实验 3：三次重复，报方差

**回答：** R1-W3，R1-Q2。

论文表格里的那一次**保留**，回复里说明它是单次运行。再跑 2 个独立重复，连同已发表的一次共 3 次。报告准确率和 token 节省的 mean±std。看多次运行下 PC 是否仍接近 SC，token 节省是否仍为正。

协议不变。不要从 3 次里挑最好的一次，不要改 C。

**DeepSeek（API，不占 GPU）。** API 没有可依赖的 seed。`rep1` / `rep2` 是新的独立调用，SC 与 PC 在同一次重复内配对（PC 复用**这一次**的 SC root，不要复用 `rep0` 的 root cache）。

| 数据 | 已发表 tag（rep0，不要重跑） | 新 tag |
|---|---|---|
| AIME 2024，30 题 | `aime_2024_deepseek_v4_flash_b32_c08_rep0` / `e4cd6dcd6672` | `..._rep1`，`..._rep2` |
| AIME 2025，30 题 | `aime_2025_deepseek_v4_flash_b32_c08_rep0` / `771c0b2351c6` | 同上 |
| MATH-500，490 | `math500_deepseek_v4_flash_b32_c08_rep0` / `4856d4ef5673` | 同上 |

```bash
# 在已配置 DEEPSEEK_API_KEY 的机器上。key 不要写进仓库。
python run_math_deepseek.py \
  --tag aime_2024_deepseek_v4_flash_b32_c08_rep1 \
  --methods sc,pc \
  --max-branches 32 --window-size 4 --beta-threshold 0.8 \
  --max-tokens 32768 --logprobs 20 --reasoning-effort high
```

数据集 flag 按 `run_math_deepseek.py --help` 和现有 `configs/aime_2024_deepseek_v4_flash_rep0.json` 对齐。MATH-500 加 `--exclude-engineering-split`。`rep2` 只改 tag。

**Qwen（要 GPU）。** 这里 `--seed` 真的改变抽样。已发表的是 seed 0。再跑 seed 1 和 seed 2。同一种子内 SC/PC 配对。

| 数据 | 新 tag |
|---|---|
| AIME 2024 | `aime_2024_qwen3_30b_a3b_b32_c08_seed1` 与 `seed2` |
| AIME 2025 | `aime_2025_qwen3_30b_a3b_b32_c08_seed1` 与 `seed2` |

Qwen 的 MATH-500 / GSM8K / GPQA **不**做 3 次重复。审稿人点名噪声的是 AIME。若 seed 0 的 AIME JSONL 在另一台机器上，先拷过来作为三次里的第一次，不要重跑 seed 0。

**完成标准：** 每个格子 3 个独立数字，表里写 mean±std，并给出 3 次里 PC−SC 的准确率差和 token 节省率。

---

## 5. 实验 4：相同 token 预算的 ToT、beam、MCTS

**回答：** R3-W1，R3-Q1。

仓库里**还没有** Tree-of-Thoughts、step-level beam、rollout MCTS 的入口。这项要先写对照脚本，再跑。概念上的差别（PC 不建树，最终仍是 B=32 多数票）写进回复，但不能代替这组数字。

**对照：**

| 系统 | 定义 |
|---|---|
| PC | 现有协议，C=0.8，B=32 |
| Tree-of-Thoughts | Yao et al. 2023。按步提议、评估、搜索，不是共享一条连续前缀 |
| step-level beam search | 步级束搜索。束宽使总生成 token 接近 PC |
| rollout MCTS | 带 rollout 的搜索，对应审稿人引用的 Inference Scaling Laws（Wu et al.）。不要把 PC 的 trial/rollback 记成 MCTS |

**预算：** 每题生成 token 对齐该题 PC 的 `total_generated_tokens`。先跑 PC（或引用已有 PC），再把这个 token 数当作 ToT / beam / MCTS 的上限。报告准确率和实际生成 token。不要为了让某一边好看去改预算。

**两套数据：**

1. **Llama-3.1-8B-Instruct / GSM8K 1319，全量。** 单卡即可。T=0.6，top-p=0.9，max_tokens=2048。PC 若已有 `gsm8k_llama31_b32` 的完整 JSONL，不要重跑 PC，只跑三个搜索方法，预算用那份 PC 的逐题 token。
2. **Qwen3-30B-A3B / MATH-500，只抽 50 题。** 固定 `source_row`：0–49（其中没有工程行）。thinking on，采样同第 1 节。这 50 题的 PC 若还没有，用实验 2 的协议只跑这 50 行，tag 与全量 490 分开，例如 `math500_qwen3_30b_a3b_totcmp_n50_b32_c08_seed0`。

**建议 tag：**

- `gsm8k_llama31_tot_matched_pc_seed0`
- `gsm8k_llama31_beam_matched_pc_seed0`
- `gsm8k_llama31_mcts_matched_pc_seed0`
- `math500_qwen3_30b_a3b_tot_matched_pc_n50_seed0`
- `math500_qwen3_30b_a3b_beam_matched_pc_n50_seed0`
- `math500_qwen3_30b_a3b_mcts_matched_pc_n50_seed0`

**完成标准：** 两套数据上四个系统都有准确率和生成 token。token 均值与对应 PC 的差距要在记录里写出来。不要把搜索准确率更高写成 PC 失败，也不要写成 PC 赢过搜索；这组实验是为了把任务差别量出来。

---

## 6. 做完怎么交回

每项交回这些文件，不要交模型权重和 API key：

- `*.jsonl`、`*.run.json`、`*.summary.json`
- 实验 1 另交一张墙钟 / 吞吐 / GPU 利用率表
- 实验 4 另交逐题 PC token 预算和搜索方法实际 token

汇总时按 `source_row` 或 `question_id` 对齐。同一 tag 只保留一次完整结果。

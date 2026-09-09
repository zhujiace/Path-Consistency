# AGENTS.md

仓库两条线：（1）vLLM 评测管线，复现 Path-Consistency（arXiv 2409.01281，Jiace Zhu 等）并接 ICLR 2027 冻结协议；（2）ICLR 2027 修稿。论文决策先读 `paper/report_ICLR2027_revision.md`（§8 审稿对照，§9 执行日志，§10 图表口径，§12 实验记录）；审稿原文 `paper/ICML_review.md`、`paper/NeurIPS_review.md`。新服务器开 Qwen 实验只读本文件 + [`handoff.md`](handoff.md)。

GitHub：`git@github.com:zhujiace/path-consistency.git`（**private**；含未投稿稿与审稿，不要改 public）。远程结果与权重不进 git。

用户用中文；仓库 markdown 用中文；论文 LaTeX 保持英文。

---

## 两台机器怎么分（2026-09-09 起）

| | **4090 实验室（本机）** | **新服务器（Qwen）** |
|---|---|---|
| 角色 | Llama-3.1-8B-Instruct + DeepSeek-V4-Flash API | **Qwen3-30B-A3B** 主表，以及随后的 DeepConf / 官方 CPT |
| 代码 | `/home/26xyw/projects/path-consistency` | clone 同一 GitHub 仓库 |
| 不要做 | 不要在这里再启动 A3B（要 TP=4 且 4 张空闲 4090；GPU 1–3 已被 Llama 占） | 不要跑 Llama 分片、不要配 DeepSeek key、不要把 C=0.8 写成 \(C^\*\) |
| 交接 | 见下文「本机路径」 | **先读 [`handoff.md`](handoff.md)** |

两机协议必须一致：paired-root v2、B=32、m=4、thinking on（Qwen/DeepSeek）、工程 10 题排除、速度口径 `generation_seconds`。

---

## 冻结协议（禁止改名、禁止混表）

- 方法名 **PC**。Beta 门控是 **top-1 vs all-rest**（不是 top-2）。主表只报 **dev 选出的 \(C^\*\)**。
- 每个逻辑窗口 **一条**共享前缀；trial/commit；单层回退；两击重置；有 logprobs 时用 **BLC**。
- 长度策略现状是 `provisional_fixed_plus_20_percent`。**PVC 查找表尚未实现**，不要写成 adaptive PVC。
- paired-root v2：PC 复用 SC 的无前缀窗口。速度只报 **`generation_seconds`（含复用 root）**，**禁止**把 `actual_generation_seconds` 当 speedup。
- **C=0.8 是临时阈值，不是 \(C^\*\)**。禁止用 GSM8K test、MATH-500 DeepSeek 490、或任何测试集回写选 C。
- MATH-500 主统计排除工程 10 题，`source_row`（0-index JSONL 行号）：
  `[303, 457, 359, 315, 316, 415, 407, 370, 151, 332]`（剩 490）。
- **Qwen3 与 DeepSeek 必须 thinking on**。Llama 没有 thinking 开关，只用 chat template + `\boxed{}`。禁止把 Llama CoT 格子和 Qwen thinking-on 格子混成「同一协议」。
- 旧 `results/SC_seed0.json` 等是 Llama-3-8B **Sentence-PC**（B=20）。不得与新协议 B=32 混算节省率。
- 本地 Qwen 主模型是 **Qwen3-30B-A3B**，不是 32B。暂停的 32B V0 eager 490（fingerprint `9f54d1c4ba0d`）**不要续跑、不进 Table 1**。
- `margin` 只是 top1−top2 票差 ≥ 2 的启发式，**不是**忠实 MARS；禁止写成 MARS。
- 公平 CPT 必须走官方 Qwen 入口 + A3B。Llama 上的 CPT adapter smoke **不是**公平比较。

Go/no-go 与 PVC/Dev-Mix 细则：`paper/plan_2weeks_ICLR2027.md`。

---

## 实验地图（2026-09-09）

### 已完成（可引用，勿重跑同一 tag）

| 设置 | 数字 | 备注 |
|---|---|---|
| Llama-3.1-8B GSM8K 1319 | SC 1215/1319=92.12%；PC 1210/1319=91.74%（−0.38 pp）；decode 8316→5039（−39.4%） | 新协议；Pass@1 均为 84.69% |
| DeepSeek-V4-Flash MATH-500 490 | SC 430/490=87.76%；PC 428/490=87.35%（−0.41 pp）；decode −34.5% | fingerprint `4856d4ef5673`；thinking + reasoning_effort=high |
| Llama MATH-50 | SC=PC 54%；decode −4.17% | 不可用于 \(C^\*\) / PVC / 主表 |
| DeepConf+CPT Llama GSM8K smoke 10 题 | DeepConf-online majority 7/10；其它官方 vote 8/10；CPT-adapted-llama 8/10 | 隔离环境；**CPT 不公平** |

### 本机仍在跑（不要杀）

2026-09-09 13:00 CST 起，`scripts/run_llama31_remaining.sh`，vLLM V1+CUDA graph，TP=1，GPU 1–3（约 22.1 GiB/卡）。**GPU 4 空着**。Llama 采样：T=0.6，top-p=0.9，top-k=−1，**max_tokens=8192**（不是 4096）。

| GPU | 数据集 | tag | fingerprint |
|---|---|---|---|
| 1 | MATH-500 490 | `math500_llama31_b32_mt8192_c08_seed0` | `abc27a862c6a` |
| 2 | GPQA-D 198 | `gpqa_diamond_llama31_b32_mt8192_c08_seed0` | `dc17da48aca4` |
| 3 | AIME 2024→2025 | `aime_2024_llama31_…` 然后 2025 | `8e57f7657fc7`（2024） |

DeepSeek AIME24/25 + GPQA：`scripts/run_deepseek_aime_gpqa.sh`（API，不占卡），2026-09-08 15:07 CST 启动。tags：`aime_2024_deepseek_v4_flash_b32_c08_rep0`（`e4cd6dcd6672`）、`aime_2025_…`（`771c0b2351c6`）、`gpqa_diamond_…`（`a8a29bd70ad3`）。完成与否以 `*.run.json` / JSONL 题数为准，不要凭控制台印象结案。

### 新服务器下一步（P0 Qwen）

1. **立即**：Qwen3-30B-A3B × MATH-500 490 paired SC/PC，C=0.8（仍非 \(C^\*\)），thinking on。启动：`bash run_math500_qwen30a3b.sh`（先改环境变量，见 handoff）。
2. 同一模型 × AIME24/25、GPQA-D（协议相同）。
3. 公平 DeepConf、DeepConf+PC、官方 CPT，全部在 A3B 上。不要用 Llama adapter 填主表。

---

## 本机（4090 实验室）路径与环境

- ML python = `/home/26xyw/.conda/envs/PC/bin/python`（torch 2.6.0+cu118，vllm 0.8.5+cu118，transformers 4.51.3）。系统 python 没有 torch/vllm。
- `/home/26xyw` **100 GB 配额**；`~/data` → `/data/26xyw`（无配额）。模型、数据集、大 JSONL 必须在 `~/data`。
- `HF_HOME=~/data/huggingface`；评测 JSONL：`~/data/Datasets/*.jsonl`。结果：`~/data/path-consistency-results/{math500,aime,gpqa,gsm8k,deepconf-cpt}/`。
- HF 下载：`export HF_HOME=~/data/huggingface HF_HUB_DISABLE_XET=1`。gated 仓需要用户给的 `HF_TOKEN`，禁止写入仓库或 echo。
- **HuggingFace / GitHub HTTPS 直连经常超时**。Clash 监听 `127.0.0.1:27890`：
  - 交互：`proxy-on`（只 export，不启动 clash）。
  - Agent / 非交互：`export HTTP_PROXY=http://127.0.0.1:27890 HTTPS_PROXY=http://127.0.0.1:27890 http_proxy=http://127.0.0.1:27890 https_proxy=http://127.0.0.1:27890`
  - 端口没在听：`clash-start`（`~/clash-for-linux-without-sudo`），`clash-status` 确认。
  - 关掉：`proxy-off`。测通：`curl -I https://huggingface.co` 或 `https://github.com` 应 200。
  - Git SSH `github.com:22` 也会超时；git 用  
    `GIT_SSH_COMMAND='ssh -o ProxyCommand="nc -X connect -x 127.0.0.1:27890 %h %p"'`。
  - DeepSeek API **不必**走这条代理。
- GPU：5× RTX 4090，与他人共享（GPU0 最忙）。Llama-3-8B fp16 需要 ≥ ~19 GB 空闲。segfault 会留下 orphan EngineCore 占 19–23 GB：`pkill -9 -f run_gsm8k`（或 `-f EngineCore`），再查 `nvidia-smi`。只杀本仓库进程，不要杀别人的 `llama-server`。
- **`CUDA_VISIBLE_DEVICES` 必须在 import vllm/torch 之前设置**（`run_gsm8k.py` / `run_math.py` 自己解析 `--gpu` / `--gpus`）。
- CUDA 上下文建立后 **禁止** `datasets.load_dataset` / pyarrow：jemalloc 后台线程会在之后某个时刻 segfault。所以数据全是 JSONL。
- 后台任务：`setsid nohup … < /dev/null &`。普通 `nohup … &` 会在启动 shell 结束时被杀。
- DeepSeek key：**只读** `/data/26xyw/env/deepseek_api.env`，永不进 git。
- math-verify 0.9.0 在 `/data/26xyw/python_packages/math_verify_0_9`（`MATH_VERIFY_PATH`）。不要把它的 SymPy 1.14 插到 vLLM 环境前面。
- 控制台：`console/serve.py`，http://127.0.0.1:8787/ 。改 HTML/API 后必须重启；用户需硬刷新。改过的 tag 分类：`llama31_math500` / `llama31_aime` / `llama31_gpqa`。Table 1 的 PC Tok 列是相对 SC 的节省（GSM8K −39.4%）。墙钟 `DD:HH:MM:SS`。
- TinyTeX：`export PATH=$HOME/.TinyTeX/bin/x86_64-linux:$PATH`。`pdflatex` + `bibtex` 两遍。不要用 tectonic 或 `/data/26xyw/envs/texlive`。

## GPU / vLLM 经验（两机通用）

- Qwen3-30B-A3B：`num_attention_heads=32`，`num_key_value_heads=4`。**TP 必须整除二者 → 只允许 1 / 2 / 4。TP=3 与 TP=8 都不行。** 24 GB 卡上 BF16 **不能 TP=2**，4090 上必须 TP=4。大显存卡优先 TP=1 或 2。
- V1 CUDA-graph：先 `unset VLLM_USE_V1`，再传 `--vllm-v1 --allow-cudagraph`。torch cu118 自带 NCCL（CUDA 11.0）capture 会 `NCCL invalid usage`。本机修复：`path_consistency/vllm_nccl.py` 在 import torch 前加载 CUDA-12.2 的 `libnccl.so.2`。新机器若 stock NCCL 能 capture 就不要硬拷贝本机路径；否则设 `PC_NCCL_SO` / `PC_CUDA_LIB`。
- 4090 上 A3B 用 `num_gpu_blocks_override=8192` 是为了给 graph capture 留显存；**在大显存机器上不要带这个 cap**（会限制 KV）。启动脚本：未设置时默认 8192；新机器 `export QWEN_GPU_BLOCKS=`（空）以省略该 flag。
- Dense Qwen3-32B：capture max=512 会 OOM；`{32…1}` 才能 capture。已归档，主表不用 32B。

## 入口命令

```bash
# 旧 Sentence-PC GSM8K（附录连续性；不要当主表）
run_gsm8k.py --method {sc,pc,ac,esc,margin} --gpu N --seed 0

# 新协议 MATH / AIME / GPQA（本地 vLLM）
run_math.py --methods sc,pc --exclude-engineering-split ...

# DeepSeek API
run_math_deepseek.py ...
scripts/run_deepseek_aime_gpqa.sh

# 本机 Llama 剩余集
scripts/run_llama31_remaining.sh

# Qwen3-30B-A3B MATH-500（新服务器主入口）
bash run_math500_qwen30a3b.sh
```

- `pc` 的 `--confidence` / `--beta-threshold` 是 beta 门控阈值；`ac` 用它当 stop 阈值；`sc` = PC 且 C=1.0。
- 默认新协议：32 分支 / 窗口 4，Qwen T=0.6 top-p=0.95 top-k=20；Llama T=0.6 top-p=0.9。
- `--max-questions K` 做 smoke；`--store-text` 才存 `generations`+`prefixes_used`（离线分析需要；32k thinking 不要默认开）。
- 输出：`results/<TAG>.jsonl` 或 `~/data/path-consistency-results/.../<TAG>.jsonl` **按题 id 增量追加**（可续跑）。改 flag 重跑必须先删旧 jsonl，否则所有 id 会被 skip。
- 分析：`analyze_results.py`（旧 Table 1）；`analysis/offline_stats.py`（V(t)、门控、多样性）；`analysis/plot_accuracy_tokens.py`。分析脚本用系统 python 即可。

## 论文 `paper/src/`

已改：`03_methodology.tex`（beta 式 (4)，§4.2 idealization）、`04_evaluation.tex`（阈值措辞）。编译 PDF 放到 `paper/`。投稿 PDF/zip 与审稿原文不要改内容。

## 禁止清单（两机）

- 把 C=0.8 标成 \(C^\*\)，或从测试集选 C
- 混用旧 Sentence-PC 与新协议
- 用 `actual_generation_seconds` 报 speedup
- 续跑暂停的 32B eager 490；用 32B 替换 A3B
- Qwen/DeepSeek 主表关 thinking
- Llama MATH-500 490 用 **4096** max_tokens（本机剩余 run 已改 8192）
- 提交 DeepSeek key / HF token / GPQA 全文 JSONL
- 杀其他用户的 GPU 进程
- `git config --global`；force-push；把仓库改 public

# ICLR 2027 两周实验与修稿计划 v4

> 版本日期：2026-09-09（取代 v3 的 \(C^\*\) 与 PVC 查找表）。权威记录：`report_ICLR2027_revision.md` §12.14。
>
> **冻结口径：** 主实验与 Table 1 一律 **PC (C=0.8)**。0.8 是先验默认操作点（沿用原论文 / Adaptive-Consistency），**不是** Dev-Mix \(C^\*\)，禁止从任何测试集回写「最优 C」。阈值 sweep \(\{0.7,0.8,0.9,0.95,1.0\}\) 只进消融。长度用固定 20% 网格（gate 通过则前进一档）。**不做 PVC 查找表控制器。** 干预式 PVC 若做，只当分析图。
>
> 目标是在 9 月 18 日摘要截止前冻结方法、题目和核心证据；9 月 25 日全文截止前只补预先登记的实验、写作与审计。官方截止时间均为 23:59 AOE。
>
> 本计划依据论文原稿、ICML/NeurIPS 审稿意见、当前代码审计及 2026 年最新并行推理论文制定。三个主模型已经位于 ~/data 或 API，不安排模型下载。v3 中与本节冲突的 \(C^\*\) / PVC 查找表条文作废。

---

## 0. 最终研究决策

### 0.1 一句话主线

保留原论文“多路径采样 → Beta 门控 → 前缀复用 → 聚合”的骨架，将方法升级为：

> **PC（Path-Consistency）用答案级 Beta 后验决定是否共享，用固定 20% 网格决定共享多长（gate 通过则前进一档），用局部置信度从多数答案簇中选择唯一共享前缀，并在门控恶化时单层回退；它既可与 PoLR / CPT 比较，也可叠加到 DeepConf / AC / ESC。**

不要把主方法写成 adaptive PVC，也不要写成 PC(\(C^\*\))。

### 0.2 对当前意见的裁决

| 问题 | 最终决定 | 原因与边界 |
|---|---|---|
| Beta 门控还是简单投票 | **Beta 为主，简单多数票仅作消融** | Beta 给出多数答案概率超过 0.5 的后验统计；简单投票无法度量证据强弱。但 Beta 只支持“答案占优”，不证明答案或前缀正确。 |
| 主表是否展示阈值 sweep | **不展示 sweep，只展示默认 C=0.8** | 0.8 是投稿前预先声明的操作点，不是测试集最优。完整曲线放消融/附录。叙事：更低 C → 更大加速、更高错前缀风险；更高 C → 更接近 SC、更安全。C=1 即 SC。 |
| 自适应前缀 | **不采用 PVC 查找表。主策略是固定 20% 网格** | 名义 level 为 0%→20%→40%→60%→80%；gate 通过则前进一档，gate 失败则停留；失败 trial 可回退一层。查找表换模型/数据要重估，泛化差。固定比例序列就是主方法（Relative-PC），不要另称 adaptive。 |
| 每个 level 的并行语义 | **一个 level 只有一个前缀，所有分支共享** | 这是主算法而非部署消融；逻辑窗口可拆成物理 micro-batch，但状态只在整个窗口完成后更新。 |
| 与截断型方法的关系 | **组合，不是附录点缀** | Table 2 下块：DeepConf / AC / ESC 作为截断型 parent，各报 parent 与 parent+PC。旧 Sentence-PC（B=20）的 AC/ESC 不进本表。 |
| 最新强基线 | **PoLR 与 CPT 为必须比较；DeepConf / AC / ESC 为必须组合；DTS-Stable 为 P1** | PoLR / CPT 与 PC 同属并行共享；截断型控制「停多少条」，PC 控制「后面怎么走」。 |
| 前缀下降后的处理 | **候选先试用、后提交；失败回退一层** | 推理时看不到真实准确率，只能用 modal-answer 改变或 Beta 门控失效作为可观测代理；高置信一致错误仍是明确限制。 |

### 0.3 Go / no-go 总标准

只有同时满足以下条件，才按当前主线投稿：

1. PC 相对同预算 SC 在至少两个模型、两个现代数据集上准确率非劣（允许界 \(\epsilon=1.0\) percentage point），并有稳定 token 或 latency 收益；
2. DeepConf+PC 相对相同 DeepConf parent 至少在 Qwen3-30B-A3B 的两个设置上继续产生增量；
3. PC 在 matched-budget 或 matched-latency 下不劣于最接近的 CPT；
4. 代码、公式、门控、回退和表格口径完全一致。

**不再作为 go/no-go：** 干预式 PVC 是否画出安全区；Dev-Mix 是否选出 \(C^\*\)。PVC 分析图与阈值 sweep 是消融/附录，缺了不阻止投稿，但不得用测试集挑 C 来补救第 1 条。

若第 3 条失败，应把贡献缩为“Beta 驱动的连续前缀重定向”，不得声称并行共享方法整体领先。若第 1/2 条也失败，建议不提交本轮 ICLR。

---

## 1. 最小改稿策略与新增贡献

### 1.1 保留与修改

| 论文部分 | 保留 | 最小必要修改 |
|---|---|---|
| 问题与框架 | SC 中重复生成、窗口式多路径推理、训练后无需更新参数 | 把定位改为“有损、受门控的计算重定向”，不再笼统声称无损或 model-agnostic |
| Beta 门控 | top-1-vs-all-rest 的 Beta 后验 | 统一公式和代码；明确相关分支使独立同分布假设近似化；主表操作点 C=0.8 |
| 多级前缀 | 逐级增加复用信息 | 句子数改为归一化 token 比例；固定 20% 网格 + 候选验证和单层回退。**不要写 PVC 查找表控制器** |
| 前缀来源 | 多数答案对应轨迹 | 随机抽样改为答案置信 + 局部 token 置信的唯一前缀选择 |
| 最终聚合 | SC majority vote | DeepConf parent 保持其官方 filtering/weighted vote，不偷换为普通投票 |
| 理论 | Beta 后验解释 | 删除无效 Taylor “不降准确率”推导和错误 theorem；改成风险分解与可检验假设 |
| 旧实验 | 作为历史连续性 | 十个旧任务和原始 Sentence-PC 移附录；新主表只用修复后协议 |

### 1.2 投稿版贡献（相对 v3 的调整）

1. **Beta-gated unique shared prefix**：每个逻辑窗口一条共享前缀；trial/commit、单层回退、两击重置。
2. **Relative normalized prefixing**：跨短 CoT 与长 reasoning trace 使用 20% 比例网格；gate 通过则前进一档。这是主长度策略，不称为 adaptive PVC。
3. **Beta-guided Local-Confidence selection（BLC）**：Beta 先确定可靠答案簇，再从该簇选择局部最可靠的一条连续前缀，使每个 level 可被完全并行执行。
4. **Composable control**：在相同 parent 下验证 SC→SC+PC、DeepConf→DeepConf+PC、AC→AC+PC、ESC→ESC+PC。

**Prefix Value Curve** 降为可选分析：若时间允许，用独立 discovery/evaluation rollouts 画准确率—成本曲线，解释何时前缀有益。它不是测试时策略，也不是方法主贡献。

### 1.3 与 2026 最近工作的区分

- **DeepConf** 控制低置信轨迹的停止与最终投票；**AC / ESC** 按答案一致性提前结束采样；PC 决定后续轨迹从哪个连续推理前缀继续。三者与 PC 控制不同维度，因此做 parent vs parent+PC。
- **CPT** 抽取、去重并广播多个“信息单元”，保留各分支私有历史；**PoLR** 用前缀聚类剪枝后展开主导簇；PC 只选择一个模型原生、连续、可缓存的因果前缀，并用 Beta/回退控制风险。PoLR 与 CPT 是必须定量比较的最近邻。
- **DTS** 动态扩展/筛选解码树；PC 维持窗口式并行采样，不做 token-level 树搜索。若官方实现能在冻结协议下适配 Qwen3-30B-A3B，则列为现代结构化搜索基线。

---

## 2. 冻结后的 PC 方法

### 2.1 主设置与状态

- 主预算：每题最多 \(B=32\) 条 launched traces；
- 逻辑窗口：\(m=4\)，共 8 个窗口；
- 前缀比例：\(\mathcal R=\{0,0.2,0.4,0.6,0.8\}\)；
- 最大复用：80%，避免把最终结论或 answer span 当作前缀；
- 主门控：cumulative valid completed traces 上的 Beta gate，**阈值 C=0.8**；
- 长度策略：`provisional_fixed_plus_20_percent`（gate 通过则前进一档）；
- 最终聚合：SC parent 用 majority；DeepConf parent 用其原始 confidence filtering/weighted vote；
- 保存两套答案集合：validated answers 用于后续控制，all completed answers 用于最终 parent-compatible 聚合。失败窗口的 token 和答案不得从成本或最终结果中静默删除。

原论文 \(B=20,m=5\)、per-branch random source 的 Sentence-PC 只作为附录连续性基线；不得与 \(B=32\) 的结果直接计算节省率。长度消融中的 Shared Sentence-PC 使用 \(B=32,m=4\) 和相同 BLC/rollback，仅把 token-ratio 换回“每层 +1 sentence”，从而隔离长度策略。

### 2.2 Beta 门控与阈值

设当前 validated answers 共 \(n\) 条，众数答案获得 \(v_m\) 票，采用均匀先验：

\[
\operatorname{conf}(n,v_m)
=\Pr(p_m>0.5\mid v_m,n)
=1-I_{0.5}(v_m+1,n-v_m+1).
\]

当 \(\operatorname{conf}\ge C\) 时才允许提出新前缀。这衡量“众数答案占优”的后验置信，不等价于答案正确率，更不构成前缀无害保证。

**默认操作点 \(C=0.8\)：**

1. 沿用原论文与 Adaptive-Consistency 常用值，投稿前预先声明，不是开发集选出的 \(C^\*\)；
2. 同一 C 跨 Llama / Qwen3-30B-A3B / DeepSeek-V4-Flash 与 MATH-500、AIME24、AIME25、GPQA-D 固定；
3. **禁止**从 GSM8K test、MATH-500、AIME、GPQA 或任何测试集回写「最优 C」；
4. 已完成的 C=0.8 配对 run **不必重跑**。

**消融 A（敏感性，不选阈）：** 候选 \(\{0.70,0.80,0.90,0.95,1.00\}\)；\(C=0\) 只作为“纯多数门控”可选对照；\(C=1\) 即 SC。正文解释权衡，不把某一测试格标成最优。Llama GSM8K 目前只有 C=0.8 与 SC，其余 C 待补。

主表只写 **PC (C=0.8)**。完整阈值曲线仅进消融。

### 2.3 前缀长度（固定 20% 网格）

**候选构造：**

1. 先从完整 trace 删除 final-answer span；Qwen/DeepSeek 只从 reasoning span 取前缀；
2. 设剩余 token 长度为 \(L\)，比例 \(r\) 的候选终点为 \(\lfloor rL\rfloor\)；
3. 在终点前最多回看 32 tokens，优先吸附到换行、空白或安全标点；不得切入 LaTeX 环境、代码块、\(\boxed{}\)、Final Answer 或答案解释段；
4. 保存原始 token IDs、吸附前后位置和 mask reason，禁止只靠字符串 “answer is”。

**level 的具体行为：**

| 当前安全比例 | 下一候选比例 | 允许动作 |
|---:|---:|---|
| 0% | 20% | gate 通过则试用 20%；否则保持 0% |
| 20% | 40% | gate 通过则前进；否则停留 |
| 40% | 60% | 同上 |
| 60% | 80% | 同上 |
| 80% | — | 保持 80%，只继续采样或回退 |

因此 Level 1/2/3/4 的名义长度分别为 20%/40%/60%/80%，PC 每个窗口最多前进一个 level。这就是主方法 **Relative-PC**。Fixed-20 / Fixed-40 / Shared Sentence-PC 只作消融对照。

**已取消：PVC 查找表控制器。** 不再在 Dev-Mix 上按 confidence/length bin 拟合 \(V_{\mathrm{corr}}(r)\) 并在测试时查表决定 advance/stay。不要在方法节或主表把当前实现写成 adaptive PVC。

### 2.4 每个 level 如何选择唯一前缀

采用 **BLC 两阶段选择**：

1. **答案级选择**：Beta gate 确定当前 cumulative modal answer \(a_m\)；
2. **路径级选择**：候选池只包含“刚完成窗口中、答案等于 \(a_m\)、继承当前安全前缀，且候选 endpoint 严格长于当前前缀”的完整轨迹；候选池为空则本轮 stay，不从旧窗口拼接不相容路径；
3. 对每条候选轨迹 \(z\)，只在准备新增的区间 \(r\rightarrow r+0.2\) 上计算 lower-tail local confidence：

\[
s(z)=\min_{W\in\mathcal W}
\frac{1}{|W|}\sum_{t\in W}\log p(z_t\mid z_{<t}),
\]

其中滑窗宽度 128 tokens、stride 64；区间不足 128 tokens 时使用整个区间；
4. 选择 \(s(z)\) 最大者；平局依次按完整轨迹更短、branch id 更小决定；
5. 从该轨迹提取唯一候选前缀，下一逻辑窗口所有分支共同使用。

这比随机抽样多一层可解释的路径可靠性标准，但 token likelihood 仍不是正确性证明。必须与 random source、shortest source、token-level medoid 和 mean-logprob source 做消融。

若某 API 不能返回 reasoning-token logprobs，不得静默换算法：该模型使用预先声明的 token-medoid fallback，并在表中标成 **PC-medoid**。DeepSeek-V4-Flash 必须先通过 logprob + prefix-completion smoke test，才能进入 BLC 主结果。

### 2.5 候选提交、取消与回退

新前缀不是立即永久提交，而是下一窗口的 **trial prefix**。每个安全状态保存：

    (prefix_tokens, ratio, modal_answer, beta_confidence, source_id)

下一窗口完成后：

1. 若 modal answer 未改变且 Beta confidence 仍不低于 \(C=0.8\)，trial 被提交并压入 prefix stack；
2. 若 modal answer 改变或 Beta gate 失败，trial 被判失败，相关 source/ratio 被 blacklist，并弹回上一个安全前缀；
3. 失败窗口不再进入后续 gate/source pool，但其生成 token 必须计费，其合法答案仍进入最终 parent-compatible vote；
4. 同一题连续两次失败后，清空 prefix stack，剩余预算退化为无前缀 SC 或原始 DeepConf，不再晋升；
5. 到达 80% 后不再延长，只能保持或回退。

这里的“效果下降”只能指可观测的 mode flip 或 Beta gate deterioration；推理阶段无法知道真实 accuracy 是否下降。高置信一致错误不能被纯 self-consistency 信号可靠识别，必须通过错误前缀分层、无前缀 audit-window 消融和 limitations 明示。

### 2.6 并行执行语义

- 一个逻辑窗口的 \(m=4\) 条轨迹拥有完全相同的 prefix token IDs；
- 所有结果返回前不更新 gate、prefix 或 blacklist，避免异步完成顺序改变算法；
- parent-paired 实验对 SC 与 PC 使用共同随机数：PC 尚处于 root 时直接复用同题 SC 对应窗口的完整轨迹；从首次 prefix proposal 起才分叉生成。统计中的算法等价 latency 仍计入被复用 root 窗口的原始耗时，并与实际执行 wall time 分列；**speedup 只报 `generation_seconds`（含复用 root），禁止把 `actual_generation_seconds` 当加速**；
- Qwen3-30B-A3B 若 KV 显存不足，可把一个逻辑窗口拆成两个 physical micro-batches of 2；两批仍共享同一前缀，完成四条后才更新状态；
- SC、DeepConf parent 与 PC 使用相同 physical batching 做 latency 对比；
- 同时报告 cache on/off、prefill tokens、decode tokens、wall-clock latency 和 peak KV memory，不能只用 decode-token saving 代表实际加速。

### 2.7 与 DeepConf 的组合

Qwen3-30B-A3B 上使用官方 DeepConf 语义：

1. \(N_{\mathrm{init}}=16\) 条无前缀 warmup traces，用于确定 DeepConf threshold；
2. total launched budget \(B=32\)，后续 16 条才启用 PC，避免前缀相关性污染 warmup；
3. DeepConf 控制 continuation 的在线停止与最终 confidence filtering/weighted vote；
4. PC 只控制每个后续窗口的 shared prefix；仅完整且可抽取答案的轨迹进入 Beta gate；
5. source trace 的 prefix confidence 只计算一次，不能在所有 child 中复制为独立证据；
6. standalone DeepConf 与 DeepConf+PC 的 warmup、预算、采样、聚合和 batch schedule 完全相同。

DeepSeek API 分两档：

- **P0：DeepConf-offline + PC**，从同一批完整 traces 复现 confidence filtering/weighted vote；
- **条件 P1：DeepConf-online + PC**。只有流式取消后仍能获得可信的 token usage/计费，且算法与官方实现核对通过，才可报告在线 token saving；否则不得使用 “online DeepConf” 名称。

公平 CPT 必须走官方 Qwen 入口 + A3B。Llama 上的 CPT adapter smoke **不是**公平比较。

### 2.8 干预式 Prefix Value Curve（可选分析，非测试时策略）

查找表控制器已取消。若仍做干预式 PVC，只用于分析图：

对每题先生成独立 discovery rollouts \(D\)，由冻结的 BLC 选择 source；随后对每个 \(r\in\{0,0.2,0.4,0.6,0.8\}\) 独立重采样 \(K\) 条 continuations，计算 \(V_{\mathrm{corr}}(r)\) 与 \(V_{\mathrm{agree}}(r)\)。gold 只用于事后计算 \(V_{\mathrm{corr}}\)，不得参与 source 或 length 选择。

Pilot 规模若做：每模型 50 题 × 5 ratios × \(K=4\)。不要为查找表去铺三模型全量。缺这项不阻止投稿。

---

## 3. 模型、数据集与 Prompt 冻结

### 3.1 模型

| 规模 | 冻结模型 | 运行方式 | 固定生成设置 |
|---|---|---|---|
| 小 | Meta-Llama-3.1-8B-Instruct | 本地 BF16，单卡；使用 ~/data 中现有 checkpoint | temperature 0.6，top-p 0.9，max new tokens **8192**（MATH/AIME/GPQA；不是 4096） |
| 中 | Qwen3-30B-A3B | 本地 BF16；**新服务器**跑主表。TP 必须整除 32 Q / 4 KV heads → 只允许 1/2/4 | thinking on；temperature 0.6，top-p 0.95，top-k 20，min-p 0，max new tokens 32768 |
| 大/API | deepseek-v4-flash | 官方 API，并发独立请求；**不必**走 Clash 代理 | thinking enabled，reasoning_effort=high，max tokens 32768；temperature/top-p 在 thinking mode 无效，不写作控制变量 |

说明：

- 文中“Llama-3.1-8B”统一指 Instruct checkpoint，因为评测依赖 chat template；base checkpoint 不混入同一表。Llama 没有 thinking 开关，只用 chat template + `\boxed{}`。禁止把 Llama CoT 格子和 Qwen thinking-on 格子混成「同一协议」。
- **两机分工（2026-09-09）：** 4090 实验室跑 Llama + DeepSeek；新服务器跑 Qwen3-30B-A3B。不要在 4090 上再开 A3B（GPU 1–3 已被 Llama 占用，A3B 需要 TP=4）。
- **2026-09-08 作者决定**：本地 Qwen 主模型是 Qwen3-30B-A3B，不是 32B。暂停的 32B V0 eager 490（fingerprint `9f54d1c4ba0d`）**不要续跑、不进 Table 1**。
- Qwen3-30B-A3B 在大显存机器上优先 TP=1 或 2；**不要**带 4090 上的 `num_gpu_blocks_override=8192` cap（会限制 KV）。启动：`export QWEN_GPU_BLOCKS=`（空）。
- DeepSeek API 没有可依赖的多样本 seed 或单请求 \(n\)；每条 branch 是独立并发请求，并记录请求时间、返回 model 字段、system fingerprint、usage 和 retry。

### 3.2 数据集

| 用途 | 数据集 | 规模 | 数据源与评分 |
|---|---|---:|---|
| 主表 | MATH-500 | 490（排除工程 10 题） | HuggingFaceH4/MATH-500；symbolic equivalence grader |
| 主表 | AIME 2024 | 30 | HuggingFaceH4/aime_2024；最终整数 exact match |
| 主表 | AIME 2025 | 30 | MathArena/aime_2025；最终整数 exact match |
| 主表 | GPQA-Diamond | 198 | 官方 Idavidrein/gpqa 的 gpqa_diamond；四选一 exact match |
| 附录连续性 | GSM8K test | 1319 | 现有 JSONL；Llama-3.1-8B 新协议已完成，可进 Table 1 连续性列 |
| ~~开发校准~~ | Dev-Mix | — | **已取消选 \(C^\*\) / PVC 查找表用途。** 不必为选阈再造 200 题集 |

当前 `sample_seed=2027` 抽出的 10 个 MATH-500 test 题已经用于 runner/门控调试，固定记为 **engineering pilot split**。`source_row`（0-index JSONL 行号）：`[303, 457, 359, 315, 316, 415, 407, 370, 151, 332]`。它们不参与任何超参选择，也从主文 MATH-500 统计中排除；主文报告其余 490 题。

选择理由：MATH-500、AIME24/25、GPQA-D 是 2025–2026 reasoning/TTS 工作最常见的一组数学与科学推理基准；AIME24+25 的 pooled 60 题作为主要 AIME 统计量，同时分年报告。LiveBench、LiveCodeBench、HMMT、AIME26 暂不进入两周 P0。

GPQA 官方数据是 gated asset；必须由用户接受条款后转成 JSONL，不能发布题目原文。所有数据在创建 CUDA context 前转换/读取为 JSONL，禁止 CUDA 后调用 pyarrow loader。

### 3.3 冻结 Prompt

MATH-500、AIME24、AIME25：

    {problem}

    Please reason step by step, and put your final answer within \boxed{}.

GPQA-Diamond：

    Answer the following multiple-choice question.

    Question:
    {question}

    Choices:
    (A) {choice_a}
    (B) {choice_b}
    (C) {choice_c}
    (D) {choice_d}

    Please reason step by step, and put the single correct option letter
    (A, B, C, or D) within \boxed{}.

执行约束：

- GPQA choices 用 question id 与 seed=2027 做一次确定性 shuffle，映射写入 JSONL；所有模型、方法、重复完全复用；
- 三个模型都使用各自 tokenizer 的原生 chat template，不手写 special tokens；
- Qwen3 明确 enable_thinking=true；连续前缀追加在尚未闭合的 assistant reasoning span；
- DeepSeek 必须使用官方 beta chat-prefix-completion 能力注入 reasoning prefix；普通多轮对话会忽略历史 reasoning_content，不能替代；
- 数学只解析最后一个合法 \(\boxed{}\)；MATH 做等价判断，AIME 做 0–999 整数判断，GPQA 只接受 A/B/C/D；
- parser failure、context overflow、API refusal、truncation分别计数，不能当作空答案后静默丢弃。

---

## 4. Baseline 选择与公平协议

### 4.1 正文必须出现

| 方法 | 角色 | 运行范围 |
|---|---|---|
| Pass@1 / CoT | 单轨迹参考点；可从 SC pool 第一条取得 | 三模型、四主数据集 |
| SC | 固定预算 parent | 三模型、四主数据集 |
| PC (C=0.8) | 新主方法 | 三模型、四主数据集 |
| DeepConf | 当前强 confidence controller / parent | Qwen3-30B-A3B 必做；DeepSeek offline 必做；Llama GSM8K Table 2 组合块 |
| DeepConf + PC | 组合主张 | 同上 |
| AC / ESC | 截断型 parent | Llama GSM8K Table 2 组合块（新协议 B=32）；Qwen 有余量再补 |
| AC+PC / ESC+PC | 组合主张 | 与各自 parent 配对 |
| CPT | 最近并行共享比较 | Qwen3-30B-A3B 的 AIME24/25、MATH-500 必做（公平）；Llama adapter 只占 Table 2 比较块 |
| PoLR | 最近前缀聚类比较 | Llama GSM8K Table 2 比较块（N=32，\(L_p=64/128/256\)；128 在跑） |

### 4.2 P1 与附录

- **DTS-Stable**：P1 现代结构化并行搜索基线；官方 vLLM 版本若能在 1 天内完成 Qwen3-30B-A3B smoke test，则在 AIME24/25 加入。不进 Table 2 主行。
- **Sentence-PC**：原始 \(B=20,m=5\) 版本放附录；Shared Sentence-PC 只在 matched-component 长度消融出现。旧 AC/ESC（B=20）不得与新协议 B=32 混算。
- **simple margin heuristic**：仅作便宜 Pareto 诊断，必须明确不是 MARS。
- **干预式 PVC**：可选分析图，不进主表方法列。
- **MARS、RASC、STAND、Prefix Consistency、LeaP**：Related Work 必须准确讨论；只有官方/忠实实现通过协议审计才给数字，不能用简化代理冒名。

### 4.3 统一预算与统计

- 主方法一般化表：\(B=32,m=4\)，C=0.8，相同 prompt、采样、最大输出、answer parser 和 launched-trace budget；
- DeepConf 表：\(N_{\mathrm{init}}=16,B=32\)；代表子集补 \(B\in\{16,32,64\}\)；
- 不再出现 PC@20 对 SC@40 的节省率；任意百分比只相对同模型、同数据、同预算 parent；
- CPT/DTS 的树或 chunk 结构不能强行按“轨迹数”等价，需同时给 matched launched budget、matched generated tokens/FLOPs 和 matched latency Pareto；
- 本地模型用 seed 0/1/2；DeepSeek API 做三次独立 repetition，因 API 无 seed 控制而不称为 seeds；
- AIME 主统计合并 24+25 后按 question 与 repetition 两层 bootstrap，分年数值仍保留；
- 主表报 mean 与 95% CI；附录补 mean±std、paired difference、有效样本数和 parser failure；
- 所有 warmup、probe、失败 trial、截断与重试 token 都进入总成本。

### 4.4 必报指标

1. accuracy、decode/reasoning tokens、prompt/prefix tokens、total billed/processed tokens；
2. end-to-end latency（`generation_seconds`，含复用 root）、throughput、P50/P95、peak KV memory、prefix-cache hit；
3. launched/completed/validated/voting traces 与 parser-failure rate；
4. 相对同 parent 的 \(\Delta\)accuracy、\(\Delta\)tokens、\(\Delta\)latency；
5. matched-accuracy token saving 与 accuracy-token Pareto；
6. gate pass、prefix level survival、wrong-prefix promotion、rollback/recovery；
7. distinct answers、pairwise agreement 与 source-prefix diversity；
8. Beta calibration/ECE 和高置信错误比例。

---

## 5. 主表初稿

表内 **TBD** 表示待实验；N/A 只用于接口上确实不可实现的设置。已完成格用控制台 `/api/state` 的 paper tables 为准，不要在本文件手填过期数字。

### Table 1：跨模型与任务的一般化（\(B=32\)，PC 为 C=0.8）

| Model | Method | MATH-500 Acc | Tok | AIME24 Acc | Tok | AIME25 Acc | Tok | GPQA-D Acc | Tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Inst. | Pass@1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
|  | SC | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
|  | **PC (C=0.8)** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen3-30B-A3B | Pass@1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
|  | SC | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
|  | **PC (C=0.8)** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DeepSeek-V4-Flash | Pass@1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
|  | SC | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
|  | **PC (C=0.8)** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

GSM8K 可作为附录连续性列写入控制台 Table 1（Llama-3.1-8B 与 DeepSeek 新协议已完成），不是四主现代数据集。DeepSeek GSM8K C=0.8 融合：Acc overlay 1277/1319；PC Tok 用 vote-v2（−34.2%）。正文 Tok 默认写平均 generated reasoning/decode tokens；latency 用 `generation_seconds`。

### Table 2：现代基线与组合（Llama-3.1-8B-Inst. × GSM8K，\(B=32\)）

MATH / AIME / GPQA 对 8B 过难，本表只报 GSM8K test 1319。Tok = 平均 decode generated tokens；Latency = `generation_seconds`（含配对根）。PC 为默认 C=0.8，不是 \(C^\*\)。

**比较块**（并行共享替代，宽度一律 N=32 / B=32）：SC-32、SC-20（只用于 matched-token）、PoLR（\(L_p=64 / 128 / 256\)）、CPT、PC。已完成的 CPT N=20 **不进本表**。Llama adapter **不是** Qwen thinking-on 公平 CPT。

**组合块**（截断型 parent ± PC）：DeepConf、DeepConf+PC、AC（τ=0.95）、AC+PC、ESC、ESC+PC；另报诊断 AC τ=0.99 + PC C=0 与 ESC 两窗/满 8。AC 的 0.95 是 Adaptive-Consistency 论文默认，不是 PC 前缀门控 C=0.8。旧 Sentence-PC AC/ESC（B=20）不进本表。DTS-Stable 仍是 P1，不占本表。CPT 的 Latency 是官方 `total_latency_sec`。**N=32 CPT 不降 decode token**（每 worker 仍写满一条，8,411 vs SC 8,316），提升的是墙钟（54.4 s→34.9 s）。

| Method | GSM8K Acc | Tok / q | Latency |
|---|---:|---:|---:|
| **比较** | | | |
| SC (B=32) | 92.12% (1215/1319) | 8,316 | 54.4 s |
| SC (B=20, same pool) | 91.28% (1204/1319) | 5,214 (−37.3%) | 34.2 s |
| PoLR (N=32, \(L_p=64\)) | 91.81% (1211/1319) | 4,826 (−42.0%) | 10.6 s |
| PoLR (N=32, \(L_p=128\)) | 92.19% (1216/1319) | 5,863 (−29.5%) | 11.2 s |
| PoLR (N=32, \(L_p=256\)) | 91.58% (1208/1319) | 7,554 (−9.2%) | 13.0 s |
| CPT (N=32, Llama adapter) | 92.80% (1224/1319) | 8,411 (+1.1%) | 34.9 s |
| **PC (C=0.8)** | 91.74% (1210/1319) | 5,039 (−39.4%) | 33.8 s |
| **组合** | | | |
| DeepConf (\(*\), freeze config) | TBD | TBD | TBD |
| DeepConf + PC | TBD | TBD | TBD |
| AC (τ=0.95) | 92.49% (1220/1319) | 2,730 (−67.2%) | 19.3 s |
| AC + PC | 92.27% (1217/1319) | 2,539 (−69.5%) | 18.0 s |
| ESC | 92.49% (1220/1319) | 2,552 (−69.3%) | 18.1 s |
| ESC + PC | 92.27% (1217/1319) | 2,374 (−71.5%) | 16.8 s |
| AC (τ=0.99) | 92.04% (1214/1319) | 4,035 (−51.5%) | 28.0 s |
| AC + PC (C=0) | 92.19% (1216/1319) | 3,003 (−63.9%) | 21.1 s |
| ESC (两窗或满 8) | 92.04% (1214/1319) | 4,299 (−48.3%) | 29.8 s |
| ESC + PC (两窗或满 8) | 92.04% (1214/1319) | 3,501 (−57.9%) | 24.4 s |

相对 SC-32，PC 准确率 −0.38 pp、decode −39.4%。相对等预算 SC-20，PC **+0.46 pp**（1204→1210），token 略少（5214→5039）。PoLR \(L_p=64\) Acc 1211、tok −42.0%；\(L_p=128\) Acc 1216、tok −29.5%；论文默认 \(L_p=256\) Acc 1208、tok −9.2%（本机 SC mean≈260，256 接近全文）。CPT N=32 Acc **1224/1319=92.80%**（比 SC +9 题），tok **8,411（+1.1%）**，时延 34.9 s；Llama adapter 不是 Qwen thinking-on 公平 CPT。CPT N=20（1218/1319，5,263 tok）是错误的 token-budget 对齐，已归档。默认 AC/ESC Acc 均为 1220/1319；+PC 各 −3 题、再省约 7% token（多数题第一窗 4/4 就停）。诊断行：AC τ=0.99 与 ESC 两窗/满 8 不再在 4 条时停，+PC 相对 parent 再省 25.6% / 18.6% token。DeepConf 未满 1319 不填。

### Table 3：parent-paired 增量

| Model / Dataset | Parent | +PC | Parent Acc | +PC Acc | \(\Delta\)Acc | Parent Tok | +PC Tok | Saving | \(\Delta\)Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B / AIME24+25 | SC | SC+PC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen3-30B-A3B / AIME24+25 | DeepConf | DeepConf+PC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen3-30B-A3B / GPQA-D | SC | SC+PC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen3-30B-A3B / GPQA-D | DeepConf | DeepConf+PC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DeepSeek / AIME24+25 | SC | SC+PC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DeepSeek / AIME24+25 | DeepConf-offline | +PC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Llama GSM8K 的 SC→SC+PC 可作为已完成对照写在控制台 Table 3 计划行之前，但不替代 Qwen/DeepSeek 现代主表。

---

## 6. 消融实验表格初稿

消融优先用已有 **Llama-3.1-8B × GSM8K 1319** 填主方法/SC 行；其余对照在 representative subset 上补。除阈值曲线外均使用默认 C=0.8。

### Ablation A：Beta 阈值

| \(C\) | Test Acc | Total Tok | Gate Pass % | Wrong Promotion % | Rollback % |
|---:|---:|---:|---:|---:|---:|
| 0.00（纯多数） | TBD | TBD | TBD | TBD | TBD |
| 0.70 | TBD | TBD | TBD | TBD | TBD |
| **0.80（默认操作点）** | 已有 GSM8K | 已有 | 已有 | 已有 | 已有 |
| 0.90 | TBD | TBD | TBD | TBD | TBD |
| 0.95 | TBD | TBD | TBD | TBD | TBD |
| 1.00（SC） | 已有 GSM8K | 已有 | — | — | — |

本表是敏感性复盘，**不用于挑 \(C^\*\)**。叙事：更低 C → 更大加速、更高错前缀风险；更高 C → 更接近 SC、更安全。

### Ablation A（DeepSeek-V4-Flash × GSM8K test 1319）

同一冻结协议（B=32，m=4，paired-root v2，thinking on）。Tok = 平均 decode generated tokens。每一行的 SC 来自**同一次** `sc,pc` 配对（C=0.8 Acc 为 first100+next500+rest719 overlay；C=0 为独立 tag `c00`）。**Acc 为格式容错复判。** C=0.8 效率列为 vote-v2（numeric-core 投票键，复用同一 SC 根）。SE(PC) 是二项标准误 \(\sqrt{p(1-p)/n}\)，不是多 seed 标准差。禁止从本表回写最优 C。

| \(C\) | SC Acc（配对） | PC Acc | \(\Delta\) (PC−SC) | SE(PC) | Total Tok | Gate % | Wrong Promo % | Rollback % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00（纯多数） | 97.27% (1283/1319) | 97.04% (1280/1319) | −0.23 pp | ±0.47 pp | 7,070 | 97.1% | 19.4% | 1.8% |
| 0.70 | 97.19% (1282/1319) | 97.12% (1281/1319) | −0.08 pp | ±0.46 pp | 7,399 | 93.7% | 24.0% | 1.6% |
| **0.80（默认操作点）** | 96.82% (1277/1319) | 96.82% (1277/1319) | 0.00 pp | ±0.48 pp | 8,230 | 98.4% | 2.8% | 0.4% |
| 0.90 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.95 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| 1.00（SC，同 C=0.8 root） | 96.82% (1277/1319) | = SC | 0.00 pp | ±0.48 pp | 12,514 | — | — | — |

C=0.8 为融合记录：Acc 仍是原配对 overlay（Δ=0），decode 用 vote-v2 **12514→8230（−34.2%）**。C=0.7 相对**自己的** SC 为 −0.08 pp、decode −25.2%（9898→7399）。C=0 相对自己的 SC 为 −0.23 pp。C=0.8 的 −34.2% / 98.4% Gate 来自投票键对齐，**不是**「比 C=0.7 更激进的阈值」；C=0 / C=0.7 的 Gate / Wrong Promo 仍是旧投票键，不要跟 C=0.8 这三列横比。跨行 PC 绝对值不要比。禁止从本表回写最优 C。

### Ablation B：前缀粒度与长度策略

| Length policy | Level schedule | Acc | Total Tok | Latency | Avg final ratio | Wrong Promotion % |
|---|---|---:|---:|---:|---:|---:|
| Shared Sentence-PC | +1 sentence / level；其余组件同主方法 | TBD | TBD | TBD | TBD | TBD |
| Original Sentence-PC | \(B=20,m=5\)，per-branch random | TBD | TBD | TBD | TBD | TBD |
| Absolute-token PC | +128 tokens / level | TBD | TBD | TBD | TBD | TBD |
| **Relative-PC（主方法）** | 20→40→60→80%；gate 通过则前进 | 已有 GSM8K | 已有 | 已有 | 已有 | 已有 |
| Fixed-20 | 始终 20% | TBD | TBD | TBD | TBD | TBD |
| Fixed-40 | 始终 40% | TBD | TBD | TBD | TBD | TBD |
| PVC Adaptive | ~~查找表 advance/stay~~ | **已取消** | — | — | — | — |
| PVC Adaptive (agreement only) | ~~无标签查找表~~ | **已取消** | — | — | — | — |

### Ablation C：DeepSeek MATH-500 选择 × recovery（2×2）

替换原 Llama GSM8K 来源选择占位表。共享 SC 根 `4856d4ef5673`。Acc = raw JSONL，**不是** numeric_regrade overlay（452/490）。不进 Table 1。长度干预 R=8 complete，已进 Ablation G（`2ebec2c2eced`）。

| Setting | PC Acc（raw） | Δ vs SC | Tok / q | vs SC | Gate Pass | Wrong Promo | Fail windows | Fail-set Acc |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| SC（共享根，无前缀） | 87.76% (430/490) | 0.00 pp | 41,166 | — | 0.0% | 0.0% | 0 | — |
| Uniform + recovery on（臂 B） | 86.94% (426/490) | −0.82 pp | 25,680 | −37.6% | 98.2% | 11.6% | 58 rollback | 41.2% (14/34) |
| Uniform + recovery off（臂 D） | 86.33% (423/490) | −1.43 pp | 25,405 | −38.3% | 98.2% | 11.3% | 161 retain | 29.0% (9/31) |
| **BLC + recovery on（主方法 · 臂 A）** | 87.35% (428/490) | −0.41 pp | 26,968 | −34.5% | 98.2% | 12.4% | 17 rollback | 42.9% (6/14) |
| BLC + recovery off（臂 C） | 86.33% (423/490) | −1.43 pp | 24,757 | −39.9% | 98.2% | 11.5% | 141 retain | 30.3% (10/33) |

相对同一 SC：A −2（0/2），B −4（2/6），C −7（4/11），D −7（4/11）。D 与 C 同为 423 题，但是 4/4 翻转。交互 (D−A)−(B−A)−(C−A)=+2，伤害略小于相加。McNemar 精确双侧 p 均 >0.1。

### Ablation D：共享方式与系统收益

| Prefix mode | Acc | Decode Tok | Prefill Tok | Latency | Cache hit | Distinct answers |
|---|---|---:|---:|---:|---:|---:|
| Per-branch random（原算法） | TBD | TBD | TBD | TBD | TBD | TBD |
| Shared random | TBD | TBD | TBD | TBD | TBD | TBD |
| Shared token-medoid | TBD | TBD | TBD | TBD | TBD | TBD |
| **Shared BLC（主方法）** | 已有 GSM8K | 已有 | 已有 | 已有 | 已有 | 已有 |

### Ablation E：DeepSeek MATH-500 recovery / 回退

替换原 Llama GSM8K 回退占位表。与 Ablation C 同一四臂。

| Setting | PC Acc（raw） | Δ vs SC | Tok / q | Fail windows | Fail-set Acc | Fail-set vs SC | Disabled q | High-conf. wrong % |
|---|---:|---:|---:|---|---:|---|---:|---:|
| Uniform + recovery on（臂 B） | 86.94% (426/490) | −0.82 pp | 25,680 | 58 rollback | 41.2% (14/34) | SC 17/34 | 11 | 8.8% |
| Uniform + recovery off（臂 D） | 86.33% (423/490) | −1.43 pp | 25,405 | 161 retain | 29.0% (9/31) | SC 14/31 | 0 | 8.4% |
| **BLC + recovery on（主方法 · 臂 A）** | 87.35% (428/490) | −0.41 pp | 26,968 | 17 rollback | 42.9% (6/14) | SC 7/14 | 3 | 9.4% |
| BLC + recovery off（臂 C） | 86.33% (423/490) | −1.43 pp | 24,757 | 141 retain | 30.3% (10/33) | SC 15/33 | 0 | 9.0% |

recovery 关从不 two-strike；fail-set 相对该子集 SC 为 A −1、B −3、C −5、D −5。High-conf. wrong 不是区分因素。

### Ablation F：SC / DeepConf 与 PC 的 2×2 因子

| Trace controller | Prefix redirection | Acc | Decode Tok | Total Tok | Latency | Completed traces |
|---|---|---:|---:|---:|---:|---:|
| SC | off | 已有 GSM8K | 已有 | 已有 | 已有 | 已有 |
| SC | PC | 已有 GSM8K | 已有 | 已有 | 已有 | 已有 |
| DeepConf | off | TBD（A3B） | TBD | TBD | TBD | TBD |
| DeepConf | PC | TBD（A3B） | TBD | TBD | TBD | TBD |

DeepConf 10 题 Llama smoke **不填**本表。

### Ablation G：Prefix Value Curve 数据表（可选）

| Ratio | \(V_{\mathrm{corr}}\) | \(V_{\mathrm{agree}}\) | Correct-source \(V\) | Wrong-source \(V\) | Suffix Tok | Distinct answers |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | TBD | TBD | TBD | TBD | TBD | TBD |
| 20% | TBD | TBD | TBD | TBD | TBD | TBD |
| 40% | TBD | TBD | TBD | TBD | TBD | TBD |
| 60% | TBD | TBD | TBD | TBD | TBD | TBD |
| 80% | TBD | TBD | TBD | TBD | TBD | TBD |

若做，主图画 \(V_{\mathrm{corr}}/V_{\mathrm{agree}}\) 与 token cost；不由此生成测试时控制器。缺表不阻止投稿。

---

## 7. 实现与可信度清单

### 7.1 P0 代码修复

- [x] confidence.py 统一为 top-1-vs-all-rest Beta 后验（以当前 runner 为准；单测仍应补）
- [x] 配对根 v2、trial/commit、单层回退、两击重置、有 logprobs 时 BLC
- [x] 固定 20% 网格长度策略；**明确不做 PVC 查找表**
- [ ] 删除未 seeded 的 random.choice；每个问题使用可复现独立 RNG；
- [ ] 实现 reasoning/final-answer span 分离、token-ratio endpoint 与 leakage tests；
- [ ] 保存每个 level 的 source id、prefix token IDs、ratio、BLC score、gate 前后状态和 rollback reason；
- [x] 稳定 question id、resume 汇总、tag/config hash；修改 flags 后不得复用旧 JSONL；
- [x] summary 必须从完整 JSONL 重算；
- [x] 对 DeepConf/CPT 建隔离环境到 ~/data 下（Llama smoke 已做；公平比较仍须 A3B）；
- [x] 所有大结果、API cache 和临时数据写入 ~/data，仓库只保留小型汇总。

### 7.2 G0 接口 smoke tests

- [x] Llama-3.1-8B：chat template、logprobs、prefix continuation；GSM8K 全量已完成
- [ ] Qwen3-30B-A3B：在**新服务器** TP 合法值启动；不要在 4090 实验室抢 Llama 的卡
- [x] DeepSeek-V4-Flash：prefix completion + reasoning logprobs（MATH-500 490 已完成）
- [x] GPQA JSONL、确定性 choice shuffle（本机 Llama/DeepSeek 正在跑）
- [ ] CPT 官方代码先复现一个 Qwen 小样例，再允许适配 Qwen3-30B-A3B；若复现失败，不能写“公平比较完成”。

### 7.3 实验分层与砍项顺序

| 层级 | 必须完成 | 若时间不足如何处理 |
|---|---|---|
| P0-A | 三模型的 SC 与 PC（C=0.8）；四主数据集 seed/repetition 0 | 不可砍；缺失则主张不完整 |
| P0-B | Qwen3-30B-A3B 的 DeepConf、DeepConf+PC、官方 CPT | 不可用 AC/ESC 或 Llama CPT adapter 代替 |
| P0-C | Beta 阈值 sweep、length/selector/rollback 关键对照 | 每项至少 representative subset；**PVC 查找表砍掉** |
| P0-D | 主表核心行 seed/repetition 1/2、CI、latency | 先补主方法，不补已明显失败配置 |
| P1 | DTS-Stable、DeepSeek online DeepConf、干预式 PVC 分析图、budget scaling | 接口一天未通即降级并如实写明 |
| P2 | LiveBench/HMMT/AIME26、十个旧任务重跑、复杂理论 | 全部砍掉 |

---

## 8. 从 9 月 6 日开始的排期（v4 修订）

### 9/6：冻结方法骨架

- [x] 冻结模型、数据集、prompt、Beta 主门控、20% ratio grid、shared-prefix 语义；
- [x] 确定 BLC source selector、trial/commit 和 one-level rollback；
- [x] 确定 CPT/DeepConf 为必做现代对照。

### 9/6–9/8：runner、数据与接口

- [x] 新协议 runner（配对根 v2、B=32、m=4）；
- [x] MATH-500、AIME24、AIME25、GPQA-D JSONL（本机）；
- [x] Llama-3.1-8B GSM8K 1319 全量 SC/PC（C=0.8）；
- [x] DeepSeek-V4-Flash MATH-500 490；
- [x] 隔离环境 DeepConf / CPT Llama smoke（CPT **不公平**，不进主表）。

**G0：** Qwen 主表改到新服务器；DeepSeek reasoning-prefix 已通。

### 9/9：协议 v4 与本机主表（进行中）

- [x] **取消 Dev-Mix 选 \(C^\*\)；取消 PVC 查找表控制器；默认 C=0.8。**
- [ ] Llama-3.1-8B MATH-500 490（GPU 1，`math500_llama31_b32_mt8192_c08_seed0`）——**不要杀**；
- [ ] Llama GPQA-D 198（GPU 2）——不要杀；
- [ ] Llama AIME 2024→2025（GPU 3）——不要杀；
- [ ] DeepSeek AIME24/25 + GPQA API（不占卡）——完成与否以 `*.run.json` / JSONL 题数为准。

### 立即：新服务器 Qwen P0-A

- [x] 4090 实验室 first100（JSONL `source_row` 0–99）complete：fingerprint `630825247308`，SC=PC 94/100。
- [x] next100（`source_row` 100–199，排除工程 151，n=99）complete：fingerprint `664997212910`，SC=PC 94/99。Table 1 部分格已改为合并 **n=199**（188/199，decode −41.3%）。
- [ ] Qwen3-30B-A3B × MATH-500 490，paired SC/PC，C=0.8，thinking on。入口：`bash run_math500_qwen30a3b.sh`（先改环境变量，见 `handoff.md`）；
- [ ] 同一模型 × AIME24/25、GPQA-D。

全量 490 不要在 4090 上再开。n=199 不得冒充 490，全量到位后替换 Table 1 该格。

### 9/10–9/14：P0-B 现代对照

- [x] Table 2 分工：比较 = PoLR / CPT；组合 = DeepConf / AC / ESC（各 ±PC）。DTS 仍 P1。
- [x] 新协议 AC / ESC / AC+PC / ESC+PC 实现（`run_math.py --methods ac,pc_ac` / `esc,pc_esc`）；启动脚本 `scripts/run_llama31_gsm8k_ac_esc.sh`。
- [x] Llama GSM8K AC / AC+PC / ESC / ESC+PC 满 1319，已填 Table 2；
- [ ] Qwen3-30B-A3B：DeepConf、DeepConf+PC、官方 CPT；
- [ ] 完成 Table 1/2/3 的第一版和 accuracy-token Pareto；
- [ ] 每天检查 JSONL 完整性、失败率、API 费用和 GPU orphan process。

**G1（不再卡 PVC）：** 主表至少有「两模型 × 两数据集」的完整 SC/PC 配对，且口径一致。BLC vs shared-random 消融可后补，不作为冻结 PC 的前提。

### 9/14–9/17：重复、消融与统计

- [ ] 主表关键行补 seed/repetition 1/2；
- [x] Ablation A DeepSeek GSM8K 补 C=0.7（C=0.8 与 C=0 已有；C∈{0.9, 0.95} 仍空）；
- [x] DeepSeek GSM8K C=0.8 融合写入 Table 1 / Ablation A（Acc overlay 1277/1319，Tok vote-v2 −34.2%）；
- [x] selector / rollback representative subset（DeepSeek MATH-500 A/B/C/D 已替换 Ablation C/E）；
- [x] matched-history 长度干预（DeepSeek MATH-500，R=8 complete，进 Ablation G；未写 PVC）；
- [ ] **不做** PVC 查找表；干预式 PVC 分析图仅在有余量时；
- [ ] 若有余量，跑 DTS-Stable 与 \(B\in\{16,32,64\}\)；
- [ ] 9/16 冻结数字，9/17 完成独立重算和 CI。

**G2（9/16 晚）**：若 PC 对 SC 无稳定非劣收益、DeepConf 组合无增量或 CPT 明显支配，应缩主张或停止投稿，不能靠测试集阈值、删 baseline 或只报 decode tokens 补救。

### 9/15–9/18：摘要与主体同步写作

- [ ] 摘要只写已冻结、可复算的 parent-relative 数字；
- [ ] 引言写 shared-prefix controller 与 CPT 的边界；**不要**把 PVC 查找表或 \(C^\*\) 写成已做贡献；
- [ ] 方法替换句子级/random-source/reset 语义，加入算法伪代码；长度节写固定 20% 网格；
- [ ] 删除无效 Taylor guarantee 和附录 theorem；
- [ ] 评测改为现代模型/数据/parent-paired 表；主表方法名 **PC (C=0.8)**；
- [ ] 9/18 23:59 AOE 前提交真实摘要与完整作者。

### 9/19–9/25：全文缓冲周

- [ ] 9/19–9/21：只补预登记 P1、失败模式、limitations 和 appendix；
- [ ] 9/22：迁移并冻结 ICLR 2027 模板，正文压到 9 页；
- [ ] 9/23：匿名代码说明、config manifests、数据来源和复现命令；
- [ ] 9/24：公式、表格、引用、匿名性、AI-use policy、PDF 独立审计；
- [ ] 9/25 23:59 AOE 前提交全文和 supplement。

---

## 9. 投稿前必须回答的审稿问题

- \(C\) 如何选？答：**不选 \(C^\*\)。** 主表预先声明 C=0.8（原论文 / AC 常用操作点）。sweep 只证明敏感性：更低 C 更快、更高 C 更安全。禁止测试集 oracle。
- “adaptive” 在哪里？答：**测试时没有 PVC adaptive。** 长度是固定 20% 网格，gate 通过则前进一档；回退处理失败 trial。
- 为什么唯一前缀不是随意的？答：Beta 选答案簇，BLC 选簇内局部最可靠路径；随机选择有直接消融。
- 坏前缀如何取消？答：trial/commit、单层 stack rollback、blacklist、两次失败退化为 parent。
- 为什么还需要 PC，DeepConf / AC / ESC 已经省 token？答：截断型停止低价值或已一致的轨迹，PC 缩短仍要生成的轨迹；用 parent vs parent+PC 验证而非口头声称。
- 与 CPT / PoLR 有何不同？答：连续因果前缀 + answer-level posterior + rollback，对比 CPT 的信息池提取/广播和 PoLR 的前缀聚类剪枝；必须有 matched-width 数字。不要靠 PVC 查找表制造差异。
- 并行和 cache 是否真的带来速度？答：同 level 单一 token-identical prefix，报告 `generation_seconds`（含复用 root）以及真实 prefill/decode/KV，而非只报告理论 token。
- Beta 是否保证正确？答：不保证；它只度量众数占优，依赖性与高置信错误作为 limitations 和分层实验。

---

## 10. 主要外部依据

- ICLR 2027 Call for Papers：https://www.iclr.cc/Conferences/2027/CallForPapers
- Qwen3-30B-A3B 官方模型卡与推荐采样：https://huggingface.co/Qwen/Qwen3-30B-A3B
- DeepConf 官方实现：https://github.com/facebookresearch/deepconf
- CPT 论文与官方实现：https://arxiv.org/abs/2605.27030 ，https://github.com/WangXinglin/CPT
- DTS 官方实现：https://github.com/ZichengXu/Decoding-Tree-Sketching
- DeepSeek thinking / prefix API：https://api-docs.deepseek.com/guides/thinking_mode/ ，https://api-docs.deepseek.com/guides/chat_prefix_completion/
- MATH-500：https://huggingface.co/datasets/HuggingFaceH4/MATH-500
- AIME24：https://huggingface.co/datasets/HuggingFaceH4/aime_2024
- AIME25：https://huggingface.co/datasets/MathArena/aime_2025
- GPQA：https://huggingface.co/datasets/Idavidrein/gpqa
- 本仓库协议与两机分工：`AGENTS.md`、`handoff.md`
- 2026-09-09 决策原文：`paper/report_ICLR2027_revision.md` §12.14

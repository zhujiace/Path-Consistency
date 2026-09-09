# ICLR 2027 两周实验与修稿计划 v3

> 版本日期：2026-09-06。目标是在 9 月 18 日摘要截止前冻结方法、题目和核心证据；9 月 25 日全文截止前只补预先登记的实验、写作与审计。官方截止时间均为 23:59 AOE。
>
> 本计划依据论文原稿、ICML/NeurIPS 审稿意见、当前代码审计及 2026 年最新并行推理论文制定。三个主模型已经位于 ~/data，不安排模型下载。

---

## 0. 最终研究决策

### 0.1 一句话主线

保留原论文“多路径采样 → Beta 门控 → 前缀复用 → 聚合”的骨架，将方法升级为：

> **PC（Path-Consistency）用答案级 Beta 后验决定是否共享，用 Prefix Value Curve 决定共享多长，用局部置信度从多数答案簇中选择唯一共享前缀，并在门控恶化时单层回退；它既可增强固定预算 SC，也可叠加到 DeepConf。**

### 0.2 对当前意见的裁决

| 问题 | 最终决定 | 原因与边界 |
|---|---|---|
| Beta 门控还是简单投票 | **Beta 为主，简单多数票仅作消融** | Beta 给出多数答案概率超过 0.5 的后验统计；简单投票无法度量证据强弱。但 Beta 只支持“答案占优”，不证明答案或前缀正确。 |
| 主表是否展示阈值 sweep | **不展示 sweep，只展示开发集选出的 \(C^\*\)** | “最优”必须在独立开发集选择并冻结，不能取测试集最好结果。阈值曲线放消融/附录。 |
| 自适应前缀 | **采用 20% 网格 + PVC 控制器** | 名义 level 为 0%→20%→40%→60%→80%；每次最多前进 20%，可停留或回退。固定比例序列是对照，不称为 adaptive。 |
| 每个 level 的并行语义 | **一个 level 只有一个前缀，所有分支共享** | 这是主算法而非部署消融；逻辑窗口可拆成物理 micro-batch，但状态只在整个窗口完成后更新。 |
| 与截断型方法的关系 | **不再作为论文中心，但不能全部删除** | 主叙事是 parent 与 parent+PC；DeepConf 是核心 parent。AC/ESC 只留附录连续性对照。 |
| 最新强基线 | **CPT 为必须比较，DeepConf 为必须组合；DTS-Stable 为 P1** | 2026-05 的 CPT 与“并行分支共享中间信息”高度重叠，若不定量比较，创新性论证不成立。 |
| 前缀下降后的处理 | **候选先试用、后提交；失败回退一层** | 推理时看不到真实准确率，只能用 modal-answer 改变或 Beta 门控失效作为可观测代理；高置信一致错误仍是明确限制。 |

### 0.3 Go / no-go 总标准

只有同时满足以下条件，才按当前主线投稿：

1. PC 相对同预算 SC 在至少两个模型、两个现代数据集上准确率非劣（允许界 \(\epsilon=1.0\) percentage point），并有稳定 token 或 latency 收益；
2. DeepConf+PC 相对相同 DeepConf parent 至少在 Qwen3-30B-A3B 的两个设置上继续产生增量；
3. PC 在 matched-budget 或 matched-latency 下不劣于最接近的 CPT；
4. 干预式 PVC 能解释何时前缀有益、何时错误锁定，而不是仅展示相关性；
5. 代码、公式、门控、回退和表格口径完全一致。

若第 3 条失败，应把贡献缩为“Beta/PVC 驱动的连续前缀重定向分析”，不得声称并行共享方法整体领先。若第 1/2 条也失败，建议不提交本轮 ICLR。

---

## 1. 最小改稿策略与新增贡献

### 1.1 保留与修改

| 论文部分 | 保留 | 最小必要修改 |
|---|---|---|
| 问题与框架 | SC 中重复生成、窗口式多路径推理、训练后无需更新参数 | 把定位改为“有损、受门控的计算重定向”，不再笼统声称无损或 model-agnostic |
| Beta 门控 | top-1-vs-all-rest 的 Beta 后验 | 统一公式和代码；明确相关分支使独立同分布假设近似化 |
| 多级前缀 | 逐级增加复用信息 | 句子数改为归一化 token 比例；增加可停留、候选验证和单层回退 |
| 前缀来源 | 多数答案对应轨迹 | 随机抽样改为答案置信 + 局部 token 置信的唯一前缀选择 |
| 最终聚合 | SC majority vote | DeepConf parent 保持其官方 filtering/weighted vote，不偷换为普通投票 |
| 理论 | Beta 后验解释 | 删除无效 Taylor “不降准确率”推导和错误 theorem；改成风险分解与可检验假设 |
| 旧实验 | 作为历史连续性 | 十个旧任务和原始 Sentence-PC 移附录；新主表只用修复后协议 |

### 1.2 投稿版四项贡献

1. **Prefix Value Curve（PVC）**：用独立 discovery/evaluation rollouts 做因果式前缀干预，刻画前缀长度的准确率—成本—多样性曲线。
2. **Adaptive normalized prefixing**：跨短 CoT 与长 reasoning trace 使用 20% 比例网格，由开发集 PVC 决定 advance/stay。
3. **Beta-guided Local-Confidence selection（BLC）**：Beta 先确定可靠答案簇，再从该簇选择局部最可靠的一条连续前缀，使每个 level 可被完全并行执行。
4. **Composable control**：在相同 parent 下验证 SC→SC+PC、DeepConf→DeepConf+PC，并用回退降低错误前缀锁定。

### 1.3 与 2026 最近工作的区分

- **DeepConf** 控制低置信轨迹的停止与最终投票；PC 决定后续轨迹从哪个连续推理前缀继续。两者控制不同维度，因此做严格 2×2 组合。
- **CPT** 抽取、去重并广播多个“信息单元”，保留各分支私有历史；PC 只选择一个模型原生、连续、可缓存的因果前缀，并用 Beta/PVC/回退控制风险。机制有差异，但研究问题高度重叠，CPT 是必须定量比较的最近邻。
- **DTS** 动态扩展/筛选解码树；PC 维持窗口式并行采样，不做 token-level 树搜索。若官方实现能在冻结协议下适配 Qwen3-30B-A3B，则列为现代结构化搜索基线。

---

## 2. 冻结后的 PC 方法

### 2.1 主设置与状态

- 主预算：每题最多 \(B=32\) 条 launched traces；
- 逻辑窗口：\(m=4\)，共 8 个窗口；
- 前缀比例：\(\mathcal R=\{0,0.2,0.4,0.6,0.8\}\)；
- 最大复用：80%，避免把最终结论或 answer span 当作前缀；
- 主门控：cumulative valid completed traces 上的 Beta gate；
- 最终聚合：SC parent 用 majority；DeepConf parent 用其原始 confidence filtering/weighted vote；
- 保存两套答案集合：validated answers 用于后续控制，all completed answers 用于最终 parent-compatible 聚合。失败窗口的 token 和答案不得从成本或最终结果中静默删除。

原论文 \(B=20,m=5\)、per-branch random source 的 Sentence-PC 只作为附录连续性基线；不得与 \(B=32\) 的结果直接计算节省率。长度消融中的 Shared Sentence-PC 使用 \(B=32,m=4\) 和相同 BLC/rollback，仅把 token-ratio 换回“每层 +1 sentence”，从而隔离长度策略。

### 2.2 Beta 门控与阈值选择

设当前 validated answers 共 \(n\) 条，众数答案获得 \(v_m\) 票，采用均匀先验：

\[
\operatorname{conf}(n,v_m)
=\Pr(p_m>0.5\mid v_m,n)
=1-I_{0.5}(v_m+1,n-v_m+1).
\]

当 \(\operatorname{conf}\ge C^\*\) 时才允许提出新前缀。这衡量“众数答案占优”的后验置信，不等价于答案正确率，更不构成前缀无害保证。

**开发集选择 \(C^\*\)：**

1. 固定 Dev-Mix：GSM8K train 中 100 题 + MATH train 中 100 题，seed=2027；与全部测试集去重；
2. 每个模型只选择一次 \(C^\*\)，并跨 MATH-500、AIME24、AIME25、GPQA-D 固定；
3. 候选为 \(\{0.70,0.80,0.90,0.95,1.00\}\)；\(C=0\) 只作为“纯多数门控”消融；
4. 可行条件：相对 SC 的 paired-bootstrap 单侧 95% 下界不低于 \(-1.0\) pp；
5. 在可行候选中选择平均总 token 最少者；平局依次选择准确率更高、\(C\) 更大者；
6. 若不存在 \(C<1\) 的可行值，记录该模型设置失败，不能用测试集 oracle 值替代。

主表只写 **PC (\(C^\*\), dev-selected)**。完整阈值曲线仅进消融。

### 2.3 自适应前缀长度

**候选构造：**

1. 先从完整 trace 删除 final-answer span；Qwen/DeepSeek 只从 reasoning span 取前缀；
2. 设剩余 token 长度为 \(L\)，比例 \(r\) 的候选终点为 \(\lfloor rL\rfloor\)；
3. 在终点前最多回看 32 tokens，优先吸附到换行、空白或安全标点；不得切入 LaTeX 环境、代码块、\(\boxed{}\)、Final Answer 或答案解释段；
4. 保存原始 token IDs、吸附前后位置和 mask reason，禁止只靠字符串 “answer is”。

**level 的具体行为：**

| 当前安全比例 | 下一候选比例 | 允许动作 |
|---:|---:|---|
| 0% | 20% | gate 通过且 PVC 认为安全时试用；否则保持 0% |
| 20% | 40% | advance 或 stay |
| 40% | 60% | advance 或 stay |
| 60% | 80% | advance 或 stay |
| 80% | — | 保持 80%，只继续采样或回退 |

因此 Level 1/2/3/4 的名义长度分别为 20%/40%/60%/80%，但 PC 每个窗口最多前进一个 level，可以停留；固定的 20→40→60→80 序列命名为 **Relative-PC**，用于证明收益不是“比例切分”本身造成。

**PVC controller：**

- 在 Dev-Mix 上按模型分别建立查找表；
- 可观测状态为当前比例、Beta confidence tertile、source masked-length tertile；
- 对每个状态估计 \(V_{\mathrm{corr}}(r)\) 和 suffix cost；
- 仅当候选比例相对无前缀的准确率差下置信界不低于 \(-1.0\) pp 且 continuation token 更少时 advance；
- cell 少于 30 个干预样本时，依次回退到“仅 confidence bin”及全局规则；仍无可靠估计则 stay；
- 测试时只查冻结表，不访问 gold、不重新拟合。

另做无标签版本：以 \(V_{\mathrm{agree}}\) 替代 \(V_{\mathrm{corr}}\)。它只能作为消融；若表现相当，可将最终方法改为无需标签校准。

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

1. 若 modal answer 未改变且 Beta confidence 仍不低于 \(C^\*\)，trial 被提交并压入 prefix stack；
2. 若 modal answer 改变或 Beta gate 失败，trial 被判失败，相关 source/ratio 被 blacklist，并弹回上一个安全前缀；
3. 失败窗口不再进入后续 gate/source pool，但其生成 token 必须计费，其合法答案仍进入最终 parent-compatible vote；
4. 同一题连续两次失败后，清空 prefix stack，剩余预算退化为无前缀 SC 或原始 DeepConf，不再晋升；
5. 到达 80% 后不再延长，只能保持或回退。

这里的“效果下降”只能指可观测的 mode flip 或 Beta gate deterioration；推理阶段无法知道真实 accuracy 是否下降。高置信一致错误不能被纯 self-consistency 信号可靠识别，必须通过错误前缀分层、无前缀 audit-window 消融和 limitations 明示。

### 2.6 并行执行语义

- 一个逻辑窗口的 \(m=4\) 条轨迹拥有完全相同的 prefix token IDs；
- 所有结果返回前不更新 gate、prefix 或 blacklist，避免异步完成顺序改变算法；
- parent-paired 实验对 SC 与 PC 使用共同随机数：PC 尚处于 root 时直接复用同题 SC 对应窗口的完整轨迹；从首次 prefix proposal 起才分叉生成。统计中的算法等价 latency 仍计入被复用 root 窗口的原始耗时，并与实际执行 wall time 分列；
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

### 2.8 干预式 Prefix Value Curve

对每题先生成独立 discovery rollouts \(D\)，由冻结的 BLC 选择 source；随后对每个 \(r\in\{0,0.2,0.4,0.6,0.8\}\) 独立重采样 \(K\) 条 continuations：

\[
V_{\mathrm{corr}}(r)
=\Pr(\operatorname{answer}(Y)=y^\*
\mid q,\operatorname{do}(P_{\pi(D),r})),
\]

\[
V_{\mathrm{agree}}(r)
=\Pr(\operatorname{answer}(Y)=a_m(D)
\mid q,\operatorname{do}(P_{\pi(D),r})).
\]

gold 只用于事后计算 \(V_{\mathrm{corr}}\)，不得参与 source 或 length 选择；discovery 与 evaluation 样本不得复用。每个点同时报告：

- \(V_{\mathrm{corr}}\)、\(V_{\mathrm{agree}}\) 及按题 cluster-bootstrap 95% CI；
- suffix decode tokens、prefix/prefill tokens、总 tokens、latency；
- distinct answers、pairwise agreement、有效答案数；
- correct-source 与 wrong-source 两条曲线；
- easy/medium/hard 分层及 rollback 后恢复率。

Pilot 为每模型 50 题 × 5 ratios × \(K=4\)；最终 PVC 只在 Qwen3-30B-A3B 的 MATH-500 100 题和 AIME24+25 全 60 题做 \(K=8\)，避免三模型全铺导致预算失控。

---

## 3. 模型、数据集与 Prompt 冻结

### 3.1 模型

| 规模 | 冻结模型 | 运行方式 | 固定生成设置 |
|---|---|---|---|
| 小 | Meta-Llama-3.1-8B-Instruct | 本地 BF16，单卡；使用 ~/data 中现有 checkpoint | temperature 0.6，top-p 0.9，max new tokens 8192 |
| 中 | Qwen3-30B-A3B | 本地 BF16，**vLLM TP=4 V1 CUDA-graph**；checkpoint `~/data/Qwen3-30B-A3B` | thinking on；temperature 0.6，top-p 0.95，top-k 20，min-p 0，max new tokens 32768 |
| 大/API | deepseek-v4-flash | 官方 API，并发独立请求 | thinking enabled，reasoning_effort=high，max tokens 32768；temperature/top-p 在 thinking mode 无效，不写作控制变量 |

说明：

- 文中“Llama-3.1-8B”统一指 Instruct checkpoint，因为评测依赖 chat template；base checkpoint 不混入同一表。
- 三个主模型均已在 ~/data 或 API 配置中，不安排下载；只做路径、shard、tokenizer 和最小生成 smoke test。
- **执行顺序冻结**：方法调试、协议排错和首轮 MATH500 pilot 先用 Llama-3.1-8B-Instruct 单卡完成；通过准确率—成本门后，再把冻结实现迁移到 Qwen3-30B-A3B。
- Qwen3-30B-A3B 后续推理固定使用 vLLM TP=4（32 Q heads / 4 KV heads；BF16 在 24 GiB 上 TP=2 放不下权重）。引擎为 **V1 + CUDA graph + CUDA-12.2 NCCL**（`VLLM_NCCL_SO_PATH`）。逻辑窗口 4；若 32k KV 不够再拆 physical micro-batch=2。
- **2026-09-08 作者决定**：本地 Qwen 主模型由 Qwen3-32B 换成 Qwen3-30B-A3B。32B dense 的 eager/graph 记录只作 serving 考古，不进入主表。不得把 A3B 的 tok/s 与 32B 混比。
- DeepSeek API 没有可依赖的多样本 seed 或单请求 \(n\)；每条 branch 是独立并发请求，并记录请求时间、返回 model 字段、system fingerprint、usage 和 retry。

### 3.2 数据集

| 用途 | 数据集 | 规模 | 数据源与评分 |
|---|---|---:|---|
| 主表 | MATH-500 | 500 | HuggingFaceH4/MATH-500；使用官方/兼容 symbolic equivalence grader |
| 主表 | AIME 2024 | 30 | HuggingFaceH4/aime_2024；最终整数 exact match |
| 主表 | AIME 2025 | 30 | MathArena/aime_2025；最终整数 exact match |
| 主表 | GPQA-Diamond | 198 | 官方 Idavidrein/gpqa 的 gpqa_diamond；四选一 exact match |
| 开发校准 | Dev-Mix | 200 | GSM8K train 100 + MATH train 100，seed=2027，与测试题去重 |

当前 `sample_seed=2027` 抽出的 10 个 MATH-500 test 题已经用于 runner/门控调试，固定记为 **engineering pilot split**。它们不参与阈值、PVC 或 selector 选择，也从主文 MATH-500 test 统计中排除；主文报告其余 490 题，附录可在全部配置冻结后补完整 500 题结果，并明确 10 题曾被查看。所有方法超参数仍只在 Dev-Mix 冻结。
| 附录连续性 | GSM8K test | 1319 | 复用现有 JSONL 与旧论文 8-shot prompt |

选择理由：MATH-500、AIME24/25、GPQA-D 是 2025–2026 reasoning/TTS 工作最常见的一组数学与科学推理基准；AIME24+25 的 pooled 60 题作为主要 AIME 统计量，同时分年报告。LiveBench、LiveCodeBench、HMMT、AIME26 暂不进入两周 P0，因为会新增动态版本、代码执行或额外 evaluator 变量。

GPQA 官方数据是 gated asset；必须由用户接受条款后转成 ~/data/Datasets/gpqa_diamond.jsonl，不能发布题目原文或使用来源不明的镜像。所有数据在创建 CUDA context 前转换/读取为 JSONL，禁止 CUDA 后调用 pyarrow loader。

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
| PC | 新主方法 | 三模型、四主数据集 |
| DeepConf | 当前强 confidence controller / parent | Qwen3-30B-A3B 必做；DeepSeek offline 必做 |
| DeepConf + PC | 组合主张 | Qwen3-30B-A3B 必做；DeepSeek offline 必做 |
| CPT | 最近、最直接的并行共享方法 | Qwen3-30B-A3B 的 AIME24/25、MATH-500 必做，GPQA 尽量 |

### 4.2 P1 与附录

- **DTS-Stable**：P1 现代结构化并行搜索基线；官方 vLLM 版本若能在 1 天内完成 Qwen3-30B-A3B smoke test，则在 AIME24/25 加入。
- **Sentence-PC**：原始 \(B=20,m=5\) 版本放附录；Shared Sentence-PC 只在 matched-component 长度消融出现。
- **AC、ESC**：附录，维持旧稿和审稿意见连续性；不再占正文主叙事。
- **simple margin heuristic**：仅作便宜 Pareto 诊断，必须明确不是 MARS。
- **MARS、RASC、STAND、Prefix Consistency、LeaP**：Related Work 必须准确讨论；只有官方/忠实实现通过协议审计才给数字，不能用简化代理冒名。

删除所有 early-stopping baseline 会再次触发“回避更简单方法”的审稿问题，因此正文以 DeepConf parent 为代表、附录保留 AC/ESC；重点仍是 parent→parent+PC 的增量，而不是宣称 PC 取代停止方法。

### 4.3 统一预算与统计

- 主方法一般化表：\(B=32,m=4\)，相同 prompt、采样、最大输出、answer parser 和 launched-trace budget；
- DeepConf 表：\(N_{\mathrm{init}}=16,B=32\)；代表子集补 \(B\in\{16,32,64\}\)；
- 不再出现 PC@20 对 SC@40 的节省率；任意百分比只相对同模型、同数据、同预算 parent；
- CPT/DTS 的树或 chunk 结构不能强行按“轨迹数”等价，需同时给 matched launched budget、matched generated tokens/FLOPs 和 matched latency Pareto；
- 本地模型用 seed 0/1/2；DeepSeek API 做三次独立 repetition，因 API 无 seed 控制而不称为 seeds；
- AIME 主统计合并 24+25 后按 question 与 repetition 两层 bootstrap，分年数值仍保留；
- 主表报 mean 与 95% CI；附录补 mean±std、paired difference、有效样本数和 parser failure；
- 所有 warmup、probe、失败 trial、截断与重试 token 都进入总成本。

### 4.4 必报指标

1. accuracy、decode/reasoning tokens、prompt/prefix tokens、total billed/processed tokens；
2. end-to-end latency、throughput、P50/P95、peak KV memory、prefix-cache hit；
3. launched/completed/validated/voting traces 与 parser-failure rate；
4. 相对同 parent 的 \(\Delta\)accuracy、\(\Delta\)tokens、\(\Delta\)latency；
5. matched-accuracy token saving 与 accuracy-token Pareto；
6. gate pass、prefix level survival、wrong-prefix promotion、rollback/recovery；
7. distinct answers、pairwise agreement 与 source-prefix diversity；
8. Beta calibration/ECE 和高置信错误比例。

---

## 5. 主表初稿

表内 **TBD** 表示待实验；N/A 只用于接口上确实不可实现的设置。

### Table 1：跨模型与任务的一般化（\(B=32\)）

| Model | Method | MATH-500 Acc | Tok | AIME24 Acc | Tok | AIME25 Acc | Tok | GPQA-D Acc | Tok |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Inst. | Pass@1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
|  | SC | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
|  | **PC (\(C^\*\))** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen3-30B-A3B | Pass@1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
|  | SC | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
|  | **PC (\(C^\*\))** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DeepSeek-V4-Flash | Pass@1 | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
|  | SC | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
|  | **PC (\(C^\*\))** | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

正文 Tok 默认写平均 generated reasoning/decode tokens；total tokens 与 latency 在紧邻表或 Pareto 图给出，不能混用。

### Table 2：现代基线与组合（Qwen3-30B-A3B，\(B=32\)）

| Method | AIME24+25 Acc | Total Tok | Latency | MATH-500 Acc | Total Tok | GPQA-D Acc | Total Tok |
|---|---:|---:|---:|---:|---:|---:|---:|
| SC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DeepConf (\(*\), dev-selected) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| CPT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DTS-Stable (P1) | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| PC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| **DeepConf + PC** | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

DeepConf 的星号只表示在开发集从官方 low/high 配置中选择并冻结；不能表示测试集 oracle。

### Table 3：parent-paired 增量

| Model / Dataset | Parent | +PC | Parent Acc | +PC Acc | \(\Delta\)Acc | Parent Tok | +PC Tok | Saving | \(\Delta\)Latency |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-30B-A3B / AIME24+25 | SC | SC+PC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen3-30B-A3B / AIME24+25 | DeepConf | DeepConf+PC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen3-30B-A3B / GPQA-D | SC | SC+PC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Qwen3-30B-A3B / GPQA-D | DeepConf | DeepConf+PC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DeepSeek / AIME24+25 | SC | SC+PC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DeepSeek / AIME24+25 | DeepConf-offline | +PC | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

---

## 6. 消融实验表格初稿

消融优先集中在 Qwen3-30B-A3B ×（MATH-500 100 题 + AIME24/25 60 题）；除阈值曲线外均使用冻结的 \(C^\*\)。

### Ablation A：Beta 阈值

| \(C\) | Dev Acc | Test Acc | Total Tok | Gate Pass % | Wrong Promotion % | Rollback % |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00（纯多数） | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.70 | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.80 | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.90 | TBD | TBD | TBD | TBD | TBD | TBD |
| 0.95 | TBD | TBD | TBD | TBD | TBD | TBD |
| 1.00（SC） | TBD | TBD | TBD | TBD | TBD | TBD |

测试列是冻结配置的敏感性复盘，不用于挑 \(C^\*\)；正文主表只显示被选中的一行。

### Ablation B：前缀粒度与长度策略

| Length policy | Level schedule | Acc | Total Tok | Latency | Avg final ratio | Wrong Promotion % |
|---|---|---:|---:|---:|---:|---:|
| Shared Sentence-PC | +1 sentence / level；其余组件同主方法 | TBD | TBD | TBD | TBD | TBD |
| Original Sentence-PC | \(B=20,m=5\)，per-branch random | TBD | TBD | TBD | TBD | TBD |
| Absolute-token PC | +128 tokens / level | TBD | TBD | TBD | TBD | TBD |
| Relative-PC | 20→40→60→80% | TBD | TBD | TBD | TBD | TBD |
| Fixed-20 | 始终 20% | TBD | TBD | TBD | TBD | TBD |
| Fixed-40 | 始终 40% | TBD | TBD | TBD | TBD | TBD |
| **PVC Adaptive** | advance / stay | TBD | TBD | TBD | TBD | TBD |
| PVC Adaptive (agreement only) | 无标签 advance / stay | TBD | TBD | TBD | TBD | TBD |

### Ablation C：唯一前缀来源选择

| Source selector | Acc | Total Tok | Diversity | Wrong-source % | Rollback % | Selector overhead |
|---|---:|---:|---:|---:|---:|---:|
| Random（原文） | TBD | TBD | TBD | TBD | TBD | TBD |
| Shortest trace | TBD | TBD | TBD | TBD | TBD | TBD |
| Token medoid | TBD | TBD | TBD | TBD | TBD | TBD |
| Highest mean logprob | TBD | TBD | TBD | TBD | TBD | TBD |
| **BLC lower-tail confidence** | TBD | TBD | TBD | TBD | TBD | TBD |

### Ablation D：共享方式与系统收益

| Prefix mode | Acc | Decode Tok | Prefill Tok | Latency | Cache hit | Distinct answers |
|---|---:|---:|---:|---:|---:|---:|
| Per-branch random（原算法） | TBD | TBD | TBD | TBD | TBD | TBD |
| Shared random | TBD | TBD | TBD | TBD | TBD | TBD |
| Shared token-medoid | TBD | TBD | TBD | TBD | TBD | TBD |
| **Shared BLC（主方法）** | TBD | TBD | TBD | TBD | TBD | TBD |

### Ablation E：回退策略

| Rollback policy | Acc | Total Tok | Failed trials | Recovery Acc | High-conf. wrong % |
|---|---:|---:|---:|---:|---:|
| No rollback / keep trial | TBD | TBD | TBD | TBD | TBD |
| Gate fail → reset empty（旧实现） | TBD | TBD | TBD | TBD | TBD |
| One-level rollback | TBD | TBD | TBD | TBD | TBD |
| **One-level + blacklist + two-strike reset** | TBD | TBD | TBD | TBD | TBD |
| 主策略 + periodic unprefixed audit window | TBD | TBD | TBD | TBD | TBD |

### Ablation F：SC / DeepConf 与 PC 的 2×2 因子

| Trace controller | Prefix redirection | Acc | Decode Tok | Total Tok | Latency | Completed traces |
|---|---|---:|---:|---:|---:|---:|
| SC | off | TBD | TBD | TBD | TBD | TBD |
| SC | PC | TBD | TBD | TBD | TBD | TBD |
| DeepConf | off | TBD | TBD | TBD | TBD | TBD |
| DeepConf | PC | TBD | TBD | TBD | TBD | TBD |

### Ablation G：Prefix Value Curve 数据表

| Ratio | \(V_{\mathrm{corr}}\) | \(V_{\mathrm{agree}}\) | Correct-source \(V\) | Wrong-source \(V\) | Suffix Tok | Distinct answers |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | TBD | TBD | TBD | TBD | TBD | TBD |
| 20% | TBD | TBD | TBD | TBD | TBD | TBD |
| 40% | TBD | TBD | TBD | TBD | TBD | TBD |
| 60% | TBD | TBD | TBD | TBD | TBD | TBD |
| 80% | TBD | TBD | TBD | TBD | TBD | TBD |

主图画 \(V_{\mathrm{corr}}/V_{\mathrm{agree}}\) 与 token cost 双轴曲线，并分 correct/wrong source；表格放附录。

---

## 7. 实现与可信度清单

### 7.1 P0 代码修复

- [ ] confidence.py 统一为 top-1-vs-all-rest Beta 后验，并为 ties、无有效答案、\(C=0/1\) 写单测；
- [ ] 将 gate、length policy、source selector、rollback、trace controller、aggregator 解耦；
- [ ] 删除未 seeded 的 random.choice；每个问题使用可复现独立 RNG；
- [ ] 实现 reasoning/final-answer span 分离、token-ratio endpoint 与 leakage tests；
- [ ] 保存每个 level 的 source id、prefix token IDs、ratio、BLC score、gate 前后状态和 rollback reason；
- [ ] 修稳定 question id、resume 汇总、tag/config hash；修改 flags 后不得复用旧 JSONL；
- [ ] summary 必须从完整 JSONL 重算，不能只汇总本次续跑部分；
- [ ] 对 DeepConf/CPT 建隔离环境到 ~/data 下，不原地升级现有 PC 环境；
- [ ] 所有大结果、API cache 和临时数据写入 ~/data，仓库只保留小型汇总。

### 7.2 G0 接口 smoke tests

- [ ] Llama-3.1-8B：10 题 × 4 branches，验证 chat template、logprobs、prefix continuation；
- [ ] Qwen3-30B-A3B：TP=4 启动；分别试 \(m_{\mathrm{phys}}=4\) 与 2，记录 32k context 的 KV 余量；
- [ ] DeepSeek-V4-Flash：验证 beta prefix completion 能继续 reasoning prefix；
- [ ] DeepSeek：验证 reasoning_content token logprobs、usage、model/fingerprint 字段；
- [ ] DeepSeek：并发 4 个独立请求，确认 retry 不产生重复计费/重复记录；
- [ ] GPQA gated source、确定性 choice shuffle、四类 scorer 各做 golden tests；
- [ ] CPT 官方代码先复现一个 Qwen 小样例，再允许适配 Qwen3-30B-A3B；若复现失败，不能写“公平比较完成”。

### 7.3 实验分层与砍项顺序

| 层级 | 必须完成 | 若时间不足如何处理 |
|---|---|---|
| P0-A | 三模型的 SC 与 PC；四主数据集 seed/repetition 0 | 不可砍；缺失则主张不完整 |
| P0-B | Qwen3-30B-A3B 的 DeepConf、DeepConf+PC、CPT | 不可用 AC/ESC 代替 |
| P0-C | Beta、length、selector、rollback 四个关键消融；PVC | 每项至少 representative subset |
| P0-D | 主表核心行 seed/repetition 1/2、CI、latency | 先补主方法，不补已明显失败配置 |
| P1 | DTS-Stable、DeepSeek online DeepConf、budget scaling | 接口一天未通即降级并如实写明 |
| P2 | LiveBench/HMMT/AIME26、十个旧任务重跑、复杂理论 | 全部砍掉 |

---

## 8. 从 9 月 6 日开始的排期

### 9/6：冻结 v3

- [x] 冻结模型、数据集、prompt、Beta 主门控、20% ratio grid、shared-prefix 语义；
- [x] 确定 BLC source selector、trial/commit 和 one-level rollback；
- [x] 确定 CPT/DeepConf 为必做现代对照；
- [ ] 将 v3 决策同步到 report 的 review-to-action matrix。

### 9/6–9/8：runner、数据与接口

- [ ] 完成 P0 代码修复和 CPU 单测；
- [ ] 准备 MATH-500、AIME24、AIME25、GPQA-D、Dev-Mix JSONL；
- [ ] 完成三模型 smoke、Qwen TP4/micro-batch 测试、DeepSeek prefix/logprob 测试；
- [ ] 在隔离环境复现 DeepConf 和 CPT 的最小官方例子。

**G0（9/8 晚）**：若 Qwen TP4、DeepSeek reasoning-prefix 或统一 scorer 任一不通，立即缩表，不继续大规模运行。

### 9/8–9/10：PVC pilot 与开发集冻结

- [ ] 跑干预式 PVC pilot，先看 0/20/40/60/80% 的 value-cost 结构；
- [ ] 在 Dev-Mix 选择每模型 \(C^\*\)；
- [ ] 冻结 confidence/length bins 与 adaptive lookup table；
- [ ] 对 random、medoid、BLC 做 50–100 题 selector pilot；
- [ ] 完成 rollback failure injection tests。

**G1（9/10 晚）**：只有 PVC 存在可用安全区，且 BLC/shared-prefix 不比 shared-random 更差，才冻结 PC。

### 9/10–9/14：核心 seed/repetition 0

- [ ] 三模型 × 四主数据集：Pass@1、SC、PC；
- [ ] Qwen3-30B-A3B：DeepConf、DeepConf+PC、CPT；
- [ ] 完成 Table 1/2/3 的第一版和 accuracy-token Pareto；
- [ ] 每天检查 JSONL 完整性、失败率、API 费用和 GPU orphan process。

### 9/14–9/17：重复、消融与统计

- [ ] 主表关键行补 seed/repetition 1/2；
- [ ] 完成 Ablation A–G 的 representative subset；
- [ ] 补 Qwen3-30B-A3B 最终 PVC \(K=8\)；
- [ ] 若有余量，跑 DTS-Stable 与 \(B\in\{16,32,64\}\)；
- [ ] 9/16 冻结数字，9/17 完成独立重算和 CI。

**G2（9/16 晚）**：若 PC 对 SC 无稳定非劣收益、DeepConf 组合无增量或 CPT 明显支配，应缩主张或停止投稿，不能靠测试集阈值、删 baseline 或只报 decode tokens 补救。

### 9/15–9/18：摘要与主体同步写作

- [ ] 摘要只写已冻结、可复算的 parent-relative 数字；
- [ ] 引言增加 PVC、shared-prefix controller、与 CPT 的清晰边界；
- [ ] 方法替换句子级/random-source/reset 语义，加入算法伪代码；
- [ ] 删除无效 Taylor guarantee 和附录 theorem；
- [ ] 评测改为现代模型/数据/parent-paired 表；
- [ ] 9/18 23:59 AOE 前提交真实摘要与完整作者。

### 9/19–9/25：全文缓冲周

- [ ] 9/19–9/21：只补预登记 P1、失败模式、limitations 和 appendix；
- [ ] 9/22：迁移并冻结 ICLR 2027 模板，正文压到 9 页；
- [ ] 9/23：匿名代码说明、config manifests、数据来源和复现命令；
- [ ] 9/24：公式、表格、引用、匿名性、AI-use policy、PDF 独立审计；
- [ ] 9/25 23:59 AOE 前提交全文和 supplement。

---

## 9. 投稿前必须回答的审稿问题

- \(C^\*\) 如何选？答：每模型一次、独立 Dev-Mix、非劣约束下最小 token，测试前冻结。
- “adaptive” 在哪里？答：比例网格只是候选；PVC lookup 根据当前 confidence/length 决定 advance/stay。
- 为什么唯一前缀不是随意的？答：Beta 选答案簇，BLC 选簇内局部最可靠路径；随机选择有直接消融。
- 坏前缀如何取消？答：trial/commit、单层 stack rollback、blacklist、两次失败退化为 parent。
- 为什么还需要 PC，DeepConf 已经省 token？答：DeepConf 停止低价值轨迹，PC 缩短仍要生成的轨迹；用 2×2 表验证而非口头声称。
- 与 CPT 有何不同？答：连续因果前缀 + answer-level posterior + PVC + rollback，对比 CPT 的信息池提取/广播；必须有 matched-budget 数字。
- 并行和 cache 是否真的带来速度？答：同 level 单一 token-identical prefix，报告真实 prefill/decode/latency/KV，而非只报告理论 token。
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

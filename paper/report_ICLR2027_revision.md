# Path-Consistency → ICLR 2027：文献检索 + Novelty 评估 + 修改与投稿计划

> 生成日期：2026-09-02（基于系统日期）
> 分析对象：`_NeurIPS_26__Path_consistency.pdf`（与 arXiv:2409.01281 同源）
> 配套材料：`paper/review/lit/` 下为 arXiv API 原始检索结果（XML）

---

## 0. TL;DR（结论先行）

1. **时间极紧**：ICLR 2027 摘要截止 **2026-09-18 (AOE)**、全文截止 **2026-09-25 (AOE)**，距今天仅约 2–3 周。
2. **直接复投风险很高**：2024–2026 年，"efficient self-consistency / 推理前缀复用 / token-budget" 赛道已高度拥挤，出现多个与 PC 机制或定位高度重叠的工作（最接近的四个：**STAND**（EMNLP'25 Oral）、**RASC**（NAACL'25）、**MARS**（2026）、**Prefix Consistency**（2026-05，连名字都撞）。只做润色不加新贡献，被拒概率大。
3. **必须做的最低限度动作**（无论是否投）：修复论文内部的公式-代码不一致等硬伤（§4），这些是任何审稿人都能挑出的硬伤。
4. **建议路线**：
   - **路线 A（冲 ICLR 2027）**：3 周内完成 §4 修复 + §5 中 Tier-0/部分 Tier-1 新实验（前缀价值曲线、校准、Pareto、matched-budget 多样性对比），并把定位从"启发式"改为"预算化多路径采样的重定向框架"。
   - **路线 B（推荐，若想稳）**：放弃本轮，按 §5 完整方案打磨，投 **ICML 2027**（截止约 2027-01，参考往年）或 NeurIPS 2027（约 2027-05）。ICLR 的 quota 与"rejected 公开留档"机制意味着盲目试错有长期成本。
5. **需要作者先回答的三个问题**（见 §8）：arXiv:2409.01281 是否为本团队论文、可否提供历次审稿意见、计算/时间预算。

---

## 1. 投稿形势核查

| 项目 | 状态 |
|---|---|
| 论文现状 | 匿名版 = NeurIPS 2026 模板，标题/摘要与 arXiv:2409.01281 一致 |
| arXiv 公开记录 | **2409.01281 "Path-Consistency with Prefix Enhancement for Efficient Inference in LLMs"**，2024-08-25 v1；2025-03-02 v2；**2025-11-04 v3**（作者 Jiace Zhu 等，需确认是否本团队） |
| ICLR 2027 关键日期 | Abstract 2026-09-18；Paper 2026-09-25；Review 2026-11-05；Decision 2026-12-16 |
| arXiv 双投政策 | ICLR 2027 CFP 明确允许 arXiv 预印本（"Having papers on arxiv is allowed per the dual submission policy"） |
| 需注意 | ICLR 2027 有**提交配额（quota）**政策、被拒/撤稿稿件**永久公开**、新的 AI 使用与互审政策（Author Guidelines），请逐条阅读 |

---

## 2. 文献检索结果与竞争格局

检索日期 2026-09-02，arXiv API 共 14 组查询；原始数据在 `paper/review/lit/*.xml`。

### 2.1 竞争格局：三个技术家族

**家族一：截断型（early stopping / 自适应分支数）——"算到够为止"**
| arXiv | 标题 | 时间/venue | 与 PC 关系 |
|---|---|---|---|
| 2305.11860 | Let's Sample Step by Step (Adaptive-Consistency, AC) | EMNLP 2023 | PC 的 beta 门控来源；PC 论文已引 |
| 2401.10480 | Escape Sky-high Cost: Early-stopping SC (ESC) | **ICLR 2024** | PC 论文已对比 |
| 2408.17017 | Reasoning Aware Self-Consistency (RASC) | **NAACL 2025** | 动态评估 reasoning+answer 质量做早停；用 reasoning 一致性选分支，**与"路径质量引导"最像** |
| 2408.13457 | Make Every Penny Count: Difficulty-Adaptive SC | 2024-08 | 按难度分配分支数（前置信息） |
| 2601.02970 | Reliability-Aware Adaptive SC (ReASC) | 2026-01 | 证据充分性停止，不用 count-based |
| 2606.12935 | MARS: Margin-Adversarial Risk-controlled Stopping for Parallel TTS | 2026-06 | **检查点探测部分路径的演进多数票并停止**；与 PC 观察同一信号，解不同（停止 vs 复用前缀） |
| 2511.12309 | Optimal Self-Consistency for Efficient Reasoning | **ICML 2026** | SC 样本效率/scaling 的统一理论 |
| 2608.24590 | Is Discrete Difficulty Sufficient? Continuous Difficulty for Efficient SC | 2026-08 | 连续难度分配分支数 |
| 2606.23595 | SPIRAL: Learning to Search and Aggregate | 2026-06 | RL 学 rollout+聚合调度 |

**家族二：复用型（prefix / 中间 token 复用）——"共享计算"，最接近 PC**
| arXiv | 标题 | 时间/venue | 与 PC 关系 |
|---|---|---|---|
| 2506.04708 | Accelerated TTS with Model-Free Speculative Sampling (STAND) | **EMNLP 2025 Oral** | 复用 best-of-N 轨迹的 n-gram 做**无损 draft-verify**。与 PC 差异：token 级、无损（不改采样分布）；PC 是句子级、**有损重定向**（改分布）。**必引必对比** |
| 2509.00195 | FastTTS: Accelerating Test-Time Scaling for Edge LLM Reasoning | 2025-08 | 边缘设备 TTS 加速（检索式草稿） |
| 2605.07654 | Reliable Chain-of-Thought via **Prefix Consistency** | 2026-05 | 截断 CoT 再生成，用"答案能否复现"做可靠性信号**加权投票**。名字高度撞车、机制同用前缀；目标是**可靠性加权**而非省 token。**必引必对比** |
| 2605.08070 | VecCISC: Confidence-Informed SC with Trace Clustering | 2026-05 | 轨迹聚类+候选答案选择 |
| 2607.27783 | Reasoning Consensus: Weighted DAG Aggregation | 2026-07 | 对推理"结构"做加权聚合 |
| 2607.19395 | SPORK: Self-Speculative Forking for Agentic LLM Inference | 2026-07 | 智能体解码中 fork 复用 |
| 服务端侧 | RadixAttention (SGLang) / vLLM prefix cache 生态 | 2024– | 前缀缓存是**服务端**加速；PC 是**方法层**复用，二者正交可叠加 |

**家族三：理论/失败模式/校准族（影响论文定位与论证口径）**
| arXiv | 标题 | 时间 | 与 PC 关系 |
|---|---|---|---|
| 2305.14279 | Two Failures of Self-Consistency in Multi-Step Reasoning | 2023 | 审稿人必然引用的 SC 失败模式分析；PC 的可靠性论证要正面回应 |
| 2502.00511 / 2510.15444 | Bridging Internal Probability and Self-Consistency (+理论) | 2025 | 采样式 TTS 的误差分解（估计误差 vs 模型误差）；**可用于给 PC 提供理论语言** |
| 2403.09849 | Self-Consistency Boosts Calibration for Math Reasoning | 2024 | 支撑"门控置信度≈正确率"校准主张 |
| 2608.11403 | When Self-Consistency Backfires: Majority Vote Hurts... Small LLMs | 2026-08 | 硬问题上多数票有害的新证据；PC 的"Truth in hands of few"叙事要引用 |
| 2607.20464 | Stochastic Sampling is Epistemically Shallow (temperature 多样性不足) | 2026-05 | 直接挑战"SC 靠采样多样性"前提；可支撑 PC"前缀重定向计算"叙事 |
| 2608.24590 等 | 上表 | | difficulty 连续化是 2026 新热点 |

### 2.2 Novelty 评估（直接结论）

- **2024 年视角**：PC = 多数票置信门控 + 前缀复用（句子级、改分布、重定向而非截断），相对 AC/ESC 是清晰增量，成立。
- **2026 年视角（审稿人会怎么看）**：
  - "在多数票上做门控"已被 AC/ESC/MARS/ReASC/难度自适应族吃透；
  - "复用中间推理"已被 STAND（无损、EMNLP'25 Oral）与 Prefix Consistency（2605.07654）占据身位；**审稿人第一联想将是这两个，而不是 AC**；
  - 当前论文与这些工作的**实验对比缺失**（只有 AC/ESC），且没有解释 PC 的"有损重定向"相对"无损 draft-verify"为何值得。
- **结论**：不加新贡献直接投 = 高概率得到 "incremental / not SOTA baseline / name collision" 类意见。**必须做 §5 的重定位 + 一组 STAND/Prefix-Consistency/MARS 对照与差异化指标**。

---

## 3. 论文内部硬伤清单与逐条修改（任何版本都必须做）

### H1（最严重）公式(4) 与实现不一致
正文式(4)（Beta 判据）：
$$\int_0^{0.5} p^{\frac{N}{2} - v_m}(1-p)^{v_m}\,dp$$
与附录/代码实际使用的判据（来自 AC, 2305.11860，`BetaStoppingCriteria`）：
$$\mathrm{conf}=\frac{\int_{0.5}^{1} x^{a}(1-x)^{b}\,dx}{\int_{0}^{1} x^{a}(1-x)^{b}\,dx}\ge C,$$
不一致。问题点：
1. 正文是**未归一化**的原函数积分，"概率值超过 0.5–1.0 阈值"在数学上不成立（原积分可以远小于 1）；
2. 幂次可疑：正文写 $p^{N/2-v_m}(1-p)^{v_m}$，正确贝叶斯后验（均匀先验+两项二项似然）应为 $p^{v_m}(1-p)^{n-v_m}$；
3. $N$ 应改为"当前已采样数 $n$"，不是固定总分支数；
4. 无 $+1$ 正则化（Beta 后验为 $\text{Beta}(v_m+1,n-v_m+1)$）；
5. 代码用 top-2 计数 $(a,b)$（多数 vs 次多数）而理论用多数 vs 其余 —— 当答案多于两类时二者不等价，需统一或显式声明近似。

**修改方案（可直接替换的 LaTeX）**：
```
记 $n$ 为当前已生成的分支数，$v_m$ 为多数答案出现的次数。
在均匀先验与两分类似然下，多数答案的真实概率 $\theta$ 的后验为
$\mathrm{Beta}(v_m+1,\ n-v_m+1)$，因此门控置信度为
\begin{equation}
\mathrm{conf}(n,v_m)=P(\theta\ge\tfrac12\mid n,v_m)
=1-I_{1/2}\big(v_m+1,\ n-v_m+1\big)
=\frac{\int_{1/2}^{1} p^{v_m}(1-p)^{n-v_m}\,dp}
        {\int_{0}^{1} p^{v_m}(1-p)^{n-v_m}\,dp},
\end{equation}
仅当 $\mathrm{conf}\ge C_{\mathrm{threshold}}$ 时才晋升前缀；$C=0$ 对应无条件晋升（纯多数票）。
```
并同步修改代码：用 $(v_m, n-v_m)$ 而非 top-2 计数；或保留 AC 原实现并在脚注说明该近似（推荐前者，改造成本一行）。

### H2 稳定性/多 seed 声明与证据矛盾
- 正文声明 "each reported result is averaged over multiple independent runs with different random seeds"（§4.1）；
- 但附录中的稳定性表被 **整段注释掉**（appendix.tex L68–91）。
修改：恢复稳定性表（我们已在 GSM8K 上跑通 seed 0 全量，补 2 个 seed 成本约 3–4 GPU·小时/配置）；所有主表报 mean±std。

### H3 摘要/正文头号数字不一致
- 摘要："reduces inference latency by up to 40.5%"；
- 主表（Table 2）最大值是 SVAMP **+48.3%**、GSM8K +28.9%；附录 AIME 又不同。
修改：统一口径——"on arithmetic reasoning up to +48%" 或在正文首次出现处给出可溯源的出处；确认 arXiv v3 与投稿版数字一致。

### H4 理论分析与方法的粒度不匹配
§4.2 分析假设"只在 $N/2$ 处做**一次**前缀选择"，但方法实际每窗口（每 5 分支）做一次、共 3 次；且推导用一阶 Taylor 展开 + 对称 $\pm\Delta p$ 假设。
修改（二选一）：
1. 最小：明示这是"单次选择的理想化分析"，把结论弱化为定性（多次门控的行为在 §5 用校准/实验刻画）；
2. 推荐：§5.3 给出**精确**（无 Taylor）的两阶段模型分析：两个答案情形下，PC 最终正确率 = $P_v\cdot P_{\rm inc}+(1-P_v)\cdot P_{\rm dec}$ 的闭式（二项尾部），直接数值验证 $P'_{\rm correct}\ge P_{\rm correct}$ 的区域，而不用近似。

### H5 阈值选择 = 事后挑 C（cherry-picking 风险）
正文多处"不同数据集最优 C 不同"但没有选择原则。
修改：用 §5.1 的**门控校准曲线**给出原则化选择（如：选最小 C 使 $\hat P(\text{correct}\mid \mathrm{conf}\ge C)\ge$ SC 全票正确率），并在主表同时报告"每数据集挑 C"与"统一 C=0.8"两套数字。

### H6 细节/可信度问题清单
- ESC 引用为 `\cite{liescape}`（bib key 不规范），核实为 2401.10480；
- Table 中 `-2,0` 逗号笔误、表头格式不一；
- 前缀抽取代码：以 `"answer is"` 切分 + 按 `. ` 断句——若分支生成被截断/没有该串则**不产出前缀**且可能 append 空串（`sentence=""` 也入列），与正文"总是提取 step"的表述不符；需在代码/文中说明兜底行为（我们复现时已过滤空串，建议作者代码同步并写清）；
- 附录 Case Study 中 `tokens_gen`/`prefix_level` 的模式（末窗口回退 level 0）与附录伪代码"accumulate 前缀列表"行为矛盾——伪代码应改为"门控失败即清空列表回退 level 0"（我们按 case study 行为复现，作者需统一描述）；
- 附录 overhead 表 "Model generation 2.2 s/题" 与正文 RTX 3090 + 20 分支串行生成的常识数量级不符，注明测速条件（是否并行、batch 大小）；
- 公式变量 $N/2$ 两处含义不同（总分支数 vs 检查点），统一记号。

---

## 4. 新贡献方案（若投 ICLR 2027，从中至少实现 Tier-0 + 1–2 项 Tier-1）

### 4.0 重定位：一句话卖点（建议替换全文叙事）
> SC/AC/ESC 在"采样成本"维度做预算管理；STAND 在"token 级无损复用"；**PC 是在"推理步骤级、有损但有引导地重定向测试时计算"**——保持 20 分支的聚合结构，把后续分支的算力投入高置信前缀展开的区域。

由此得到三族对照（论文里直接放一张表）：
| | 代表 | 剪枝 | 复用 | 分布 |
|---|---|---|---|---|
| 截断型 | AC/ESC/MARS/RASC | 是（少分支） | 否 | 不变 |
| 无损复用型 | STAND | 否 | token 级 draft+verify | **不变**（lossless） |
| **重定向型（本文）** | **PC** | 否 | 句子级前缀 | **有偏但受门控保护** |

### 4.1 新概念贡献：前缀价值曲线（Prefix Value Curve）★最优先
- 定义 $V(t\mid q)=P\big(a=a^\star\mid q,\ \text{前 }t\text{ 个推理步}\big)$，以**当前最优路径**（多数票路径）取前 $t$ 步为条件。
- 实证刻画 $V(t)$：随 $t$ 增长的饱和曲线 + 每级门控点所在位置 → **给"为什么每次只加一步、为什么 0.8 是拐点"一个数据驱动的答案**（对比 2608.24590 的连续难度论、2403.09849 校准论）。
- 由此把"固定每级 +1 句"扩展为**自适应步数**：门控在 $V$ 边际增益处收敛时再晋升（可后置为扩展实验）。

### 4.2 方法扩展（按性价比排序）
1. **自适应前缀长度（步级切分）**：句子 → 以换行/编号为边界的原子步（GSM8K 每行一步），晋升步长不固定；
2. **Top-k 前缀池 + 熵控采样**：不只在多数路径上取前缀，保留 top-k 候选前缀、按与置信度成比例采样 → 直接回应"前缀压缩多样性"的质疑（配合 §4.4 指标验证）；
3. **门控校准作为阈值选择原则**（H5 的落地）。

### 4.3 理论（选做，时间允许时）
- 单步门控的**精确**分析（无 Taylor），给区域断言；
- **多样性守恒引理**：从前缀池（大小为 $k$，池内每前缀来自不同多数路径）采样保持的"有效推理空间"下界（用 distinct reasoning steps 定义）；
- 若与 2502.00511 框架衔接：把门控解释为对 $P(R_{\rm prefix}\mid q)$ 的**估计误差与模型误差**折中——PC 减小估计误差（集中采样）的代价是模型误差（条件分布更尖），校准曲线即其可视化。

### 4.4 实验矩阵（回答审稿人必问的三个问题）
**Q1：与最接近的新方法比到底如何？**（基线矩阵，全部同预算）
SC(20/40) vs PC(C=0/0.7/0.8/0.9) vs AC vs ESC vs **RASC** vs **MARS** vs **STAND** vs **Prefix-Consistency(2605.07654)**，GSM8K + SVAMP + BBH 子集；指标：准确率、token/题、端到端时延、**准确率-token Pareto 前沿（扫分支数×阈值）**。

**Q2：前缀重定向会不会压掉多样性/何时 PC < SC？**
- matched-token/matched-latency 下比：distinct answers、路径自 BLEU、唯一首步数、正确分支多样性；
- 失败模式 taxonomy：按 2305.14279 / 2608.11403 的"硬问题/少数派真理"切分，报告 PC 在哪些子集劣于 SC，并给出门控失效的证据链（错误前缀被晋升的样例+统计）；
- 这个分析本身是卖点："PC 什么时候应该被禁用"。

**Q3：在真实推理引擎上快多少？**
- 用 vLLM（我们已就绪）：同一窗口共享前缀 = 单请求 n=5（已实现），报告真实吞吐/时延；与 per-branch 采样（前缀缓存命中率低）对比——**"方法层重定向 × 服务端前缀缓存"正交性演示**；
- 顺带与 appendix 的 DeepSeek-V3/小模型结果合并成"模型规模与家族泛化"小节。

### 4.5 需换/加的评测资产（ICLR 2026+ 标准）
- 模型：Llama-3.1-8B(-Instruct)、Qwen2.5-7B-Instruct、保留 Llama-3-8B 与 Llama-3.2-1B（连续性）；Mistral-7B 已有；
- 数据：保留原 10 个（连续性），新增 **MATH-500**（已有缓存）、AIME2025 从附录提正；BBH 三任务现有 prompt 复用；
- 基础设施：~/data 下模型与数据集已就绪，vLLM 并行管线可直接复用于全部新增运行。

---

## 5. 执行清单与排期（目标 ICLR 2027，2026-09-25 截止）

我们已有的资产（本次复现已就绪）：vLLM Llama-3-8B 推理管线、beta 门控与前缀抽取（与 AC 判据同源代码）、GSM8K JSONL、5×RTX 4090、约 1.5h/全量配置/seed。

| 里程碑 | 内容 | 预计 |
|---|---|---|
| D+1~2（9/2-3） | 完成 §4 修复的第一版正文（H1 公式、H2 恢复稳定性表、H4 分析重写、H5 口径）；开始 2 个额外 seed 的 GSM8K×5 配置（后台） | 与写作并行 |
| D+3~5 | Tier-0：AC/ESC/RASC/MARS/STAND 基线（前两者半天内，其余 2-3 天）；跑 Llama-3.1-8B/Qwen2.5-7B GSM8K 矩阵 | — |
| D+6~9 | Tier-1a：前缀价值曲线 + 校准曲线 + Pareto 前沿（需要 sweep 分支数 {5,10,20,40}×C） | 3-4 天 |
| D+10~13 | Tier-1b：matched-budget 多样性指标 + 失败模式 taxonomy + vLLM 时延对比 | 3-4 天 |
| D+14~16 | 全文重写（定位=三族框架）、图、tables、related work（本报告 §2 表直接可用） | — |
| D+17 | **Abstract 截止 9/18** 前注册 OpenReview + 提交摘要 | — |
| D+19~22 | 内部评审、补小实验 | — |
| **D+23** | **9/25 Paper 截止** | — |

若 Tier-1 无法全部完成：优先级 = 前缀价值曲线 > 校准阈值 > 三族对比表（至少 AC/ESC/RASC/MARS，STAND 尽量）> 多样性 > vLLM 时延。

## 6. 风险与备选

1. **撞名风险**：2605.07654 "Prefix Consistency" 与 2408.17017 "Reasoning-Aware SC"——若 2409.01281 非本团队论文，撞名+同 idea 属致命冲突，必须先核实（§8 Q1）；即使是自己的，也建议在正文明确与 2605.07654 的差异（可靠性加权 vs 计算重定向）。
2. **ICLR 2027 不中则公开留档**：每次 rejected submission 都会公开带作者名，且占 quota。若 Tier-0/1 在 9/20 前仍不齐，果断 withdraw 摘要，转 **ICML 2027**（截止约 2027-01 下旬）或 **NeurIPS 2027**（约 2027-05），用 4–8 个月把 §4.2/4.3/4.4 做满。
3. **STAN 式"无损"叙事压制**：PC 必须主动论证"分布有偏但门控保护"的收益边界（校准曲线 + 失败 taxonomy），不能回避。

## 7. 需要作者提供/确认的输入

1. arXiv:2409.01281（Jiace Zhu 等，v3 2025-11）是否为贵团队论文？（是→建议加 citation 自述+更新 arXiv；否→本轮写作与 novelty 论证须完全不同，甚至换名换题）
2. 历次被拒的**审稿意见原文**（尤其是最近一轮 NeurIPS 2026 的），用于逐条对照 H1–H6 之外的真实拒因；
3. 计算/人工时间预算上限（决定路线 A 还是 B）；
4. 是否接受对方法命名/表述做较大改动（避免与 2605.07654 混淆）。

---

### 附录：原始检索文件
- `paper/review/lit/`：q_majvote/q_pathcons/q_pathreuse/q_ttc/key/key2/ac/venue/s*.xml（arXiv API 原始结果，含 2024–2026 全部相关查询）

---

## 8. 真实审稿意见对照矩阵（2026-09-02 补充）

> 输入：ICML 2026（Submission 10772，4 位审稿人，2026-02~03）+ NeurIPS 2026（Submission 20352，2 位审稿人，2026-06~08）。
> ICLR 历次意见待作者提供原文后并入（OpenReview 在当前网络环境不可达，见 §9）。

### 8.1 逐条映射（每条 = 审稿人原话要点 → 归属章节/动作 → 对应 §4 修复/新贡献）

**A. 理论有效性（被两位审稿人同时质疑，最高优先）**
| # | 来源 | 意见要点 | 动作 | 归属 |
|---|---|---|---|---|
| A1 | ICML-kgyj；NeurIPS-gzrj | §4.2 一阶 Taylor 展开有效性未论证；`(1+Δp/p₀)` 被抬高到 N/2 次幂，线性化实际要求 `N·Δp/(2p₀(1-p₀))≪1`，恰是"前缀几乎不改分布"的平庸区 | **删掉 Taylor 论证，换成精确二项式数值/闭式验证**（两个答案情形无近似）；或降级为"理想化单步门控的定性说明"并补 128-rollout 经验验证（见 A4） | H4 / §4.3 |
| A2 | ICML-ZYMf | §4.2 假设"只生成一个错误答案"过于理想；要求对 "Truth in Hands of a Few" 做**大规模采样经验分析**：≥128 rollouts 求出样本真值→按难度分组→报告 PC 各组的正确率与前缀存活率 | 设计"难度分组 + 前缀存活率"分析实验（GSM8K 抽样 200 题 × 128 rollouts，p₀ 用 oracle 估计，PC 在同题上跑，画 正确率/存活率 vs 难度 曲线） | 新实验 Tier-1（§4.4 Q2） |
| A3 | NeurIPS-N6L9 | 错误前缀被高置信锁定的最坏情形缺乏鲁棒性分析；是否有 prefix correction/backtracking 机制的证据 | 展示"前缀修正"实例统计（门控失败回退频率、回退后正确率恢复）；worst-case：构造/标注"高置信错误前缀"样本分析 | 新实验 Tier-1 |

**B. 评测公平性与规模（两位审稿人点名）**
| # | 来源 | 意见要点 | 动作 | 归属 |
|---|---|---|---|---|
| B1 | ICML-ZYMf | Table 5 不公平：PC 锁定 20 分支 vs SC 放宽到 40，人为放大 token 缩减率 | **等预算对比**：SC@20 与 SC@40 分开列；PC@20 vs SC@20 为主表口径；PC+AC/PC+ESC 的正交性结果保留并前移为主卖点之一 | 表格修改 |
| B2 | ICML-kgyj/vPiv/aGNg | 仅 Llama-3-8B 基座；要求 reasoning 模型（DeepSeek-R1-Distill-Qwen、Qwen3）与现代基准 | 新增模型矩阵：Llama-3.1-8B、Qwen2.5-7B-Instruct、DeepSeek-R1-Distill-Qwen-7B（后台已具备跑 PC 的 vLLM 管线）；数据集至少 GSM8K+SVAMP+MATH-500 | Tier-0 |
| B3 | ICML-vPiv | 与"简单启发式"的收益边界不清 | 加统一基线矩阵 SC/AC/ESC/RASC-lite/MARS + PC（同预算 Pareto） | §4.4 Q1 |
| B4 | ICML-ZYMf | PC 随分支预算 N 的扩展行为未讨论 | 补 N∈{5,10,20,40} × C 的 scaling 表（正好构成 Pareto 前沿数据） | §4.4 Q1 |
| B5 | NeurIPS-gzrj | 主表缺方差/显著性 | 恢复多 seed 表（附录稳定性表取消注释）；报告 mean±std | H2 |

**C. 方法与动机（kgyj 的"ad hoc"、多样性担忧反复出现）**
| # | 来源 | 意见要点 | 动作 | 归属 |
|---|---|---|---|---|
| C1 | ICML-kgyj | 多级设计缺乏动机；前缀是否遮蔽前级结论句、后续是否"粘"在旧前缀上 | 用**前缀价值曲线**（§4.1）回答"为什么分级/每次一步"；新增"遮蔽 vs 不遮蔽"消融；分析前缀内答案句泄漏 | §4.1 + 消融 |
| C2 | ICML-vPiv/aGNg；NeurIPS-gzrj | 前缀采样可能压缩多样性/坍缩到单前缀 | 多样性指标：不同答案数、自 BLEU、唯一首步数，matched-token 对比 SC/AC/ESC；Top-k 前缀池+熵控采样（若时间允许） | §4.2.2 + §4.4 Q2 |
| C3 | NeurIPS-N6L9 | 置信评估的数学判据定义不清（token logprob？verifier？）+ 开销占比 | 明确判据=答案级 Beta 后验（含正确公式，修 H1）；补 附录 overhead 表已有数据（0.004% 级）前移到正文或加批注 | H1 + 表述 |
| C4 | NeurIPS-N6L9 | KV cache/长前缀峰值内存 | Limitations 里补一段量化说明（用 GSM8K 平均前缀 token 占比估峰值 KV 增量） | 文本 |

**D. Related work / novelty（被质疑最多的一条线）**
| # | 来源 | 意见要点 | 动作 | 归属 |
|---|---|---|---|---|
| D1 | ICML-aGNg | 与 PHP (Progressive-Hint Prompting, arXiv:2304.09797) 相似 | Related Work 明确辨析：PHP 是**逐步加提示**（每次重发完整问题+上一答案），无置信门控、无并行分支、目标是最佳单答而非 SC 效率；PC 是多路径采样下的前缀重定向。**表格式对比** | Related Work |
| D2 | ICML-kgyj | 漏引 Deep Think with Confidence (arXiv:2508.15260) | 补引 + 辨析（同用置信度，DeepConf 用于**停止/选答案**，PC 用于**前缀晋升**） | Related Work |
| D3 | ICML-vPiv/aGNg | baselines/related work 过时；缺 ToT/MCTS | Related Work 重写为"截断型/无损复用型/重定向型"三族（§4.0）；ToT/MCTS 定位为"树搜索家族"，声明 PC 属多路径采样式、与树搜索正交，可选做 MCTS 式预算消融 | Related Work |
| D4 | 全局 | 与 STAND (EMNLP'25 Oral)、Prefix Consistency (2605.07654)、MARS、RASC 的差异必须显式化 | §4.0 三族表 + Related Work 逐一辨析 + 至少 1 个同预算经验对比（MARS 与 RASC-lite 可实现，STAND 以引述+定性为主，若时间允许做简化版） | §2/§4.0 |

### 8.2 审稿人共通信号（与 §4 硬伤清单完全互证）
1. "ad hoc / 动机不足" ↔ H4（理论）与 §4.1（价值曲线）——**分级、每级+1 句、阈值 0.8 都要有数据驱动解释**；
2. "多样性压缩" 是第二高频 → 没有它，novelty 与实用性论证都不完整；
3. "等预算/同条件下对比" 是评测第一原则 → 所有表重做口径；
4. 审稿人普遍认可：**PC 与 AC/ESC 正交（组合实验是正资产）**——建议 ICLR 版把 "PC×早停族正交、PC×前缀缓存正交" 作为双正交卖点贯穿全文。

### 8.3 ICLR 版新增/重排内容（对照投稿节奏）
- 摘要：换头号数字口径（等预算 token 缩减率），出现 "budgeted multi-path sampling" 三族定位；
- §2 新增"失败模式前置分析"（SC 的错误/冗余 token 占比已有，新增**按难度分组的正确分支占比**）；
- §3 方法：图改为"前缀价值曲线 V(t)" 引导读者理解分级动机；补 masking 说明；
- §4.2 重写（A1/A2）；
- §5 评测：B1/B2/B3/B4/B5 全口径重做；
- Related Work 重写（D1–D4）；
- Limitations：补 C3/C4 量化说明。


## 9. 执行日志（Route A，持续更新）

### 2026-09-02
- [x] 读取 ICML 2026（4 审）与 NeurIPS 2026（2 审）意见 → §8 矩阵
- [x] 论文源码复制至 `paper/src/`，完成第一批 LaTeX 修复：
  - `03_methodology.tex`：公式(4) 重写为归一化 Beta 后验 `conf(n,v_m)=1-I_{1/2}(v_m+1,n-v_m+1)`，并写明 C=0/1 语义（H1）
  - `04_evaluation.tex`：阈值描述 0.5–1.0 → {0,0.7,0.8,0.9}，C=1.0 即 SC（H1/H5 口径）
  - `03_methodology.tex`：§4.2 增加"单次门控理想化"说明（H4 部分）
- [x] 复现代码扩展：`PCInference` 支持 early_stop ∈ {ac, esc, margin}（AC=窗口级 beta 门控停止 C=0.9；ESC=窗口全同停止；margin=多数票差≥2 停止）；`--store-text` 保存分支文本与所用前缀
- [x] 启动运行（5 GPU 并行）：
  - GPU0 PC(C=0.8) 全量带文本；GPU1 SC 全量带文本（用于 V(t)/masking/前缀正确率离线分析）
  - GPU2 AC@0.9、GPU3 ESC、GPU4 margin 全量（新基线，回复 vPiv/aGNg/B1-B4）
- [x] 新模型下载启动（后台，存 ~/data，无配额问题）：Llama-3.1-8B、Qwen2.5-7B-Instruct、DeepSeek-R1-Distill-Qwen-7B
- [ ] 离线分析（带文本数据就绪后）：V(t) 曲线、masking 统计、前缀正确率/存活率（A2/C1）
- [ ] 待办：ICLR 审稿意见原文（作者提供）→ 并入 §8；seed1/2 方差表（H2）；SC@40 等预算对比（B1）；MARS 精确停止规则实现（周 2）


## 10. 主表与图表设计决策（作者 2026-09-02，Route A 执行中）

作者决策：**不在主表汇报不同阈值设定**（无法实验前预设最优 C 的表达问题），改为：
1. 主表汇报"每模型×数据集下我们的方法达到的最好结果"；
2. 阈值探索降级为小实验/附录；
3. 用 accuracy-token 曲线展示能力边界。

### 10.1 主表 v2 结构（建议）
- 表按数据集分块（GSM8K / SVAMP / BBH 子集 / MATH-500），块内列 = Acc(%) 与 Tokens(#)；
- 行 = SC(20) / AC / ESC / PC（marginal 方法同样汇报，因为它们也要选停止阈值）；
- PC 行数值口径：`PC(C^*)` —— **dev 集规则选出的阈值**，括号附注统一 C=0.8 的成绩（透明性）。
- 规则细节（防 cherry-pick，答复 ICML-ZYMf/vPiv 口径质疑）：
  - 每数据集取 dev 集（如 GSM8K 用 train 抽 500 题；其余数据集留 5–10% 或 few-shot dev）；
  - 在 C∈{0, 0.5, 0.7, 0.8, 0.9, 0.95} 上跑 PC，选 **"dev 上准确率不低于 SC(20) dev 准确率的最省 token 的 C"**（accuracy-preserving minimal cost，APMC）；
  - 测试集只报 PC(C^*)；附录补 oracle-best（test 上挑 C）以证明规则接近 oracle。
- 注：同一 APMC 规则也用于 AC/ESC 的停止阈值/窗口参数，保证所有方法在同一"规则化选择"口径下对比。

### 10.2 accuracy-token 曲线（主图候选）
- 每数据集一张子图：x = tokens/problem（对数值刻度更直观），y = Acc(%)；
- 点集：SC@N∈{5,10,20,40} 连线 + PC@(N=20)×C∈{0,0.5,...,0.95} 连线 + AC/ESC/MARS 单点 + SC(20) 参考点；
- 解释文本："同一预算下 PC 前沿在 SC 前沿左上/同高左移"，同时回应 B4（N 扩展性）与 ZYMf 的等预算质疑；
- 首批原型只用 GSM8K seed0 已有配置（SC@20、PC@C、AC、ESC、margin），全量前沿（N×C sweep）列入周 2 排期。

### 10.3 影响
- 表格改动影响 §8 矩阵：B1 维持、B4 由前沿图承接；
- 后续补跑项新增：各数据集 dev 集 6 阈值 × 各模型 的 PC 运行（在 3 周预算内按模型优先序执行）；
- 若时间不足以对每个新数据集做 6-C sweep，可用 GSM8K 学到的规则迁移（报两套：per-dataset 规则值 + 全局固定 0.8），并在正文如实说明。


### 2026-09-03
- 两周执行计划（含 W2 提交缓冲）输出至 `paper/plan_2weeks_ICLR2027.md`；主表 v2 决策（§10）已并入其中 D1-D7 排期。
### 2026-09-03（续）
- 模型方案变更（作者定）：停 Llama-3.1-8B/Qwen2.5-7B-Instruct/R1-Distill-7B 下载并清除缓存；改为下载 **Llama-3.1-8B-Instruct**（进行中）与 **Qwen3-32B bf16**（进行中，3 卡 TP 运行）；reasoning 侧用 **DeepSeek API `deepseek-v4-flash`**（key 存 `/data/26xyw/env/deepseek_api.env`，勿写入仓库/文档）。计划文件 `plan_2weeks_ICLR2027.md` 已同步。

## 11. 2026-09-04 方法重定位与计划 v2 决策

> 触发：作者提出（1）自适应前缀长度；（2）将 PC 作为 SC/DeepConf 的增量模块而非主打截断基线；（3）前缀价值曲线。完整执行矩阵见 `paper/plan_2weeks_ICLR2027.md` v2。本节覆盖与 §5、§9、§10 中旧排期冲突的部分。

### 11.1 三项建议的裁决

1. **自适应前缀长度：有条件采用并升为方法主扩展。** 候选点改为 tokenizer-aligned normalized positions，由独立 dev 上的干预式 Prefix Value Curve 校准；必须与 fixed-sentence、fixed-token 对照。只把句子换成固定 token 数不能称为 adaptive。
2. **增量模块定位：采用；“完全不比截断方法”不采用。** 正文主表改为 SC/DeepConf 各自 `parent vs parent+PC` 的因子设计；AC/ESC/margin 等保留为紧凑诊断对照。原因是 DeepConf-online 自身包含轨迹内和轨迹数 early stopping，且 ICML 审稿明确批评基线过时、同时认可 PC+AC/ESC 的正交性。
3. **Prefix Value Curve：升为 P0，但现有结果作废。** `analysis/offline_stats.py` 未做条件重采样，使用最终多数路径且 exact-prefix 组几乎为 singleton；必须用独立 discovery/evaluation rollouts 的 prefix intervention 重新估计 `V_corr(t)`、`V_agree(t)`、cost 和 diversity。

### 11.2 新发现的 P0 可信度问题

- 正文 Beta gate 是 top-1 vs all-rest，代码是 top-1 vs top-2；现有 C=0.8 轨迹中约 747/3957 个门控决定会改变，因此旧 PC/AC 数据不得进入新主表。
- 代码每个 window 的 5 条分支共享一次抽出的 prefix，而论文伪代码逐 branch 抽取；必须选择语义并重跑，主设置采用 per-branch sampling。
- gate fail 的 keep/reset、累计/当前窗口门控、answer masking 在论文和代码间不一致；v2 协议固定 cumulative completed traces + reset-to-empty，并做消融。
- 当前 early-stop 与 prefix promotion 互斥，尚未实现任何 `controller+PC` 组合。
- AC/ESC/margin JSONL 混入 3 条 smoke-test ID 错配记录，summary 又只统计续跑部分；runner 的稳定 ID/resume/seed/config manifest 先修再跑。
- 主文 Taylor “不损害准确率”推导仍无效；附录新 theorem 的 failure-event 包含方向错误，多级推论也依赖无效假设。两周内删除保证性结论，改为风险分解和经验边界。

### 11.3 当前数据对主叙事的约束

GSM8K/Llama-3-8B seed0 诊断结果：SC 68.16%/2033 decode tokens；PC(C=0.8) 66.79%/1764；PC(C=0.9) 68.31%/1852；AC 约 67.93%/1385；simple margin 约 67.93%/989。standalone PC 尚未展示优于简单 stopper 的 Pareto，故不能删除强基线后继续声称 standalone superiority。新的 go/no-go 条件是：`DeepConf+PC-Adaptive` 必须相对同一 DeepConf parent 在至少两个核心设置上稳定产生非劣准确率和额外成本收益。

### 11.4 模型与算力决策更新

- Llama-3.1-8B-Instruct 已完整缓存；Qwen3-32B 当前仅有 refs、无权重，且 64 attention heads 不支持原计划的 3-card tensor parallel。Qwen3-32B 降为 P2。
- DeepConf 需要内部 top-k logprobs/在线 EOS，DeepSeek API 不能作为忠实主实现；P0 reasoning model 改为本地 Qwen3-8B，并为官方 DeepConf 建隔离环境，禁止原地升级现有 vLLM 0.8.5。
- 数据矩阵收缩为 Llama-3.1-8B-Instruct/GSM8K 与 Qwen3-8B/MATH-500+AIME2025；十个旧任务全量重跑、ToT/MCTS、Top-k prefix pool 暂缓。

### 11.5 截止日与止损

官方 ICLR 2027 摘要截止为 2026-09-18 23:59 AOE、全文截止为 2026-09-25 23:59 AOE。9/8 做组合/PVC 第一决策门，9/14 冻结数据并做全文 go/no-go；若组合增量、干预式 PVC 或协议一致性任一核心证据不成立，则在摘要截止前停止本轮，而不是扩大旧任务表掩盖。

## 12. 2026-09-07 MATH500 初始实验执行记录

> 本节与 `paper/plan_2weeks_ICLR2027.md` v3 一起覆盖 §11.4 的旧模型状态。方法名称固定为 **PC（Path-Consistency）**，不使用 PC-AS。

### 12.1 已完成实现与校验

- 新增 `run_math.py` 与 `path_consistency/math_inference.py`：MATH500 JSONL 在 CUDA 初始化前读取；支持 SC/PC 共用 vLLM engine、唯一共享 token-prefix、20% ratio、BLC source、trial/commit、单层 rollback、增量 JSONL 和配置指纹。
- `confidence.py` 已改成 top-1-vs-all-rest Beta 后验；CPU 单测覆盖空答案、Beta、嵌套 boxed、符号评分、BLC 和 DeepConf，共 7/7 通过。
- 主评分采用 `math-verify==0.9.0`，外置于 `/data/26xyw/python_packages/math_verify_0_9`，不改变 PC 环境的 torch/sympy 依赖。
- DeepConf 初始结果只称 **DeepConf-offline**：同一 trace pool 上计算 top-20 token confidence、LGC(2048)、top-90%/top-10% filtering + weighted vote。online early-stop 尚未完成官方隔离环境复现。
- CPT 官方仓库只提供 Qwen/GPT-OSS 执行入口；不把未经验证的 Llama 适配冒充 CPT。CPT 留到 Qwen vLLM 阶段。

### 12.2 Qwen3-32B 接口诊断（已归档；本地 Qwen 改为 A3B）

- checkpoint `/data/26xyw/Qwen3-32B` 完整；vLLM 0.8.5 能识别 Qwen3，TP=4 权重加载成功，每卡约 15.39 GiB。
- 2026-09-08 短探测：`NCCL invalid usage` 的根因是 torch cu118 自带的 NCCL（CUDA 11.0 build），不是跨 NUMA。装上 A3B 用的 CUDA 12.2 `libnccl.so.2` 后，V1 CUDA-graph 在 Qwen3-32B TP=4 / GPU 1–4 上可以 capture。A3B 的 `max_num_batched_tokens=512` 对 dense 32B 会 OOM；缩小到 capture sizes `{32,16,8,4,2,1}` 后 capture 成功（3 s，0.55 GiB/卡）。详见 §12.9。
- thinking-on、MATH500 随机 Geometry 题、B=8、max_tokens=4096 的 SC/PC 各用约 596/597 秒；16/16 路径均在输出 boxed 前截断。该 **V0 eager** 配置不可批量扩展。
- 作者 9/7 决定：当时快速迭代改用 Llama-3.1-8B-Instruct。9/8 起本地 Qwen **主模型改为 Qwen3-30B-A3B**（见 §12.10），32B 不再进入主表。

### 12.3 Llama-3.1-8B-Instruct smoke

| 设置 | SC | PC | 结论 |
|---|---:|---:|---|
| 随机 Level-5 Geometry，B=8，max_tokens=2048 | 0/1；14,045 decode tokens；87.12 s | 0/1；14,045；86.98 s | 仅 3/8 有效且 gate 未开；不能据此判断 PC 效果 |
| 指定 Level-1 Algebra，B=8，max_tokens=1024 | 1/1；1,278；8.54 s | 1/1；1,198；7.85 s | 首窗 4/4 一致；唯一 35-token 前缀提交；decode −6.3%，时间 −8.1% |

第二行确认所有 level 内分支使用同一个前缀、Beta/BLC/commit 链路可执行，但只是功能测试。首个代表性初始实验曾按固定随机 10 题、B=16、m=4、max_tokens=4096、C=0.8、seed=0 启动；其配对缺陷与替代 run 见 §12.4，旧日志为 `/data/26xyw/path-consistency-results/math500/logs/pilot10_llama31_b16_mt4096_c08_seed0.log`。

### 12.4 配对协议修正与 v2 pilot

- 首个 pilot 完成 3 对后主动终止。第三题从始至终未使用 prefix，但 SC 与 PC 的 root sample 仍出现差异；定位为同一 vLLM V0 engine 中先完整调用 SC、再调用 PC 时，多样本 RNG 会受请求序号影响。旧 run 因此只保留作功能/吞吐 smoke，**不得把逐题准确率或 token 差异归因于 PC**。
- v2 采用共同随机数：同题 SC 的无前缀窗口作为 PC 的只读 root cache；PC 仅从第一次 prefix proposal 后重新生成，rollback 到 root 时也复用对应 SC 窗口。输出同时记录 `generation_seconds`（将复用窗口原耗时计入的算法等价成本）、`actual_generation_seconds`（本次实际新生成）与 `wall_seconds`，不得用缓存缩短后的 wall time 冒充方法收益。
- 一题 GPU smoke 已验证：PC 首窗 4 条路径与 SC 文本逐条完全相同；随后 20% prefix commit。SC/PC decode tokens 为 1,090/926，算法等价生成时间为 6.69/5.69 秒；PC 本次实际新生成 2.09 秒，其余 3.60 秒是共同 root cache。
- DeepConf-offline 同步到论文/官方代码口径：每 token 精确取 top-20；同时计算 Mean、Bottom-10% Group、Tail(2048) 和 LGC(2048)，90%/10% filtering 使用 percentile threshold 后 confidence-weighted vote。`DeepConf-low/high` 名称保留给后续 online 方法，离线结果写作 `Offline-{metric}@{retention}`。
- v2 固定 10 题 pilot 已后台启动（PID 启动值 1594026）；日志为 `/data/26xyw/path-consistency-results/math500/logs/pilot10_llama31_b16_mt4096_c08_seed0_v2.log`。配置指纹现在包含协议版本和关键实现文件哈希，代码变化不会误续跑旧 JSONL。
- 数据隔离：该固定 10 题已经用于工程调试，标记为 engineering pilot split；后续参数冻结只使用 Dev-Mix，MATH-500 主结果使用剩余 490 题。若附录复算完整 500 题，必须披露其中 10 题曾被查看，不能把它称为完全未见测试集。
- v2 10 题已于 2026-09-07 12:10 CST 完成（fingerprint `969c6fca3712`，墙钟 2141 s）。Codex session 因额度中断时尚未写报告；完整配对表见 `/data/26xyw/path-consistency-results/math500/reports/pilot10_llama31_v2.md`。

### 12.5 v2 10 题结果与 50 题初始实验

协议验证通过：共同随机根生效；无前缀窗口 PC 与 SC 逐条一致；trial/commit/rollback 均出现过。准确率 SC=PC=4/10，无 sc_only / pc_only 翻转。

| 口径 | 数字 | 解读 |
|---|---|---|
| 总体 decode 节省 | 0.96% | 被 6 道永不门控的长截断题稀释 |
| 可门控 4 题 decode 节省 | 10.47% | 且 4/4 正确，答案与 SC 相同 |
| 截断 | 48/160 轨迹 | `max_tokens=4096` 下 Llama 在 L4/L5 上循环 |
| DeepConf-offline | 与 SC 同为 4/10 | 离线过滤无增量；不是 online DeepConf |
| 等价生成时间 | SC 197.7 s/题，PC 196.6 s/题 | 必须用 `generation_seconds`，禁止把 cache 后的 153 s 总实际生成写成加速 |

**Go/no-go（相对批量 490 题）**：协议可扩展，但 Llama-3.1-8B × MATH 的主要成本在“无法完成 boxed 的难例”，PC 在这些题上正确退化为 SC。10 题不足以判断非劣准确率或稳定节省。下一步不是改算法，而是在**未见的 50 题 held-out**上用同一协议估计门控率与分层收益。

**已启动**（2026-09-07 13:21 CST，PID 2541790，GPU 2，fingerprint `53f4f866f610`）：`configs/math500_llama31_initial50_heldout_seed2027.json`。同一 shuffle（seed=2027）跳过工程 10 题后取下 50 题；B=16、m=4、C=0.8、max_tokens=4096、paired-root v2。该 50 题仍不作超参选择。日志：`/data/26xyw/path-consistency-results/math500/logs/initial50_llama31_b16_mt4096_c08_seed0_v2.log`。

**已完成**（2026-09-07 15:27 CST，墙钟 7483 s）。完整分析：`/data/26xyw/path-consistency-results/math500/reports/initial50_llama31_v2.md`。

| 口径 | 数字 |
|---|---|
| SC = PC | 27/50 = 54%；翻转 0 |
| Pass@1（第 1 条轨迹） | 21/50 = 42% |
| Decode 节省 | 4.17%（14751 → 14136 tok/题）；等价时间 −3.10% |
| 可门控 24 题 | 准确率 87.5%，decode −18.7% |
| 从未提案 22 题 | 与 SC 轨迹逐条相同 |
| L5 | 14 题里 1 题门控；decode −0.4% |
| 截断 | 90/800 轨迹 |

**Go/no-go（Llama 490）**：协议可冻结，但 **不要用 4096 上限直接打满 490 题**。结构已清楚：L5 几乎不门控，总体节省会被难题稀释到几个百分点，准确率持平。若 Llama 仍要进主表，先在难例上试更高 max_tokens；否则降为附录，主证据留给 DeepSeek/Qwen。


### 12.6 DeepSeek-V4-Flash API：smoke 通过并启动 MATH-500 490

密钥只读 `/data/26xyw/env/deepseek_api.env`，未写入仓库。与 Llama GPU 实验并行，不占卡。

**G0 smoke**（`reports/deepseek_smoke.json`）全部通过：

| 检查 | 结果 |
|---|---|
| thinking + usage / model / fingerprint | 正常；`deepseek-v4-flash`，有 `system_fingerprint` |
| reasoning_content token logprobs | 有；top-20 可用，因此主结果用 **BLC**，不是 PC-medoid |
| 并发 4 独立请求 | 4/4 成功，响应 id 互异，retry=0 |
| beta prefix completion | 必须走 `https://api.deepseek.com/beta`；有效写法是末条 `assistant` 带 `prefix=true`、`content=""`、`reasoning_content=<prefix>`。只传 reasoning 会 400；把前缀塞进 `content` 会关掉 thinking |

API 返回的是 **reasoning 续写后缀**，不是整段重放；runner 只在返回值以 prefix 开头时才剥离。

**已启动 MATH-500 主表 repetition 0**（2026-09-07 14:52 CST，PID 3349348，fingerprint `4856d4ef5673`）：

- 配置：`configs/math500_deepseek_v4_flash_rep0.json`
- 排除 Llama 工程 10 题，共 490 题；B=32，m=4，C=0.8（Dev-Mix 选出 C* 之前的临时阈值），max_tokens=32768，reasoning_effort=high
- 配对根 v2；8 题并发、32 请求在途；无 API seed，本 run 只称 repetition 0
- 日志：`/data/26xyw/path-consistency-results/math500/logs/math500_deepseek_v4_flash_b32_c08_rep0.log`
- 启动后约 1 分钟已见 PC 提交到 80% 前缀、无截断；易题 decode 可低于 SC，难例仍可能全程 stay 在 root

C=0.8 不是该模型的 dev-selected C*。本 run 用于 P0-A 第一份 DeepSeek MATH 数字，不能回写去选阈值。

### 12.7 Llama-3.1-8B 改跑 GSM8K（新协议 SC/PC）

50 题 MATH 结论：8B 在 `max_tokens=4096` 下 L4/L5 几乎不门控，不适合作为 PC 主证据。作者 9/7 决定暂停 Llama MATH-500 490，先在 **GSM8K test 1319** 上跑新论文完整 PC（不是旧 Sentence-PC）。

| 项 | 设置 |
|---|---|
| 模型 | Llama-3.1-8B-Instruct，vLLM TP=1，GPU 1–4 分片 |
| 协议 | paired-root v2；B=32，m=4；20% 网格；Beta 门控 C=0.8（仍非 C\*）；BLC；trial/commit；单层回退；两击重置 |
| Prompt | chat template + `\boxed{}`（Instruct 模型；不用旧 8-shot，以免答案跨泄漏进前缀） |
| 未启用 | PVC 查找表（长度策略仍是 gate 通过则前进一档）；Dev-Mix C\* |
| 输出 | `/data/26xyw/path-consistency-results/gsm8k/`；shard 0–3 round-robin |

旧 `results/SC_seed0.json` 等是 Llama-3-8B Sentence-PC B=20，不得与本 run 混算节省率。

**已启动**（2026-09-07 16:07 CST）：GPU 1–4 各一卡 TP=1，round-robin 分片 330/330/330/329 题。

**已完成**（最晚 shard 2026-09-07 22:58 CST，墙钟 6.4–6.8 h）。四路 JSONL 齐，1319/1319 配对。完整 G0 判断见 `/data/26xyw/path-consistency-results/reports/g0_matrix_go_nogo.md`。

| 口径 | 数字 |
|---|---|
| SC | 1215/1319 = 92.12% |
| PC | 1210/1319 = 91.74%（−0.38 pp；单侧 95% 下界 −0.83 pp，过 −1.0 界） |
| 翻转 | sc_only 8，pc_only 3 |
| Decode | 8316 → 5039 /题（−39.4%） |
| 最终比例>0 | 1178/1319；其中 1149 题到 0.8 |
| 从未提案 | 98 题与 SC 逐条相同 |

DeepSeek MATH-500 490 亦已 complete：SC 430/490 = 87.76%，PC 428/490 = 87.35%（−0.41 pp，单侧下界 −1.02 pp 擦线），decode −34.5%，L5 仍门控并 −32% token。

### 12.8 G0：不要开完整矩阵，开窄队列

**协议可扩大；完整 3×4 主表现在不能开。** 缺 Dev-Mix / AIME / GPQA JSONL，缺每模型 \(C^\*\)，缺 PVC。Llama MATH-500 490 维持暂停。下一步：JSONL → Dev-Mix \(C^\*\) → DeepSeek AIME24+25。

### 12.9 Qwen3-32B MATH-500 490（V0 eager TP=4）

作者 2026-09-08 决定在 GPU 1–4 空闲窗口启动 Qwen3-32B × MATH-500 主表 seed 0，不再等 CUDA-graph。

| 项 | 设置 |
|---|---|
| 模型 | `/data/26xyw/Qwen3-32B`，vLLM **V0 + eager**，TP=4，GPU 1–4 |
| 协议 | paired-root v2；B=32，m=4；Beta C=0.8（仍非 C\*）；BLC；trial/commit；单层回退 |
| 长度 | 逻辑窗口仍是 4 条共享前缀；**物理 micro-batch=2**（计划 §3.3 KV 不足时的允许拆分） |
| 采样 | thinking on；T=0.6，top-p=0.95，top-k=20；max_tokens=32768；max_model_len=34816 |
| 题集 | MATH-500 排除工程 10 题，共 490；配置 `configs/math500_qwen3_32b_b32_c08_seed0.json` |
| 未启用 | CUDA-graph（跨 NUMA TP all-reduce 仍失败）；PVC 查找表；`--store-text`（32k thinking 文本过大） |

C=0.8 不是 Dev-Mix \(C^\*\)。本 run 写入 Table 1 的 Qwen × MATH-500 格，但不能回写去选阈值。

**已启动**（2026-09-08 12:34 CST，PID 3329564，fingerprint `9f54d1c4ba0d`）：

- GPU 1–4 TP=4，权重每卡 15.39 GiB，KV 4.90 GiB；`max_model_len=34816` 时并发约 **2.30x**，因此物理窗口必须是 2
- 引擎 V0 eager 约 60 s 加载完成，已进入 `running`，计划 490 题
- 日志：`/data/26xyw/path-consistency-results/math500/logs/math500_qwen3_32b_b32_mt32768_c08_seed0.log`
- JSONL 可续跑；控制台清单 `p0a-qwen-math500`

**已暂停**（2026-09-08 12:48 CST）：作者指出 V0 eager 过慢。进程已杀掉，manifest 标 `paused`，GPU 1–4 已释放。未完成任何一题，JSONL 为空。

**V1 CUDA-graph 诊断修正**（2026-09-08）：先前把 `NCCL invalid usage` 归因于 GPU1–4 跨 NUMA（SYS/UPI）不完整。同机 `llm-ai-infra` 已在 **同一组 GPU 1–4、同一 vLLM 0.8.5** 上用 **Qwen3-30B-A3B** 跑通 V1 CUDA-graph（eager 16.5 tok/s → graph 108–116 tok/s）。关键差别是 NCCL：torch cu118 自带 NCCL 2.21.5 按 CUDA 11.0 编译，不能 capture graph；A3B 脚本把 `VLLM_NCCL_SO_PATH` 指到 CUDA 12.2 自编译的 `~/data/qwen3.8-infra/cache/nccl/libnccl.so.2`，并前置 `/usr/local/cuda-12.2/lib64`。同日作者决定本地 Qwen 主模型改为 A3B（§12.10），不再用 32B 打主表。`run_math.py` 在 import torch/vLLM 之前安装该 NCCL。

**短探测结果**（`probe_qwen32_v1_graph.py`，不为 490）：

| 尝试 | 设置 | 结果 |
|---|---|---|
| capture max=512（A3B 默认 shape） | V1 + CUDA-12.2 NCCL，`max_num_batched_tokens=512` | **不再报 NCCL invalid usage**；在 `torch.cuda.graph` 里 **CUDA OOM**（dense 32B 激活远大于 MoE A3B） |
| capture max=32 | sizes `{32,16,8,4,2,1}`，`max_num_seqs=2`，KV override 128 | **成功**：V1 引擎；graph capture 3 s / **0.55 GiB/卡**；init 58 s；greedy 输出 `323`（4 tok / 0.144 s，短序列不能当 decode 速率） |
| n=2 × 128 tok 测速 | 同上 | GPU 3–4 被其他用户 `llama-server`（25yww，Qwen3.8-27B GGUF）占走，未测到稳定 tok/s |

MATH-500 490 **不要用 32B V0 eager 续跑**。本地 Qwen 主表改走 A3B，见 §12.10。

### 12.10 本地 Qwen 主模型改为 Qwen3-30B-A3B（2026-09-08）

作者决定：主表本地 Qwen 从 **Qwen3-32B** 换成 **Qwen3-30B-A3B**（`Qwen3MoeForCausalLM`，checkpoint `~/data/Qwen3-30B-A3B`）。原因是 4×4090 上 A3B 已验证 V1 CUDA-graph（108–116 tok/s），32B dense 即使 NCCL 修好，512-token capture 仍 OOM，eager 无法支撑 thinking-on 的 32k 协议。

| 项 | 冻结设置 |
|---|---|
| 模型 | Qwen3-30B-A3B BF16，vLLM **V1 + CUDA graph**，TP=4，GPU 1–4 |
| 协议 | paired-root v2；B=32，m=4；Beta C=0.8（仍非 C\*）；BLC；thinking on |
| 采样 | T=0.6，top-p=0.95，top-k=20；max_tokens=32768；max_model_len=34816 |
| Serving | CUDA-12.2 NCCL；capture sizes 256…1；`num_gpu_blocks_override=8192`；物理窗口默认 4 |
| 启动 | `bash run_math500_qwen30a3b.sh`；配置 `configs/math500_qwen3_30b_a3b_b32_c08_seed0.json` |

C=0.8 仍不是 Dev-Mix \(C^\*\)。32B 的 paused run `9f54d1c4ba0d` 不续跑、不进 Table 1。

### 12.11 DeepConf / CPT 单卡 Llama GSM8K smoke（2026-09-08）

在 A3B MATH-500 等待 GPU 1–4 的窗口内，用 **一块 RTX 4090（物理 GPU 2）** 跑官方 DeepConf / CPT 的最小 Llama 接口。不升级 PC conda（torch 2.6.0+cu118，vLLM 0.8.5）。隔离环境：`/home/26xyw/data/envs/deepconf-cpt-smoke`（`--system-site-packages` + `sentence-transformers`）。官方仓库已在 `~/data/path-consistency-baselines/{deepconf,CPT}`。A3B wait loop 未杀。

题集：GSM8K test JSONL 用 seed=2027 shuffle 后 10 题（source rows 321,1246,29,1277,559,1072,1283,70,157,1113）。Prompt：Llama chat template + `\boxed{}`（无 thinking switch）。**不是**旧 Sentence-PC，也未使用 C=0.8 / \(C^\*\)。

| 方法 | 口径 | n | Acc | tok/题 | 墙钟 | 官方 vs 适配 |
|---|---|---:|---:|---:|---:|---|
| **DeepConf-online** | majority（官方 `final_answer`） | 10 | **7/10** | 3602 | 186 s | 官方 warmup→conf_bar→final + 官方 voting；引擎改为 vLLM 0.8.5 V0 `SamplingParams.logits_processors`（无 v0.10 `WrappedPerReqLogitsProcessor`） |
| DeepConf 其他官方 vote | min-window / tail / top10 filter 等 | 10 | 8/10 | 同左 | 同左 | 同一 trace pool |
| **CPT-adapted-llama** | 4 workers 多数票 | 10 | 8/10 | 991 | ~150 s（含加载） | 官方 `code/qwen/math.py` 黑板循环；**不是公平 CPT 比较** |

**DeepConf 细节**

- 标签写 **DeepConf-online**，因为 V0 logits-processor 探针通过（强制 EOS 在 5 token 处停止）。预算是 smoke：`warmup=8`，`total_budget=16`，`confidence_percentile=10`，`window=2048`，`max_tokens=1024`。**不是**论文主表 \(N_{\mathrm{init}}=16,B=32\)。
- 官方 `utils.compute_confidence` 对 vLLM 0.8.5 的 logprob 字典求均值时会吃到 `-inf` padding，`conf_bar` 变成 NaN，在线过滤失效。适配层改为 **有限值 top-20** 后 `conf_bar` 有限（约 0.22–0.71）；65/80 条 final traces 因 `min_conf < conf_bar` 被标成 `gconf_threshold` 并踢出投票。GSM8K 轨迹远短于 2048，**decode 期 EOS 基本不触发**，所以 mean tokens 与修 NaN 前相同。这不是 AIME 长推理上的 token 节省证据。
- 现成 `path_consistency/deepconf_offline.py` 仍只是离线视图；本次才是官方控制流 + 在线 processor 适配。
- 输出：`/home/26xyw/data/path-consistency-results/deepconf-cpt/deepconf_llama31_gsm8k_smoke10.jsonl` 与 `.summary.json`。

**CPT：接口能跑，但不能当公平比较**

无适配时官方入口在 Llama 上直接失败：

1. `python code/qwen/math.py` 把 `code/qwen` 放进 `sys.path[0]`，`import math` 遮蔽标准库，循环导入崩溃。
2. 官方 `LLM()` 不设 `max_model_len`，Llama-3.1 默认 131072，单卡 KV 只能存约 31536 token，引擎拒绝启动。
3. Llama 没有 thinking 开关；`enable_thinking=True` 被 tokenizer 忽略。40 条 completion 的 `reasoning_content` 全空，`</think>` 截断/finalize **从未发生**。

`scripts/run_cpt_llama_adapter.py` 只补了（1）（2）：以模块方式加载官方代码，并设 `max_model_len=4096`、`enforce_eager=True`。之后黑板循环可以跑：workers 写出 `[BB_WRITE]`，10 题共 64 个 new_items。这只证明代码路径在 Llama 上能转，**不能**写成 matched-budget CPT vs PC/DeepConf（4 workers vs 16 traces；无 thinking 协议）。公平 CPT 仍须官方 Qwen 入口 + Qwen3-30B-A3B。

输出：`.../cpt_llama31_gsm8k_smoke10/results/` 与 `results_bb/`；汇总 `/home/26xyw/data/path-consistency-results/deepconf-cpt/smoke_llama31_gsm8k.json`。

### 12.12 DeepSeek-V4-Flash × AIME24/25 与 GPQA-D（2026-09-08）

作者要求在 MATH-500 490 已完成后，用同一 API 协议跑 AIME 与 GPQA 的 SC/PC。C=0.8 仍不是 Dev-Mix \(C^\*\)，不得回写选阈值。

| 项 | 设置 |
|---|---|
| 模型 | `deepseek-v4-flash`，官方 beta prefix completion；BLC（沿用 MATH-500 smoke） |
| 协议 | paired-root v2；B=32，m=4；reasoning_effort=high；max_tokens=32768 |
| 并发 | 每数据集 4 题 / 16 请求在途（三任务并行，总在途约 48） |
| AIME | 整数 0–999 exact match；prompt 与 MATH 相同 |
| GPQA | 四选一 A–D；选项按 `seed=2027` + Record ID 洗牌后写入 JSONL |
| 数据 | HF 当时不可达；AIME/GPQA 从本机官方字段副本转入 `~/data/Datasets/*.jsonl`，题目未写入仓库 |

启动脚本：`scripts/run_deepseek_aime_gpqa.sh`。**已启动**（2026-09-08 15:07 CST）：

| tag | fingerprint | PID | n |
|---|---|---:|---:|
| `aime_2024_deepseek_v4_flash_b32_c08_rep0` | `e4cd6dcd6672` | 501549 | 30 |
| `aime_2025_deepseek_v4_flash_b32_c08_rep0` | `771c0b2351c6` | 501550 | 30 |
| `gpqa_diamond_deepseek_v4_flash_b32_c08_rep0` | `a8a29bd70ad3` | 501551 | 198 |

输出：

- `/data/26xyw/path-consistency-results/aime/`
- `/data/26xyw/path-consistency-results/gpqa/`

启动后约 1 分钟：AIME 2025 第 1 题 SC/PC 均对（70），PC 到 0.8 前缀，decode 11206→5254；GPQA 第 1 题均对，PC 到 0.8；retry=0，无截断。C=0.8 不能回写去选阈值。

### 12.13 Llama-3.1-8B 其余主数据集（V1 CUDA-graph，GPU 1–3）

A3B 等待已取消；3 卡无法跑 A3B。作者要求先把 Llama 剩余主集铺上 GPU 1–3。GSM8K 1319 已完成，不再重跑。

| 卡 | 数据集 | tag | 设置 |
|---|---|---|---|
| GPU 1 | MATH-500 490（排除工程 10 题） | `math500_llama31_b32_mt8192_c08_seed0` | V1 graph TP=1 |
| GPU 2 | GPQA-Diamond 198 | `gpqa_diamond_llama31_b32_mt8192_c08_seed0` | 同上 |
| GPU 3 | AIME 2024 → AIME 2025 | `aime_2024/2025_llama31_b32_mt8192_c08_seed0` | 同年卡顺序跑，保持占用 |

协议：paired-root v2；B=32 m=4；C=0.8（仍非 \(C^\*\)）；T=0.6 top-p=0.9；**max_tokens=8192**（计划冻结上限，不用 initial50 的 4096）；`gpu_memory_utilization=0.92`；capture `{32…1}`。Llama 无 thinking 开关。启动脚本：`scripts/run_llama31_remaining.sh`。

**已启动**（2026-09-09 13:00 CST）。三卡均为 vLLM **V1**，`enforce_eager=False`，graph capture 成功（约 0.20 GiB/卡）。显存约 22.1 GiB/卡（util 96%）。

| tag | fingerprint | PID | GPU |
|---|---|---:|---:|
| `math500_llama31_b32_mt8192_c08_seed0` | `abc27a862c6a` | 1151216 | 1 |
| `gpqa_diamond_llama31_b32_mt8192_c08_seed0` | `dc17da48aca4` | 1151217 | 2 |
| `aime_2024_llama31_b32_mt8192_c08_seed0` | `8e57f7657fc7` | 1151221 | 3 |

KV cache 约 32k–37k token/卡；10240 context 下并发约 3.2–3.7×，窗口 4 会由调度器排队，不必改协议。C=0.8 不能回写去选阈值。








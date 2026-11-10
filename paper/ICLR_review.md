# ICLR 2027 审稿意见

Submission 28921。原文是 2026-11-08 从 OpenReview 粘贴的页面文本，没有改写。Weakness 与 Question 的编号只加在段首，方便和其他服务器上的 [`handoff_rebuttal.md`](../handoff_rebuttal.md) 对照。补实验分工以那份交接为准，不在本文重写。

| 编号 | Reviewer | 日期 | Rating | Confidence |
|---|---|---|---|---|
| R1 | rS6y | 22 Oct 2026（modified 06 Nov 2026） | 2: Weak rejection | 3 |
| R2 | uwQR | 21 Oct 2026（modified 06 Nov 2026） | 1: Clear rejection | 4 |
| R3 | zwLH | 16 Oct 2026（modified 06 Nov 2026） | 2: Weak rejection | 4 |

三位的 Ethics Review 都是 No。

---

## R1 · Reviewer rS6y

Official Review of Submission 28921 by Reviewer rS6y. 22 Oct 2026, 07:52 (modified: 06 Nov 2026, 20:26).

### Paper Summary

This paper proposes path-consistency (PC), a training-free method that cuts the cost of self-consistency (SC). PC samples in windows of four trajectories that share one reasoning prefix, which grows in steps of 20% across windows.

An answer agreement gate, based on the regularized incomplete beta function, decides when to propose an extension. Local confidence selection (LCS) picks the candidate with the highest minimum sliding-window mean log-probability over its new tokens. A trial window then either commits the extension or returns to the last committed prefix. Two consecutive failures stop reuse for that question. All valid answers, including those from failed windows, enter the final majority vote.

With 32 trajectories on two models and five benchmarks, PC cuts generated tokens by 25.8–42.0% (average 34.4%). Accuracy drops by at most 0.4 points.

Presentation: 2: Generally understandable, but requires substantial effort in places.
Informativeness: 2: I learned something interesting or useful.
Soundness: 2: I have minor or fixable concerns about the paper's soundness.

### Critical Strengths

The evaluation covers three model scales, both short and long chain regimes, five benchmarks, and the composition with two early stopping methods. More importantly, the token accounting charges the failed and unprefixed windows in full (Appendix C.3), and it clearly refuses to treat the nominal prefix fractions as the measured savings. This makes the reported token reductions credible as the real token reductions.

Instead of claiming that answer agreement means correctness, the paper proves an exact decomposition but cannot guarantee its sign, and it also constructs the explicit finite-budget cases where the prefix reuse actually harms the accuracy, even the per-trajectory correctness is above 0.5. I think this honesty is valuable.

### Critical Weaknesses

**W1.** The baseline (SC) was proposed in 2023. This paper does not compare with other more recent methods that are mentioned in the related work section, so it is hard to know the real position of PC among current works.

**W2.** The central claim is efficient inference, but it is only supported by the token count. The more decisive metric, which is the wall-clock latency / throughput, is not measured in the regime that really matters, and it is possibly neutral or even worse. The reason is that SC is parallel across all the 32 trajectories, but PC has to serialize the sampling windows, because each window depends on the previous window's outcome and the prefix decision.

**W3.** The accuracy preservation is only concluded from single runs, and the test sets are too small to support such claim. PC never improves over SC and even degrades a little in several settings. Many entries in Table 3 are numerically identical to SC, and AIME 2024/2025 only have 30 questions each (one question is about 3.3pp), so the "matching accuracy" here is actually within the noise.

### Questions For Authors

**Q1.** For the long-chain models, what is the wall-clock latency and throughput of PC compared with SC, under the batched serving that can fully use SC's parallelism (not the serial per-question timing)? Does the ~34% token saving really become some real latency / throughput improvement?

**Q2.** Are the accuracies in Table 3 from single runs? What is the variance across different seeds, especially for the 30-question AIME benchmarks? If the authors can report the mean ± standard deviation over several seeds, it will affect a lot how I read the "maintained accuracy" claim.

**Q3.** How does PC compare with the other reasoning-sharing methods (PoLR, CPT) and also DeepConf? Currently these baselines only appear on the short-chain Llama/GSM8K, and DeepConf is never compared in the experiments.

**Q4.** In a paired way, what is the fraction of questions that PC changes compared with SC, and can it be split into “correct->wrong”and “wrong->correct” flips? The aggregate accuracy may hide the offsetting flips that are caused by the appending prefix errors.

Rating: 2: Weak rejection.
Confidence: 3: I am familiar with this topic and its relevant literature.

### 中文要点

- **W1.** 基线是 2023 年的 SC。相关工作里更新的方法没有比较，看不出 PC 的位置。
- **W2.** 效率主张只有 token。真正要紧的墙钟和吞吐没有在满并行设定下测量，结果可能持平甚至更差，因为 SC 的 32 条可以并行，PC 必须把窗口串行化。
- **W3.** 准确率来自单次运行，测试集也太小。PC 从未超过 SC。AIME 各只有 30 题，一题约 3.3 个百分点，所谓准确率匹配在噪声之内。
- **Q1.** 长链模型在能吃满 SC 并行的批服务下，墙钟和吞吐是多少？约 34% 的 token 节省是否变成真实的延迟或吞吐收益？
- **Q2.** 表 3 是否单次运行？尤其 AIME，多种子的方差是多少？若能给出 mean ± std，会改变对「准确率保持」的判断。
- **Q3.** PC 与 PoLR、CPT、DeepConf 比如何？这些基线现在只出现在短链 Llama/GSM8K，DeepConf 完全没有实验。
- **Q4.** 配对看，PC 改了多少题？请拆成「对→错」和「错→对」。总准确率可能掩盖由错误前缀造成的互相抵消。

---

## R2 · Reviewer uwQR

Official Review of Submission 28921 by Reviewer uwQR. 21 Oct 2026, 05:57 (modified: 06 Nov 2026, 20:26).

### Paper Summary

The paper proposed Path-Consistency (PC); the idea is to try to reuse prefixes on different path to have more efficient self consistency parallel thinking. PC progressively builds a shared prefix that later samples only generate new continuations. They evaluate 3 models across GSM8k, math500, AIME, and GPQA-D, and show comparable accuracy with self-consistency while getting token savings upto 40%.

Presentation: 2: Generally understandable, but requires substantial effort in places.
Informativeness: 1: I learned little or nothing new from this paper.
Soundness: 2: I have minor or fixable concerns about the paper's soundness.

### Critical Strengths

I think the general research question is very timely and important, how to be token efficient while retaining capabilities of reasoning models.

### Critical Weaknesses

**W1.** Limited contribution and novelty. The main results are in Table 3. On reported benchmarks, SC accuracy is not much different from pass@1. This raises the question of why SC is needed in the first place in these settings. Also, the token reductions can be misleading, since Table 3 compares against an inefficient fixed-budget SC setup. The authors compare with similar methods such as PoLR [1] in Table 4, but only on Llama-3.1-8B and GSM8K. These comparisons should be extended to the main experimental settings. Also general idea of PC looks very close to related works and generic.

**W2.** Diversity. Sharing prefixes may hurt diversity. However, because pass@1 is already close to SC accuracy in several evaluated settings, it is unclear whether PC actually retains useful diversity or whether the sampled solutions simply offer little useful diversity to begin with. More direct analysis is needed to distinguish these possibilities.

[1] Jindal, Ishan, et al. "The Path of Least Resistance: Guiding LLM Reasoning Trajectories with Prefix Consensus." ICLR 2026.

### Questions For Authors

**Q1.** Can the authors evaluate another model or benchmark where SC accuracy is significantly higher than pass@1? If PC still retains SC accuracy in that setting, it would provide stronger evidence that the method does not sacrifice too much useful diversity.

Rating: 1: Clear rejection.
Confidence: 4: I have substantial direct research experience on this topic.

### 中文要点

- **W1.** 贡献和新颖性有限。所报基准上 SC 与 pass@1 差得不多，因此为什么还需要 SC 本身就是问题。token 降幅也可能误导，因为对照的是固定预算、偏浪费的 SC。与 PoLR 等的比较只在 Llama-3.1-8B / GSM8K，应扩到主实验。想法与相关工作很接近。
- **W2.** 共享前缀可能损害多样性。多个设定里 pass@1 已经接近 SC，看不出 PC 是保住了有用多样性，还是这些样本本来就没有多少有用多样性。
- **Q1.** 能否再评一个 SC 显著高于 pass@1 的模型或基准？如果那里 PC 仍保住 SC 的准确率，就能更有力地说明没有牺牲太多有用多样性。

---

## R3 · Reviewer zwLH

Official Review of Submission 28921 by Reviewer zwLH. 16 Oct 2026, 12:00 (modified: 06 Nov 2026, 20:26).

### Paper Summary

This paper proposes Path-Consistency, a training-free test-time scaling framework that reduces the token generation cost of Self-Consistency by progressively sharing and extending a validated reasoning prefix across sequential sampling windows. The key observation is that independently sampling full trajectories repeatedly generates similar early reasoning steps, which can be amortized through controlled prefix reuse. Specifically, Path-Consistency combines an incomplete-Beta answer agreement gate to trigger prefix extension, a local confidence selection rule that ranks candidate prefixes by their minimum sliding-window token log-probability, and a trial-and-recovery mechanism that rolls back prefixes upon consecutive agreement failures. The authors also provide a theoretical decomposition of voting accuracy and an error bound under conditional-margin assumptions. Experiments across three models and five reasoning benchmarks show that Path-Consistency reduces generated tokens by 25.8% to 42.0% while largely preserving voting accuracy, and achieves an additional 20% token reduction when combined with early-stopping methods.

Presentation: 3: Clear and easy to follow.
Informativeness: 1: I learned little or nothing new from this paper.
Soundness: 2: I have minor or fixable concerns about the paper's soundness.

### Critical Strengths

The paper identifies substantial early-step redundancy across independently sampled reasoning trajectories and proposes a training-free progressive prefix reuse mechanism with rollback protection. By combining an answer agreement gate, local sliding-window confidence selection, and a trial-and-recovery controller, the method amortizes early reasoning computation while preventing the irreversible lock-in of misleading prefixes.

The proposed approach achieves consistent token reductions on long-chain reasoning models and composes effectively with existing early-stopping techniques. Across three model scales and five benchmarks, Path-Consistency reduces generated tokens by 25.8% to 42.0% with minimal impact on majority-voting accuracy, and provides an additional 20% token saving when integrated with early-stopping methods.

### Critical Weaknesses

**W1.** The core propose-trial-rollback mechanism closely resembles existing Tree-of-Thoughts and Monte Carlo Tree Search paradigms, yet the paper lacks both conceptual discussion and empirical comparisons against tree-search baselines. Extending a shared high-confidence prefix, evaluating it via multi-path rollouts, and backtracking upon poor outcomes is structurally equivalent to a single-branch depth-first tree search, especially in the sentence-level variant presented in the appendix. However, the authors neither cite foundational Tree-of-Thoughts literature nor compare their method against step-level beam search or MCTS under matched token budgets.

**W2.** Partitioning a parallel sampling budget into sequential windows reduces generation concurrency and may degrade end-to-end latency in realistic serving environments. While standard Self-Consistency can generate all 32 trajectories in a single parallel batch, Path-Consistency enforces 8 sequential rounds of 4 trajectories where each window must wait for the previous one to finish before updating the prefix. The evaluation only reports single-GPU timing under small batch constraints and does not analyze how this serialization bottleneck affects wall-clock latency and hardware utilization on high-capacity GPUs where 32 trajectories can be decoded concurrently without throughput saturation.

### Questions For Authors

**Q1.** How does Path-Consistency conceptually and empirically compare against Tree-of-Thoughts, step-level beam search, or rollout-based MCTS methods under an equal token or compute budget? [1][2]

[1] Inference Scaling Laws: An Empirical Analysis of Compute-Optimal Inference for Problem-Solving with Language Models.

[2] Tree of Thoughts: Deliberate Problem Solving with Large Language Models.

Rating: 2: Weak rejection.
Confidence: 4: I have substantial direct research experience on this topic.

### 中文要点

- **W1.** 提议–试验–回退很像 Tree-of-Thoughts 和 MCTS，附录里的句子级变体尤其像单分支深度优先。既没有引用 ToT，也没有在相同 token 预算下对比 step-level beam search 或 MCTS。
- **W2.** 把并行预算切成顺序窗口会降低并发，端到端延迟可能变差。SC 可以一批生成全部 32 条，PC 是 8 轮、每轮 4 条。现有计时只是小 batch 单卡，没有分析 32 路能同时解码且吞吐未饱和时的墙钟和硬件利用率。
- **Q1.** 在相同 token 或计算预算下，PC 与 Tree-of-Thoughts、step-level beam search、rollout MCTS 在概念上和实验上如何比较？

---

## 编号对照

论文主表是单次运行。回复里承认这一点，方差靠额外重复，不从重复里重选 C，也不换成结果最好的一次。现行实验怎么分机器，看 [`handoff_rebuttal.md`](../handoff_rebuttal.md)。

| 编号 | 审稿人要的东西 | 交接 |
|---|---|---|
| R1-W1，R1-Q3 | 主设定上比较更新的方法：PoLR、CPT、DeepConf。现在这些比较只在短链 Llama/GSM8K，DeepConf 没有实验。 | 实验 2 |
| R1-W2，R1-Q1，R3-W2 | 长链模型上，SC 能把 32 条放进一批时的墙钟和吞吐，而不是每题串行计时。 | 实验 1 |
| R1-W3，R1-Q2 | 表 3 是否单次运行；AIME 上多种子的 mean±std。 | 实验 3 |
| R1-Q4 | 配对统计 PC 相对 SC 的「对→错」和「错→对」。已有 DeepSeek 日志可直接回复，不新跑。 | 交接 §0 |
| R2-W1 | 新颖性，以及 SC 与 pass@1 接近、固定预算对照是否误导。长链 PoLR/CPT 并入实验 2。 | 实验 2，其余直接回复 |
| R2-W2，R2-Q1 | 共享前缀是否损害多样性；要一个 SC 明显高于 pass@1 的设定。已有 AIME/GPQA 数字可直接回复。 | 交接 §0 |
| R3-W1，R3-Q1 | 相同 token 预算下对照 Tree-of-Thoughts、step-level beam search、rollout MCTS。 | 实验 4 |

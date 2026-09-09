Official Review of Submission10772 by Reviewer kgyj
Official Reviewby Reviewer kgyj12 Mar 2026, 14:35 (modified: 24 Mar 2026, 21:07)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer kgyjRevisions
Summary:
The paper presents a method to speed up self-consistency that relies on generation by stages. Instead of generating all trajectories all at once, the paper proposes to first generate a few subsets of the traces and expand on the ones that the model is reaching early consensus, which will be used as the prefix for subsequent generation. The paper claims to achieve speedup over self-consistency with the same performance in evaluating the downstream tasks.

Strengths And Weaknesses:
Strength:

The paper is well-presented and with figures and tables
The paper is well-motivated.
Weaknesses:

The paper doesn't mention whether the answer of each level will be masked out. Without that, the subsequent generation will likely to stick to the prior stage, instead of leveraging the diversity from more trajectories
The paper doesn't evaluate on any reasoning models, which are more suitable for methods aiming at reasoning boost. Also, the tasks selected aren't challenging enough to induce long trajectories, which is more proper for evaluating parallel sampling.
The paper doesn't compare or mention the important and related work, Fu, Y., Wang, X., Tian, Y., & Zhao, J. (2025). Deep think with confidence. arXiv preprint arXiv:2508.15260.
The paper doesn't evaluate speedup on the frontier inference engine, such as vllm or sglang. It is unclear how much practical speedup the method achieves
The method given in the paper is ad hoc, without a lot of motivation leading to the proposed multiple level decision.
Soundness: 1: poor
Presentation: 3: good
Significance: 1: poor
Originality: 1: poor
Key Questions For Authors:
What is the motivation behind designing the multiple-level?
Does the prior level conclusion sentences affect the subsequent generation?
Does the method proposed also work for recent model with the reasoning COT?
Limitations:
Yes

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Official Review of Submission10772 by Reviewer ZYMf
Official Reviewby Reviewer ZYMf11 Mar 2026, 18:17 (modified: 24 Mar 2026, 21:07)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer ZYMfRevisions
Summary:
This paper introduces Path-Consistency (PC), a training-free and model-agnostic method designed to accelerate Self-Consistencydecoding in LLMs. PC utilizes an iterative "extract-and-sample" mechanism. During test-time search, it generates a small window of paths, evaluates answer confidence, and extracts the common prefix from high-confidence branches. This prefix is then reused as a reliable intermediate conclusion to initialize and guide subsequent generations. By recycling these early successful reasoning steps, PC prevents the model from wasting compute and tokens on erroneous dead-end paths. Extensive evaluations demonstrate that PC significantly speeds up inference and reduces token consumption while maintaining or improving accuracy, and it seamlessly complements other existing SC optimizations.

Strengths And Weaknesses:
Strength
This paper proposes an effective approach to improve the efficiency of self-consistency.

The proposed method PC is a lightweight, model-agnostic plug-and-play method, and it can seamlessly transition across models and tasks.

Weakness
In Section 4.2, the theoretical analysis heavily relies on an overly idealized assumption: "apart from the correct answer, the model generates only one unique incorrect answer". While in real-world open-ended reasoning, the distribution of errors can be highly dispersed and diverse. Consequently, the authors should provide an empirical analysis of the "Truth Is in the Hands of a Few" scenario. For instance, they could perform large-scale sampling (e.g., 128 or more rollouts) on the base model to determine the ground-truth accuracy of each sample, group the dataset by difficulty, and specifically analyze PC's performance and prefix survival rate on the low accuracy groups.

In Table 5, the branch budget for PC is strictly locked at 20, whereas the budget for the baseline SC is maximized to 40. This evaluation methodology is fundamentally unfair. The ability to restrict the maximum number of generations (i.e., a hard-coded early stop) is not an intrinsic capability of the PC method itself, but an externally imposed hyperparameter. This asymmetric comparison artificially inflates PC's token reduction rate (claimed to be 57.8%) by comparing it against an overly "fattened" baseline. Despite this flaw in the standalone comparison, the combined experimental results of PC+AC and PC+ESC in Table 5 are positive. They demonstrate that the PC mechanism is orthogonal to various confidence- or entropy-based early stopping strategies, and combining them can further improve the sc efficiency.

How the performance of the PC method scales across varying compute budgets (the number of branches, N) is not discussed in this paper.

Soundness: 3: good
Presentation: 2: fair
Significance: 3: good
Originality: 3: good
Key Questions For Authors:
See weakness

Limitations:
See weakness

Overall Recommendation: 3: Weak reject: A paper with clear merits, but also some weaknesses, which overall outweigh the merits. Papers in this category require revisions before they can be meaningfully built upon by others. Please use sparingly.
Confidence: 3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Official Review of Submission10772 by Reviewer vPiv
Official Reviewby Reviewer vPiv04 Mar 2026, 23:42 (modified: 24 Mar 2026, 21:07)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer vPivRevisions
Summary:
This paper proposes Path-Consistency, an inference strategy to improve the efficiency of self-consistency reasoning in large language models. The method extracts high-confidence prefixes from early reasoning paths and uses them to guide the generation of subsequent branches, thereby reducing redundant exploration of unpromising reasoning trajectories. Experiments on reasoning tasks (e.g., mathematical, commonsense, and symbolic reasoning) show that the method can reduce token usage and improve inference latency by about 7.8%–40.5% while largely maintaining accuracy.

Strengths And Weaknesses:
Strengths
The paper addresses an important practical issue: the high computational cost of self-consistency reasoning, which is widely used to improve reasoning accuracy in LLMs.

The proposed method is simple and easy to deploy, as it requires no additional training or model modification and operates purely at inference time.

The idea of using early reasoning prefixes to guide later generation is intuitive and can reduce redundant reasoning paths.

Weaknesses
The related work and baselines are outdated. The paper mainly compares with earlier methods such as self-consistency, adaptive-consistency, and early-stopping self-consistency, while recent work on efficient reasoning and adaptive test-time compute allocation is largely missing from the discussion and empirical comparisons.

The experimental evaluation is limited. Experiments focus on relatively classical reasoning benchmarks and modest-scale models, and the paper does not evaluate the method on stronger modern reasoning models or more challenging benchmarks.

The performance improvements are modest. Accuracy gains are generally small, and the main benefit comes from inference speed improvements. It remains unclear whether similar efficiency gains could be achieved with simpler heuristics.

The approach may reduce reasoning diversity, since early prefix selection biases the generation process toward a subset of reasoning trajectories, but this trade-off is not analyzed in detail.

Soundness: 1: poor
Presentation: 2: fair
Significance: 2: fair
Originality: 2: fair
Key Questions For Authors:
please refer to S & W

Limitations:
yes

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
Official Review of Submission10772 by Reviewer aGNg
Official Reviewby Reviewer aGNg26 Feb 2026, 17:48 (modified: 24 Mar 2026, 21:07)Program Chairs, Senior Area Chairs, Area Chairs, Reviewers Submitted, Authors, Reviewer aGNgRevisions
Summary:
This paper introduces a multi-round decoding method for short-CoT models: path-consistency, which leverages the confidence of earlier-generated answers to identify the most promising prefix and guide the generation of subsequent branches. The authors demonstrate that the path-consistency on Llama3-8B can reduce token consumption on several benchmarks.

Strengths And Weaknesses:
Strengths
No obvious strengths.

Weakness
Experiments are not comprehensive: experiments are conducted on Llama3-8B only. The authors claim that their method is model-agnostic (Lines 062-063), apparently, more models across different model families are needed.

No ablation study on "confidence metric". The authors only use a certain confidence metric. More experiments with other metrics should be conducted.

Lack of baselines, e.g., ToT, MCTS.

The method has limited novelty. The idea is quite similar to PHP [1].

[1] Progressive-Hint Prompting Improves Reasoning in Large Language Models

Soundness: 1: poor
Presentation: 3: good
Significance: 1: poor
Originality: 1: poor
Key Questions For Authors:
Is path-consistency still useful for large reasoning models, e.g., DeepSeek-Distilled-Qwen or Qwen3?

Have you considered the costs of multi-turn generation of path-consistency?

Limitations:
Yes

Overall Recommendation: 2: Reject: For instance, a paper with technical flaws, weak evaluation, inadequate reproducibility, incompletely addressed ethical considerations, or writing so poor that it is not possible to understand its key claims.
Confidence: 5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.
Compliance With LLM Reviewing Policy: Affirmed.
Code Of Conduct Acknowledgement: Affirmed.
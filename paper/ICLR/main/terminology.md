# ICLR 论文术语表

更新日期为 2026-09-24。本文件记录本稿的首选技术表达、定义和使用边界。后续修改摘要、正文、图注和附录前先核对本表。英文论文保持英文，中文仅用于内部解释。

## 使用规则

- 摘要、正文和附录叙述使用全称 **path-consistency** 和 **self-consistency**，句首首字母大写。**PC / SC** 仅用于图表及其标题、图注和数学符号。标题和章节名也不把 consistency 的 c 大写。文献中的原始题名和代码标识不作机械替换。
- 名称中的小写 c 与数学符号无关。前缀门控阈值仍为大写 $C$，一致性分数仍为 $c(n,v_\star)$。
- 同一概念优先使用首选表达。不同含义不得为了表面统一而合并。
- 英文叙述尽量不用冒号和破折号展开解释。保留必要的复合词连字符、模型名连字符和数学记号。
- 优化术语时先更新本表及变更记录，再核对摘要、正文、图注和附录。历史文献名称、实验 tag 和代码接口保持原样。
- 本表是写作约定，不代表所有旧措辞已经完成替换，也不授权修改方法或实验协议。

## 方法与控制流程

| 首选英文 | 中文含义与使用边界 | 避免混用 |
|---|---|---|
| path-consistency | 本文方法名称；图表和数学符号可简写 PC | Path-Consistency、PC-AS、正文叙述中简称 PC |
| prefix level | 前缀占来源轨迹完整推理部分 token 长度的名义比例，包含继承的前缀，不含最终答案；0 表示空前缀，实际端点允许边界调整 | 固定 token 长度、占后续新生成轨迹的比例、保证的 token 节省率 |
| reasoning trajectory | 一条完整采样推理轨迹；图中简写 path 可以保留 | 同一段落无必要地交替使用 trace、path、trajectory |
| sampling window | 共享同一前缀并在全部轨迹完成后更新状态的逻辑窗口 | physical batch；硬件拆批不产生新逻辑窗口 |
| shared prefix | 一个窗口内所有轨迹共用的连续推理前缀 | 多前缀池、共享提示词池 |
| prefix reuse | 使用已有推理前缀作为后续生成的条件 | prefix caching；二者不是同一操作 |
| prefix conditioning | 对给定前缀进行条件生成，强调分布变化 | distribution-preserving reuse |
| answer agreement | 采样答案之间的一致程度，不涉及真值 | correctness、answer accuracy |
| answer agreement gate | 基于答案一致性决定是否允许前缀提议的门控 | correctness verifier |
| Beta agreement score | $c(n,v_\star)$，top-1 versus all-rest 的操作性分数 | calibrated correctness probability、top-2 gate |
| gate threshold | 前缀门控阈值 $C$，默认操作点为 0.8 | $C^*$、dev-selected threshold |
| source selection | 从合格候选轨迹中选择前缀来源 | 与 prefix extension 或 commitment 等同 |
| local confidence selection | 用新增片段的局部 logprob 分数选择前缀来源；Beta 门控与来源排序是不同步骤，不再使用 BLC 作为正文名称；旧图表标签待同步 | Beta-guided Local-Confidence selection、大小写混用、全部 API 决策均有完整 logprob 的说法 |
| prefix extension | 将连续前缀向后延长，描述内容与长度变化 | 已经验证成功或已经提交 |
| prefix proposal | 提议供下一窗口试用的前缀 | prefix commitment |
| prefix promotion | 进入下一名义长度档位的概括表达；涉及状态时优先明确 proposal 或 commitment | 默认已经通过下一窗口验证 |
| trial window | 使用提议前缀生成并检查其一致性的窗口 | committed window |
| prefix validation | 检查 mode 与门控分数是否满足提交或保留条件 | 语义正确性验证 |
| prefix commitment | 试用通过后将前缀加入已提交状态栈 | 仅生成了候选前缀 |
| rollback | 名词；动词用 roll back。失败试用回到最后已提交前缀，已提交前缀失败则回退一层 | 自动纠错、抹除此前投票 |
| recovery mechanism | trial、rollback 和连续失败后禁用复用的控制机制 | 已证明恢复正确率 |
| root window | 不使用推理前缀的窗口 | 仅指第一个窗口 |
| final vote | 所有已完成窗口的有效答案参与的最终投票 | 只汇总通过门控的窗口 |
| plurality vote | 取票数最多的答案，不要求超过半数；首次说明与代码中 majority_vote 的对应关系 | absolute majority requirement |
| fixed prefix-length grid | 固定 20% 档位，门控通过才允许逐档前进 | adaptive PVC controller |

不再将 **extract-and-sample mechanism / loop / procedure** 作为当前方法的名称。需要概括时使用 **controlled prefix reuse** 或 **windowed prefix reuse**。这不否认代码仍会从已有轨迹选择连续前缀。

## 方法类别与组合

| 首选英文 | 中文含义与使用边界 |
|---|---|
| reasoning-sharing methods | 共享型方法的总称，复用或交换已生成的中间推理信息；不意味着各方法使用相同表示或保持相同分布 |
| early-stopping methods | 提前停止型方法的统一类别名称。摘要及类别比较中不再使用 truncation-based methods 或 stopping-based methods 指代同一类方法。AC/ESC 的本文实现停止后续采样窗口，不等同于截断单条轨迹 |
| early stopping | 表示具体的提前停止操作时使用；作 methods 的定语时统一写作 early-stopping，不是另一种方法类别 |
| composition with early-stopping methods | 组合类别的统一表达，保留所组合方法的停止及聚合规则 |
| parent method | 被组合的原方法；组合增量必须相对该方法自身计算 |
| prefix caching | 推理引擎对相同 token 前缀的计算缓存，与选择何种前缀分别讨论 |

摘要可使用 competitive，不在现有证据下写 uniformly superior、best 或 universally compatible。PoLR 比较和 CPT-Llama adapter 结果不能证明 PC 全面优于共享型方法，也不能替代公平的官方 Qwen/CPT 比较。

## 理论与实验

### 表头统一规则（Table 1–5）

- `Accuracy (%)` 表示准确率，SC/PC 并列时放在同一分组表头下。Pass@1 也归于准确率组。
- Table 3、4 不再列 `Correct`。Table 1 的 `Windows` 表示窗口数量。
- `Tokens` 表示平均新生成 token 数，不再混用 `tok`、`Tokens/q`、`New tokens`。统计单位在图注或相邻正文中说明，Table 1 为每题每窗口，其余表为每题。
- Table 1 简写为 `Level / Windows / Prefix / Tokens / Save (%) / Time (%)`。其中 `Prefix` 表示共享前缀 token 长度，`Time (%)` 表示相对时间减少率。
- `Save (%)` 统一表示相对指定基线的 token 节省率；Table 2、4、5 放在 `Tokens` 分组下。
- `Time (s)` 表示平均每题时间；`Time reduction (%)` 表示相对时间减少率，两者不可混用。CPT 的计时范围差异保留在表注中。
- `Δ` 放在 `Accuracy (%)` 分组下，表注说明其单位为百分点；Table 4 为各设置减 SC，Table 5 为 PC 减 SC，不是相对百分比变化。

### 技术表达

| 首选英文 | 中文含义与使用边界 |
|---|---|
| exact finite-horizon decomposition | 有限预算下准确率差的精确分解；不是在线识别有益前缀的算法 |
| conditional-margin assumptions | 正确答案相对各竞争答案的累计条件优势假设；门控不保证其成立 |
| conditional plurality error bound | 有限答案集合上的条件投票误差界，允许依赖历史的采样 |
| generated tokens | 新生成的 decode/reasoning tokens，包含失败窗口和复用 root 的原始成本；不等于输入 token 或 API 费用 |
| token savings | 相对明确基线的生成 token 减少比例；SC 对比与 parent 对比不得混算 |
| generation time | `generation_seconds`，包括复用 root 的原始时间；不能用 `actual_generation_seconds` 代替 |
| accuracy decrease | 准确率下降；绝对差用 percentage points，不能写成相对百分比 |
| answer-format regrading | 已披露的答案格式复判，部分包含模型等价性判断；不等于原始评分或人工复核 |
| matched trajectory budget | 最大轨迹预算一致；不自动意味着 token 或时间一致 |

## 模型与基准名称

- 三模型评估范围为 **Llama-3.1-8B-Instruct**、**Qwen3-30B-A3B**、**DeepSeek-V4-Flash**。不将 Qwen3-32B 替换进来。模型规模描述不得混淆 MoE 总参数与激活参数，不凭名称推算未核验的 DeepSeek 参数量。
- 五个基准为 **GSM8K**、**MATH-500**、**AIME 2024**、**AIME 2025**、**GPQA-Diamond**。表格空间有限时允许 GPQA-D。
- “三个模型、五个基准”描述评估范围，不自动意味着 15 个组合全部完成或使用完全相同的生成协议。
- 当前主表以作者更新的 `05_evaluation.tex` 为准，包含 DeepSeek 与 Qwen 的十个设置；Qwen/MATH-500 表注仍标明 199 题。不得用旧生成表文件中的 TBD 或早期快照覆盖作者表格。
- 当前主表节省范围为 25.8%–42.0%，十设置等权平均 34.4%；Llama/GSM8K 组合实验的增量节省为 18.6%–25.6%，基线为相应 early-stopping 方法。摘要与引言中的旧统计仍需单独同步，不能用旧值改回当前附录。

## 补充句子级实验

- 使用 **sentence-level form of path-consistency** 或 **sentence-level approach** 描述附录 D 的早期探索，按“当前方法之前的短链前缀复用实验”叙述，不假定读者了解旧论文。该部分原为附录 E，删除原附录 D 后自动顺延。
- 这组实验的 prefix level 按推理步骤计数，正文方法的 prefix level 按来源轨迹 token 比例计量，必须在附录开头明确区别。
- 补充句子级实验的 20 条轨迹、随机候选前缀选择，与当前 32 条轨迹、单一共享前缀和 trial/recovery 不得混同。
- 旧图的 confidence 在新叙述中称为 answer agreement 或 agreement score，不解释为答案正确的概率。
- 正式附录统一编辑 `appendix.tex`；`appendix_revision.tex` 为不参与编译的旧稿档案。

## 后续一致性检查

当前引言、方法、结论和正式附录不再使用 extract-and-sample 作为方法名称。final majority vote 与 plurality vote 的解释不能通过改词改变原投票规则。历史源码和未参与编译的档案不作机械替换。

## 变更记录

- 2026-09-24，按作者要求删除原附录 D，原附录 E 改为短链推理上的 sentence-level 早期探索，删除其 DeepSeek-V3/AIME 2025 实验。必要评分口径移至附录 C，正文最新长链主表不变。

- 2026-09-23，融合补充附录，明确 sentence-level prefix variant 的适用范围，统一 local confidence selection 与 answer agreement 的表述。同步当前主表的维护来源；不修改正文实验数据。

- 2026-09-17，按作者最新选择，将类别名称统一为 **early-stopping methods**，覆盖此前选择 truncation-based methods 的约定。摘要两处同步替换，统计范围及数字不变。

- 2026-09-17，摘要不再交替使用 truncation-based 与 early-stopping 类别名。结果范围为正文五项完整比较，即 DeepSeek 的 MATH-500、AIME 2024、AIME 2025、GPQA-Diamond，以及 Llama 的 GSM8K。按 `data/results_snapshot.json` 未舍入 token 均值计算各设置的 `100 * (1 - PC / SC)`，范围为 25.8254729404% 至 39.4073839733%，五个设置等权平均为 30.8301895768%，摘要分别保留一位小数。不是题目加权平均或总 token 加权节省率，不包含 Qwen 待补结果、附录 Llama 现代基准或 DeepSeek/GSM8K 原始诊断行。准确率概括仅适用于这五项比较的已披露评分口径，不声称全设置非劣。

- 2026-09-17，建立术语表。确认 Path-consistency 的拼写，区分一致性与正确性、延长与提交、前缀选择与缓存，以及共享型与截断型方法。摘要合并模型与基准范围，Qwen 完成状态保持待核验。

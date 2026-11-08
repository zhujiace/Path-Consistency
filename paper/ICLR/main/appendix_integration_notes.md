# ICLR 附录融合记录

## 2026-09-24 作者修订（优先于下方初次融合记录）

- 删除原附录 D 的 Llama 补充结果、窗口行为、诊断和配对结果表，不再在 PDF 中呈现。对应源数据文件未删除。
- 主表必要的格式容错评分说明保留为附录 C 的简短段落，`app:grading` 指向此处；删除方法正文中指向已移除诊断表的句子。
- 原附录 E 自动顺延为 D，叙述改为当前方法之前的 sentence-level 前缀复用探索，突出短链任务的节省、准确率权衡，以及对当前方法的启发。
- 删除该部分 DeepSeek-V3/AIME 2025 实验，正文 DeepSeek-V4-Flash、Qwen 的 AIME 2025 结果不变。
- token 分配图仅展示已经包含 SC/PC 的比较图，去掉重复的 SC 单独面板。跨任务 agreement 图单列，GSM8K 的 SC/PC 直方图等宽并排，保留同一轴范围。
- 本次删除前的完整附录与 PDF 临时备份位于 `/tmp/pc-appendix-short-BZy3nn/`。NIPS 原稿及现有原始数据文件不变。

## 2026-09-23 初次融合记录

2026-09-23。本文档只用于维护和核对，不进入论文 PDF。当前附录正文和全部附录表格统一位于 `appendix.tex`。`appendix_revision.tex` 保留为旧稿档案，不再参与编译。此次未修改正文实验表格，也未读取服务器实验结果或启动实验。

## 阅读结构

1. 现有理论证明与有限预算计算。
2. 投票不可逆性、共享前缀依赖及固定分布下的门控分析。
3. 当前方法的状态转换、数据与提示方式、成本定义、消融设置。
4. 当前方法的 Llama 补充结果、窗口行为、缺失分数、计时和评分敏感性。
5. 句子级前缀的补充实验，集中说明与当前方法的协议差异后，按跨任务结果、前缀长度、模型比较、motivation、案例、开销和提示结构展开。

## 来源与处理

| NIPS 来源 | ICLR 对应位置 | 处理 |
|---|---|---|
| 正文 motivation 与 Characterization.pdf | `app:motivation` | 保留按最终正确性分组的 token 分布，不再等同于可安全删除的计算 |
| 正文 `tab:path-consistency` | `tab:sentence_example` | 保留原始生成文本，问题只展示一次，统一黑色前缀和蓝色新增内容 |
| 正文 `tab:speedup` | `tab:sentence_levels` | 保留整体和三级的 token 节省及 speedup，不混用当前 20% 网格 |
| 正文 `tab:result` | `tab:sentence_tasks`、`tab:sentence_time` | 保留十任务、四阈值全部准确率和相对成本数据；删除原有错误/冗余的准确率增量标注 |
| 正文 `tab:level` | `tab:sentence_frequency` | 明确更新频率和最大层数同时变化，不称为单因素窗口大小消融 |
| 正文 `tab:adapt`、`tab:deepseek`，附录 `tab:slm` | `tab:sentence_models` | 合并三模型、三任务结果，保留各自基线，不重新拼接实验 |
| 附录 Mistral 和 AIME 2025 | `tab:sentence_mistral`、`tab:sentence_aime` | 保留数据，删除无证据的泛化与纠错因果推断 |
| Characterization-chage.pdf、conf.pdf、base.pdf、c0.8.pdf | `fig:historical_motivation`、`fig:sentence_agreement` | 原图复制至 figs，补充易读图注，不从图片反推新数值 |
| 附录 prefix correction JSON | `tab:sentence_recovery` | 转为四窗口摘要表，保留答案序列和前缀层数；tokens 按五条轨迹求和为 383、283、89、380 |
| 附录 `tab:overhead` | `tab:sentence_overhead` | 保留时间，限定为句子级实验的测量，不外推为当前完整控制器开销 |
| NIPS 正文单次选择推导 | `app:proofs` 中精确二项式计算 | 使用当前保留全部投票的精确表达，不复用一阶近似的无损保证 |
| 附录 Beta promotion 理论 | `app:gate_analysis` | 保留并严格限定“错误概率集中于单一答案”的结论；用计数阈值公式替换错误的事件包含关系，不假设跨层独立 |
| 附录旧 Python 类和实现清单 | `app:sentence` 设置说明、当前 `app:algorithm` | 不重复已经过时的实现；明确随机候选池与当前单一共享前缀不同 |
| 七组完整 few-shot 提示词 | `tab:sentence_prompts` | 压缩为任务、推理演示内容、最终答案格式；完整原文保留于 NIPS 源文件 |
| 资源、限制、影响声明 | 补充实验末段及现有声明 | 合并重复内容，不复制未经重新核验的具体许可证结论 |

## 必须保留的证据边界

- NIPS 主要设置为 Llama-3-8B、句子级前缀、20 条轨迹；当前方法为 token 比例前缀、单一共享前缀和 trial/recovery，不能汇总节省率。
- NIPS 跨任务表的 GSM8K 基线准确率为 64.1%，方法比较表为 68.4%。保留为独立比较，在表注中明确，不擅自归并。
- NIPS `speedup.pdf` 中 C=0.7 和 C=0.8 的 level-3 排序与 `tab:speedup` 不一致。以表格数值为准，图不进入 PDF。原图另存为 `figs/supp_prefix_speedup.pdf` 供核对，非论文引用资源。
- NIPS `tab:speedup` 另有 96.60、73.37、81.09、80.91、87.01 的平均 token 数，但均值单位未充分说明，且不能与方法比较表的每题 tokens 直接对应。因此论文保留其明确的相对节省数据，不将这些均值伪装成每题总量。
- NIPS 注释掉的重复运行标准差表不属于已渲染正文证据，未提升为正式结果。
- NIPS 中“多次门控独立”“错误前缀存活概率随层数指数衰减”“不损害准确率”等说法未原样迁入。当前附录给出条件明确的计算和边界。
- 水量案例的答案 10 和 14 各出现 7 次，原记录最终答案为 10。附录明确这是并列票数，不把它写成正确答案获得唯一多数，也不据此宣称回退具有因果纠错效果。
- 当前阈值附录以作者最新 Table 6 为准，不保留旧稿的多基线阈值表解释，也不建议从验证集选择阈值。
- 删除了旧附录中 Qwen 尚无结果、对比/消融尚未完成、运行状态、控制台和待办事项等过时叙述。
- 原始 DeepSeek/GSM8K 旧口径诊断不与新主表合并。旧值仍在 `tables/raw_rows.tex`、`ci_rows.tex`、`diagnostic_rows.tex` 和未参与编译的 `appendix_revision.tex` 中保存；原始正确数为 1005/1319 与 1004/1319，不能当作当前主表准确率。

## 图表与术语

- 附录不再调用 `tables/*_rows.tex`，所有正式数值表直接写入 `appendix.tex`。
- 当前附录引用的五个补充 PDF 全部位于 `figs/`，不依赖 `paper/NIPS/` 的相对路径。
- 正文叙述使用 self-consistency、path-consistency、answer agreement、local confidence selection、prefix extension、trial and recovery、early-stopping methods。
- 仅在补充句子级实验中，将 prefix level 明确解释为推理步骤数量；这不改变当前方法中“前缀占完整推理 token 的比例”的定义。
- 引用需求以 LaTeX 注释保留，没有自动添加新文献。

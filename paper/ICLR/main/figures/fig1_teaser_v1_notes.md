# Figure 1 能力展示初版

本版只负责开篇吸引与定位，不画方法流程。尚未插入 `main.tex`，不改变现有图编号。

## 视觉层级

1. 方法名和一句定位语。
2. 左侧 30.8% 平均节省及 25.8% 至 39.4% 范围，右侧约 7% 组合增量。
3. 按实际比例绘制的生成 token 条形图。浅灰条为基线，彩色条为加入 PC 后的消耗；数字表示节省率，非彩色条的占比。

左侧五项来自正文的 Llama/GSM8K 及 DeepSeek 四个现代基准，均直接读取 `data/results_snapshot.json` 的未舍入 token 均值。平均值是五设置等权平均。不是所有已完成设置的统计，不包含附录的 Llama 低收益设置，也不包含尚待核验的 Qwen。

右侧使用正文表格的 AC 2730→2539 和 ESC 2552→2374 tokens/q，各自以 parent 为 100%。由于图中显示一位小数，两者均为 93.0%，额外节省约 7%。不意味着在 SC 基础上另加 7 个百分点。

准确率边界仍需在正式图注交代。左侧 Llama/GSM8K 从 92.12% 降至 91.74%，DeepSeek 四项按已披露复判口径准确率持平。右侧两种组合各损失 3/1319 个正确答案，约 0.23 个百分点。图中的 similar accuracy 是对这些点估计的描述，不是统计非劣结论。此图不支持 PC 优于所有其他共享型方法的论断。

## 建议英文图注

Path-consistency reduces generation cost and composes with early-stopping methods. (a) Across five completed main-text comparisons, generated-token savings range from 25.8% to 39.4%, with an unweighted mean of 30.8%. DeepSeek accuracy is unchanged under the reported regrading, while Llama/GSM8K accuracy changes from 92.12% to 91.74%. (b) Adding PC to AC or ESC reduces generated tokens by a further 7% relative to each parent, with a 0.23 percentage-point accuracy decrease. The bars report generated tokens, not total compute. Additional Llama settings and pending Qwen comparisons are not included in panel (a).

## 文件与检查

- `fig1_teaser_v1.png` 为预览。
- `fig1_teaser_v1.drawio` 为原生可编辑源文件，58 个 cell，非整图嵌入。
- `fig1_teaser_v1.drawio.svg` 为带源 XML 的矢量预览。
- `fig1_teaser_v1.pdf` 为矢量论文版本。
- `build_fig1_teaser.py` 可重新生成；未调用任何模型或实验任务。

已检查 SVG 渲染预览，没有文字溢出、图例重叠或条形对齐问题。当前环境没有 draw.io Desktop，PNG/PDF 由同几何 SVG 经 librsvg/Cairo 渲染，不宣称经过 Desktop 导出验证。

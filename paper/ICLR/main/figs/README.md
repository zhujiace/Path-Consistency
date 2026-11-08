# Figure 1 组合排版

用户提供的 `SC_matrix.pdf`、`PC_matrix.pdf`、`overview.pdf` 保持原样。`figure1_layout.tex` 在左列上下放置 SC、PC 矩阵，右列放置 overview，使用 0.435 / 0.545 倍正文宽度，保留矢量内容和纵横比。

正文入口是 `../01_introduction.tex` 的 Figure 1，引用标签为 `fig:overview`。标题为 **From repeated reasoning to shared-prefix reuse**。子图标题位于各自图片下方，分别为 SC: independently sampled paths、PC: shared-prefix reuse、Overview of SC and PC。`../03_methodology.tex` 恢复三项技术细节图，自动编号为 Figure 2，沿用 `fig:loop`；方法章节同时引用 overview 和细节图。

图注依据图像内容和 `../data/sc_repetition_matrix/README.md` 撰写，矩阵为同题 32 条路径 × 32 个相对 token 区间。灰色不声明全局唯一，整格着色不能用面积精确估计节省。PC 图中约 40% 按已记录的 Llama-3.1-8B-Instruct/GSM8K 总体 decode token 节省 39.4% 解释，来源为 `paper/report_ICLR2027_revision.md` 的实验记录；不作为该样例本身或紫色面积的测量值。右侧为宏观流程示意，答案颜色不代表正确性。

检查结果：TinyTeX 两遍编译通过，无 overfull 或未定义引用。Introduction 中的 Figure 1 浮动至第 2 页，方法章节 Figure 2 位于第 4 页。图中细小标签受原图和组合宽度限制，PDF 可放大查看。方法图已更新为作者重绘的 `../figures/method/method.drawio` 导出的 `method_1.pdf`、`method_2.pdf`、`method_3.pdf`，由 `figure2_layout.tex` 组合。

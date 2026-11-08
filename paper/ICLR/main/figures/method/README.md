# 第三章技术示意图

## 当前作者重绘版本

Figure 2 现使用作者的 `method.drawio`，只对几何位置做微调。原始文件备份为 `method_before_alignment.drawio`，所有单元格 ID、文字、样式（含配色）和连线数量保持不变。调整包括圆点/标签对齐、分支落点对齐，以及新增提议箭头的折线路由；整体移回页面原点附近，便于打开编辑。

`method_1.pdf`、`method_2.pdf`、`method_3.pdf` 是三个同尺寸独立矢量子图，也提供同名 draw.io、SVG、PNG。`method_aligned.png` 是并排检查预览。`figure2_layout.tex` 负责在第 3 节组合子图，标题均位于图片下方，分别为 Answer agreement、Prefix selection、Trial and recovery，保留论文标签 `fig:loop`。

运行 `python3 paper/ICLR/figures/method/export_author_method.py` 可从本次原始备份重放几何微调并导出。该脚本会覆盖当前 `method.drawio`，后续若再次人工编辑，应先更新脚本输入策略，不要直接重跑。导出器直接读取 XML 的形状、颜色、文字、MathJax 公式与连线坐标，使用本地矢量渲染；Helvetica 使用系统对应的 Liberation Sans，数学公式由 MathText 排版。不是 draw.io 桌面端原生导出，原生编辑请使用 draw.io 文件。

图中文字保持作者版本，包括 Same parent mode 和 Disable reuse。图注明确对应模态答案不变、仅连续两次失败才关闭复用，以及有后续窗口且有可行候选时才提出下一次扩展。本次没有擅自修改这些图内文字或条件结构。

## 先前生成版本（保留，非当前正文输入）

组合图 `fig1_method.pdf` 替换 `03_methodology.tex` 的 Figure 1，保留原标签 `fig:loop`。

| 文件名 | 对应正文 | 表达重点 |
| --- | --- | --- |
| `answer_agreement` | §3.1 | top-1 对 all-rest，答案一致性门控 |
| `prefix_selection` | §3.2 | 仅选择支持模态答案的轨迹，仅评分新增片段，选择单条连续前缀 |
| `trial_recovery` | §3.3 | 下一窗口试用，通过则提交，失败则回退，连续两次失败关闭复用 |
| `fig1_method` | Figure 1 | 三个技术的并列组合与共享图例 |

每个文件均提供 PDF、SVG、PNG、draw.io 格式。draw.io 中的色块、圆点、箭头、文字是独立可编辑对象，公式使用 MathJax，并非嵌入整张位图。SVG 字形转为矢量轮廓以避免跨渲染器字体错排，文字编辑请使用 draw.io。PDF/SVG/PNG 与 draw.io 由同一场景描述生成，但本机未安装 draw.io 桌面端，未验证其 MathJax 渲染与导出效果。

## 视觉来源与语义边界

参考 [NOVA 的 ICLR 2025 正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/ef283d62b4bce30854a8d4827f331229-Paper-Conference.pdf) 第 5 页 Figure 4 的并列小图、符号化对象、短箭头和精简标签风格。所有图形重新绘制，不移植 NOVA 的技术组件。

紫色表示已有共享前缀，蓝色表示新增前缀片段。圆点表示答案类别，相同颜色表示同一答案，不表示正确性。候选新增片段的蓝色深浅仅示意不同局部置信度，不是实验观测值。色块数量、宽度及答案点数均为示意，不代表实际 token 数或测量结果。

语义核对来源为 `paper/ICLR/03_methodology.tex` 和 `path_consistency/math_protocol.py` 的 `gate_snapshot`、`lower_tail_local_score`、`select_shared_prefix` 及 trial/commit 状态转移。图中保留正文的 $\mathcal V$、$a_\star$、$n$、$v_\star$、$c(n,v_\star)$、$C$、$s(z)$、$\mathcal W(z)$ 和 $\ell_t$。不将答案一致性解释为正确性证明。

为保持可读性，128/64 局部窗口设置、安全端点检查、缺失 logprob 的 tie-break 和 root 例外留在正文。第三幅图直接画出 trial 回退，已提交前缀的 monitoring 回退及“连续两次”条件由图注明确补充。名义前缀网格上限仍为 0.8；图中的 $r\to r+0.2$ 仅用于可扩展时的提议。

## 重新生成

所有表示 token 的单元格统一为 22 × 22 绘图单位，由 `TOKEN_SIZE` 控制，不再允许各阶段单独缩放。图例色标不表示 token。候选行、选中提议、下一窗口试用与提交结果保持相同的单元格尺寸及前缀长度。

在仓库根目录运行：

```bash
python3 paper/ICLR/figures/method/build_method.py
```

脚本检查文字是否超出画布、重新解析输出 draw.io XML。集成时另检查组合预览、SVG 栅格化结果和论文 PDF 中的实际排版。

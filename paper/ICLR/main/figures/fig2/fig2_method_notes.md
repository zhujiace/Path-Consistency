# Figure 2：路径式 Method Overview（第二版）

## 设计依据

已查看 ref/AC.png、ref/PoLR.png 和原 NeurIPS 的 path-consistency.pdf、path-consistency-detail-v2.pdf。采用参考图的“SC对照 + 方法主视图”、彩色路径条和集中答案聚合，沿用原PC的extract-and-sample与逐级共享前缀概念。没有引入PoLR的聚类、embedding或其他方法组件。

## 视觉层级

1. **上方灰色带：SC对照。** 每个窗口都是青绿色完整生成路径，说明从头采样。
2. **下方主视图：PC路径演化。** 第1窗从root采样，选定source前缀用金色细框示意。第2、3窗以相同紫色段表示该窗口共享的同一前缀，青绿色表示新生成续写。紫色段逐步变长。
3. **短标签与最终答案。** Gate+BLC、Validate→commit、失败rollback各用短标签。底部收集线强调所有完成答案进入最终投票，包括被控制机制拒绝的窗口。

白底，低饱和紫/青绿/金/蓝四色语义编码，统一无衬线字体，原生圆角路径条与答案圆点。四条紫色段表示同一个前缀在四条轨迹中的复用，不是四个独立候选前缀。

## 语义边界

- 图已标注 **Illustrative sequence**，表示成功promotion示意，不是实验轨迹。路径条长度仅示意复用位置，不按实际token缩放，也不承诺每题都单调缩短续写。
- A/B是示意答案类别，不是正确/错误标签。金色A不代表正确，最终示意答案一致不证明PC与SC准确率相同。
- 保留论文变量 B=32、m=4、C=0.8、r=0/0.2/0.4。省略号表示可继续至8窗，不表示第3窗之后始终提升，未改动80%上限。
- 完整门控/BLC公式、有效答案条件、同parent mode、连续延伸、无可行候选保持状态、不同恢复规则及两击禁用留在§3.1–3.3及图注。根窗口复用和parent停止细节位于附录实验协议（原§3.4已于2026-09-17移出正文）。
- 未替换正文图，未改动现有图序；本轮仅更新指定图资产。

## 建议英文图注

> **Method overview of Path-Consistency.** Unlike SC, which regenerates full trajectories in every window, PC extracts one continuous prefix and shares it across the next window's four continuations. Purple denotes reused tokens and teal denotes newly generated tokens. The illustrated successful sequence advances along the fixed 20% grid: answer agreement gates a BLC-selected proposal, which is tested before commitment. Failed trials revert to the committed prefix; failed committed prefixes roll back one level, and two consecutive failures disable prefixing. All completed answers contribute to final majority voting at the budget or the optional parent stopping condition. A/B indicate answer categories, not correctness; bar lengths are schematic. Paired SC roots and failed windows retain their original costs.

## 文件与验证

- fig2_method.drawio：原生可编辑矩形、圆点、文字和连线，不是单张图片。
- fig2_method.svg：从同一XML几何生成的矢量图，metadata保留源XML。
- fig2_method.pdf：同一SVG经本机librsvg/Cairo导出的单页矢量PDF。
- fig2_method_preview.png：直接由SVG渲染的预览。
- build_fig2_method.py：离线重建脚本；重建会覆盖当前输出，手动编辑前请另存。
- 归档更正：此前复制命令返回成功，但最终核验未发现 archive_v1/ 目录，故本次不宣称旧版已归档或可恢复。当前交付为已核验的第二版三个输出、预览及重建脚本。

已检查SVG预览及PDF渲染：缩短验证标签以修正左右内边距，将“to 8”移出聚合线区域。未见箭头穿字、路径条溢出或无意义连线交叉。本机无draw.io Desktop CLI，未声称做过桌面交互渲染验证。PDF不宣称携带可恢复的draw.io编辑信息，请用.drawio编辑。

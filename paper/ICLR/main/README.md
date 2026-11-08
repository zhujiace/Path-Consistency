# Path-consistency：ICLR 修订工作稿

后续修改英文摘要、正文和图注前，先核对 [术语表](terminology.md)。该表维护首选英文表达、定义边界和变更记录。

本目录是用户 2026-09-15 授权的 NeurIPS 稿修订副本。英文论文入口为 `main.tex`，输出为 `main.pdf`。原始 `paper/src/` 与投稿 PDF 不作修改。

## 修订范围与论证蓝图

按 ARS `academic-paper` 的 revision 写作指导执行，保留提取—采样、答案级 Beta 门控、多级前缀与多数投票的研究主线。用户已授权理论重写、最新实验替换及相应摘要/引言同步；方法文字只修正至现行实现。本轮直接维护 LaTeX 副本与可读修改记录，不冒充 ARS #670 的机器重放授权包，不生成作者逐项裁决或人工阅读证明。`criteria_binding_unavailable`：未建立 ARS 特定评审标准绑定；这不是投稿就绪认证。

预定论证：

1. PC 改变后续采样分布；Beta 是一致性分数，不是正确性证明。
2. 用有限窗口的精确风险分解、总变差界及有明确假设的多答案界替换 Taylor 不降准确率结论。给出完整证明和反例，不把充分条件声称为实验已验证事实。
3. 同预算 SC/PC 主表按完成的原始文件统计，区分原始评分和已有格式复判。
4. AC/ESC 组合报告各自 parent 增量；PoLR/CPT 必须保留，CPT 标注 Llama adapter。
5. 缺失的 Qwen、DeepConf、公平官方 CPT 与多次重复显式保留为待补，不通过推算填实验结果。

负面约束：不改名 PC；不选测试集最优 C；不写 adaptive PVC；不混 B=20 旧 Sentence-PC；不以实际缓存后耗时报 speedup；不声称普遍非劣、多样性保持、统计显著或完整模型无关性。

## 编译与更新

```bash
cd paper/ICLR/main
python3 scripts/build_results.py
python3 scripts/build_windows.py
python3 scripts/build_ablations.py
python3 scripts/check_theory.py
/home/26xyw/.conda/envs/PC/bin/python ../../../analysis/sc_prefix_similarity.py \
  --bootstrap-repetitions 5000 --progress-every 100
export PATH="$HOME/.TinyTeX/bin/x86_64-linux:$PATH"
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

汇总脚本只读 `/data/26xyw/path-consistency-results/`，仅向本目录写小型聚合数据和表格。结果文件、API key、GPQA 题目不复制进仓库。不启动实验，不改控制台。

模板采用 [ICLR 2027 官方作者指南](https://iclr.cc/Conferences/2027/AuthorGuidelines) 链接的 [官方样式包](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip)，2026-09-15 获取；保留官方样式字节。主文目标不超过 9 页。作者信息、资助、贡献、利益冲突及人工复核声明不得由代理代为确认。

详细来源和待办见 `revision_notes.md`；自动汇总见 `data/results_snapshot.json`。表格中的 TBD 是本机尚无完整可核验结果，不等于远程任务没有运行。

2026-09-16 结构修订：主表仅 DeepSeek/Qwen；Llama/GSM8K 独立比较与8窗口演化见正文，现代集失败结果保留附录。新增 `data/window_snapshot.json`、`tables/behavior_rows.tex`、`tables/window_rows.tex` 和原生 LaTeX 矢量图 `figures/window_evolution.tex`，均由只读日志重算。原 motivation 图作为历史材料保留在附录，不能当作当前协议实验。正文的两项机制消融明确标为待运行设计，没有启动实验或填入假结果。

# 2026-01-24 SOTA六宫格图整体评估（严格审稿人视角）

## 背景
针对 `for_submission/figures/sota_comparison_6panel.*` 的 panel (c)/(f) 异常与可读性问题，已完成修正与重绘。本文档用于记录当前图表质量、问题闭环与待改进项。

## 已修正问题（本次完成）
1. **Panel (c) 点数缺失/意义不明**
   - 原问题：仅出现 4 个点，无法覆盖全部协议与双 profile；语义不清晰。
   - 现修正：
     - 对每个 baseline（LEACH/HEED/PEGASIS/SEP/TEEN）分别计算 AERIS‑E 与 AERIS‑R 的 ΔPDR；
     - 以 “Protocol (E/R)” 逐行显示（共 10 行），避免“点消失/含义不明”。
     - 轴标题改为 “ΔPDR (AERIS profile − protocol, percentage points)”，删除 baseline 字样。
     - 明确标注正向含义（Positive favors AERIS）。

2. **Panel (f) 表格可读性差**
   - 原问题：字体过小、表格过挤、阅读困难。
   - 现修正：
     - 增大字体与缩放比例；
     - 更清晰的列宽与视觉对比（正/负 ΔPDR 颜色区分）。

## 当前输出文件（请以 PDF 为准）
- `for_submission/figures/sota_comparison_6panel.pdf`
- `for_submission/figures/sota_comparison_6panel.png`
- `for_submission/figures/sota_comparison_6panel.svg`

生成脚本：
- `scripts/generate_sota_figures.py`

## 严格审稿人视角的总体评价
- **优点**
  - 面板 (c) 结构化显示所有 baseline × profile 的差异，信息完整、解释性更强。
  - 面板 (f) 统计表可读性明显提升，减少“遮挡/过小”质疑。
  - 图内术语已去除“baseline”字样，避免不专业表达。

- **风险与待注意项**
  - 面板 (d) PDR‑Energy Trade‑off 仍较拥挤：如果审稿人质疑“点遮挡”，可考虑缩小点密度或拆分到补充材料。
  - 所有差值仍依赖 `results/sota_comparison.json`：建议在正文或附录说明数据产生流程与模型设定（包长一致 + 能耗模型一致）。
  - 统计表中的 `p` 值数量级很小（e-23/e-39）：需在正文强调“统计显著但实际效应量以 ΔPDR 为准”，避免被认为“仅靠 p 值说服”。

## 下一步建议
1. **论文正文同步更新**
   - 如果 Panel (c) 改为 “Protocol (E/R)” 行式显示，正文描述应强调“逐 baseline 的 profile 差值”。
2. **图表审计**
   - 逐项检查 panel (a)(b)(d)(e) 是否存在重叠或误导视觉的设计（特别是 TEEN 的极端值）。
3. **备份与留痕**
   - 将本次修改写入 Git 历史，避免后续“看不出变化”的争议。

## 本次修正结论
- Panel (c)/(f) 的“异常/遮挡/意义不明”问题已可视化修复。
- 建议你直接查看 PDF 进行最终确认，并以此版本为主稿替换。

# SOTA Comparison Review Update (2026-01-22)

## 1. 本次修正目标
解决 Figure 6-panel 中 (c) 子图意义不清/点数过少、(f) 表格字体过小、以及整体可读性不足的问题。

## 2. 已完成修正
### 2.1 (c) 改为 Dumbbell Plot（AERIS vs Baselines）
- 直接展示 AERIS 均值与各 baseline 均值，并用 95% CI 误差条连接。
- 每行标注 ΔPDR (pp)，避免“4 个点难以理解”的问题。
- X 轴采用 PDR (%)，直观且可解释。

### 2.2 (f) 表格可读性提升
- 增大字号与表格缩放比例，扩展表格占用区域。
- 保留 ΔPDR 与方向列，但避免“胜负化”措辞。

### 2.3 版式微调
- 整体画布略增大，右侧表格区域加宽，避免拥挤。
- (b) 仍保留 jitter 点，确保分布可见。

## 3. 当前输出文件
- 主图：`for_submission/figures/sota_comparison_6panel.pdf`
- 向量版：`for_submission/figures/sota_comparison_6panel.svg`

## 4. 审稿风险提示
- AERIS 能耗高于 baselines，必须在正文明确 trade-off。
- 若需“同机制公平对照”，建议增加 AERIS 关闭可靠性组件的对照实验。

## 5. 结论
(c) 子图已从“抽象点状效应量”改为“均值+置信区间+差值标注”，信息量显著增强；(f) 表格可读性提升。当前版本可满足审稿人可解释性要求。

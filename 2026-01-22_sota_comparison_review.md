# SOTA Comparison Review (2026-01-22)

## 1. 评估范围
本次评估聚焦于 SOTA 对比图 (sota_comparison_6panel) 的版式与数据一致性，重点修复 (c) Effect Size 分面异常、确保能耗模型/包长一致，并输出可用于论文的最终 PDF。

## 2. 已完成修正
### 2.1 统一包长与能耗模型
- 所有协议统一使用 4000 bits 报文长度。
- 统一使用 `ImprovedEnergyModel` 计算 Tx/Rx 能耗，避免 AERIS 与 baseline 之间模型偏差。
- AERIS 通过 `AerisProtocol` 执行，baseline 也基于同等环境参数 (0 dBm, 25C, 湿度 0.5)。

### 2.2 修复 (c) Effect Size 异常
- (c) 子图此前仅显示 1 个点，原因是 Cohen’s d 过大而被 x 轴裁剪。
- 已改为“动态对称范围”，自动根据置信区间上下界设置 x 轴范围，避免裁剪并展示完整误差条。

### 2.3 图表重新导出
- 已重绘并覆盖输出：
  - `for_submission/figures/sota_comparison_6panel.pdf`
  - `for_submission/figures/sota_comparison_6panel.svg`
  - `results/publication_figures/sota_comparison_6panel.pdf`

## 3. 当前数值概览 (30 runs, 200 rounds, 54 nodes, 100x100)
- LEACH: PDR 87.50% +- 0.66%, Energy 18.43 J
- HEED:  PDR 88.61% +- 0.59%, Energy 18.21 J
- PEGASIS: PDR 96.38% +- 0.44%, Energy 18.80 J
- SEP:   PDR 87.12% +- 0.75%, Energy 18.49 J
- AERIS: PDR 99.93% +- 0.03%, Energy 21.31 J

## 4. 审稿视角风险点
- Cohen’s d 数值非常大 (10+ 以上)，易被审稿人质疑“方差过小 / 分布过窄 / 统计意义夸大”。
  建议：在正文中强调差异来源于极低方差，并可补充分布图或标准化说明。
- AERIS 能耗高于所有 baselines，需明确“PDR 换能耗”的取舍关系。
  建议：在结果段增加 trade-off 描述，避免被解读为不公平优势。

## 5. 后续建议
1. 若审稿人仍质疑 effect size 可视化：可考虑轴截断或对数尺度 (需谨慎说明)。
2. 建议补充“单位能耗收益”指标 (如 PDR/J)，用于缓解 AERIS 能耗偏高的质疑。

## 6. 本次文件与脚本位置
- 脚本：`scripts/generate_sota_figures.py`
- 数据：`results/sota_comparison.json`
- 输出图：`for_submission/figures/sota_comparison_6panel.pdf`

## 7. 结论
(c) 子图已修复裁剪问题，图表版式正常，数据展示完整。仍需在论文正文解释 AERIS 高能耗与高 PDR 的权衡关系。

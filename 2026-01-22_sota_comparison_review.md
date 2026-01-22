# SOTA Comparison Review Update (2026-01-22)

## 1. 本次修正目标
解决 Figure 6-panel 中 (c) 效应量异常、(b) 箱线图信息量不足、(f) 表述过度“胜负化”、以及“公平性”措辞不严谨的问题。

## 2. 已完成修正
### 2.1 (c) 改为 PDR 增益而非 Cohen's d
- 原因：Cohen’s d 在 PDR 接近天花板时会产生不合理的夸大值。
- 当前改为展示 **ΔPDR (AERIS − baseline, pp)** 并用 bootstrap 95% CI。
- 结果更直观、可解释且更符合审稿人期望。

### 2.2 (b) 增加抖动点显示分布
- 在箱线图上叠加 jitter scatter，避免 AERIS 分布“贴顶不可见”。
- 让离散分布显式可视化，增强可信度。

### 2.3 (f) 表述修正
- Outcome 改为 Direction，避免“胜负化”措辞。
- 以 ΔPDR (pp) 作为效应方向指标。

### 2.4 公平性文字修正
- Title 从“identical channel models”改为“same geometry/energy model; AERIS includes full reliability stack”。
- 避免被审稿人指责“渠道模型不完全一致”的风险。

## 3. 当前输出文件
- 主图：`for_submission/figures/sota_comparison_6panel.pdf`
- 向量版：`for_submission/figures/sota_comparison_6panel.svg`

## 4. 审稿风险提示
- 仍需在论文正文说明：AERIS 取得更高 PDR 但能耗更高，是可靠性–能耗 trade-off。
- 若审稿人要求“完全一致机制对比”，需补充关闭 AERIS 可靠性模块的对照实验。

## 5. 结论
图 (c) 已从不合理的 Cohen’s d 改为直观的 PDR 增益展示；(b)(f) 信息密度提升且语义更严谨。当前版本符合审稿人对可解释性与统计规范的要求。

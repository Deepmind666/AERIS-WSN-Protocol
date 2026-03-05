# SOTA Comparison Review Update (2026-01-22)

## 1. 本次修正目标
解决图表遮挡问题：
- (c) 点与标注重叠、意义不清；
- (d) 图例覆盖数据点；
- (f) 表格字号偏小。

## 2. 已完成修正
### 2.1 (c) 子图去除内嵌 ΔPDR 文本
- 继续使用 AERIS vs Baseline 的均值 + 95% CI 哑铃图。
- 删除点旁文本，避免遮挡数据；改用外置图例标识。

### 2.2 (d) 图例移出绘图区
- 将 legend 外置到右侧，避免遮挡散点/误差条。
- 保留 Mean ± 95% CI 的 trade-off 展示。

### 2.3 (f) 表格可读性提升
- 进一步增大字号与缩放比例，扩大表格占用区域。

### 2.4 版式微调
- 右侧区域加宽（整体右边距缩小），为外置 legend 留出空间。

## 3. 当前输出文件
- 主图：`for_submission/figures/sota_comparison_6panel.pdf`
- 向量版：`for_submission/figures/sota_comparison_6panel.svg`

## 4. 结论
图 c 与图 d 的遮挡问题已消除，表格可读性显著提升。当前版式更符合审稿人对“图表清晰、无遮挡”的基本要求。

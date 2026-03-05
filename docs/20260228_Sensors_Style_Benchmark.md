# MDPI Sensors WSN 论文图表与语言风格对标分析

**日期**: 2026-02-28 | **目的**: 为 AERIS v73 图表修复提供同期刊风格参考

---

## 一、参考论文来源

1. **Thakur et al. (2025)** — "AI-Driven Energy-Efficient Routing in IoT-Based WSNs: A Comprehensive Review", Sensors 25(24), 7408. 综述类，41页，图表以概念图/分类图为主。
2. **EGWO-NN (2025)** — "Enhanced WSN Lifetime Using EGWO-Optimized Neural Network", Sensors/Eng. Proc. 6(1), 5. 实验类，含 LEACH/PEGASIS/HEED/EEHC 对比，带 t-test 统计。
3. **LEACH Dynamic Clustering (2026)** — "Research on LEACH Protocol Based on Dynamic Clustering and Routing Optimization", Sensors 26(1), 199.
4. **Dual-Phase ML (2026)** — "A Dual-Phase Machine Learning Protocol for Energy Delay Optimisation in WSNs", Sensors 26(2), 611.
5. **Enhanced DEEC (2025)** — "Enhanced DEEC Protocol for WSNs: A Modular Implementation and Performance Analysis", Sensors 25(13), 4015.

---

## 二、图表风格共性（Sensors WSN 实验论文）

### 2.1 标准图表类型

| 图表类型 | 典型用途 | 出现频率 |
|---------|---------|---------|
| Line plot (alive nodes vs rounds) | 网络寿命对比 | 几乎每篇必有 |
| Line plot (residual energy vs rounds) | 能耗衰减曲线 | 高频 |
| Bar chart (PDR/throughput by protocol) | 协议性能对比 | 高频 |
| Bar chart (energy by protocol) | 能耗对比 | 高频 |
| Table (数值摘要) | 精确数值报告 | 每篇必有 |
| Heatmap | 参数敏感性 | 较少见，AERIS 的创新点 |
| Box plot / violin | 分布展示 | 少见但加分 |

### 2.2 配色规范

**主流做法**：
- 每个协议固定一种颜色，全文一致
- 常见配色：蓝(proposed)、红(LEACH)、绿(PEGASIS)、橙(HEED)、紫(TEEN)
- proposed 协议用最醒目的颜色（深蓝或红色），baselines 用较淡色
- 白色背景，无灰色网格（或极淡网格）

**AERIS 当前配色对标**：
- AERIS=#4A86B8(蓝) — 合规，但可以再深一点以突出
- LEACH=#CC8A62(棕橙) — 可接受
- PEGASIS=#68AA94(绿) — 合规
- HEED=#AE8CB7(紫) — 合规
- TEEN=#C3A052(金) — 可接受

**建议**：配色已经不错，无需大改。

### 2.3 线条与标记

**主流做法**：
- 每个协议用不同 marker（o, s, ^, D, v）+ 不同 linestyle
- marker 大小 5-7pt，线宽 1.5-2.0pt
- 图例放在图内空白区域或图下方
- 每面板最多 5-6 条线，超过则拆分面板

**AERIS 问题**：Fig 6 有 15 条线（5协议×3功率），严重违反"每面板≤6线"惯例。

### 2.4 坐标轴与标注

**主流做法**：
- 轴标签字号 10-11pt，刻度标签 9-10pt
- 单位必须标注（如 "Energy (J)", "PDR", "Number of Alive Nodes"）
- Y 轴从 0 开始（除非有充分理由截断）
- 子面板标题用 (a), (b), (c) 格式

**AERIS 问题**：Fig 8 energy 子面板缺单位标注。

### 2.5 误差棒/置信区间

**主流做法**：
- 大多数 Sensors WSN 论文不用误差棒（这是行业弱点）
- 高质量论文用 shaded CI band 或 error bar
- AERIS 已有误差棒 → 这是加分项，保持

### 2.6 统计方法

**主流做法**：
- 大多数论文仅报告 mean 值，无统计检验（行业通病）
- 高质量论文（如 EGWO-NN）用 t-test + p-value
- AERIS 用 Welch t + Holm + Hedges' g → 远超行业平均，是论文亮点

---

## 三、语言风格共性

### 3.1 摘要结构
- 1-2 句背景（WSN 挑战）
- 1 句 gap（现有方法不足）
- 2-3 句方法（proposed 协议核心机制）
- 2-3 句结果（具体数值 + 对比基线）
- 1 句结论

### 3.2 结果描述模式
典型句式：
- "The proposed X achieves Y% improvement in PDR compared to LEACH under Z conditions."
- "As shown in Figure N, X consistently outperforms all baselines across all tested scenarios."
- "Table N summarizes the comparative results, where X demonstrates a Z% reduction in energy consumption."

**注意**：
- 用 "outperforms" 而非 "significantly outperforms"（除非有统计检验支撑）
- 用 "demonstrates" / "achieves" / "exhibits" 而非 "clearly shows" / "dramatically improves"
- 数值精度：PDR 保留 4 位小数或百分比 2 位（如 0.8176 或 81.76%）

### 3.3 局限性表述
- "The current study focuses on PDR as the primary metric; a comprehensive energy analysis is deferred to future work."
- "The simulation assumes ideal MAC-layer conditions; real-world deployments may exhibit additional variability."

### 3.4 AERIS 语言风格评估
- 声明边界控制已经很好（"rank-2 in indoor_office"）
- 统计术语使用规范
- 建议：确保每个 "significant" 都有 p-value 支撑

---

## 四、对 AERIS v73 图表修复的具体建议

### 4.1 Fig 6 (absolute profiles) — 最高优先级

**行业惯例**：每面板 ≤ 6 条线。

**建议方案**：拆为 2 行 × 4 列 = 8 面板
- 上行：AERIS only（3 条线 = tx5/tx10/tx15，用同色系深/中/浅）
- 下行：All protocols at tx10（5 条线 = 标准对比）
- 这样每面板最多 5 条线，符合行业惯例

### 4.2 Fig 5 (delta heatmap) — 高优先级

**行业惯例**：heatmap 在 WSN 论文中少见，这是创新点。但必须：
- colorbar 标签与数值单位一致
- 数值字号 ≥ 8pt
- 建议 colorbar 标签改为 "ΔPDR (tx5 − tx15)"

### 4.3 Fig 8 (tradeoff) — 中优先级

**行业惯例**：能耗图必须标注单位（J 或 mJ）。

### 4.4 Fig 9 (NS-3 trend) — 中优先级

**行业惯例**：显著/非显著标记用不同形状+颜色区分。
- 显著：实心 marker
- 非显著：空心 marker 或灰色

---

## 五、AERIS 相对同期刊论文的优势

1. **统计严谨性远超同行**：Welch t + Holm + Hedges' g，大多数 Sensors WSN 论文仅报告 mean
2. **实验规模领先**：360,000 独立运行 vs 同行典型 1-5 次重复
3. **多环境覆盖**：4 环境 × 3 功率 vs 同行典型 1 环境
4. **诚实报告劣势**：PEGASIS rank-2 被明确承认，同行论文很少这样做
5. **Heatmap 创新**：参数敏感性 heatmap 在 WSN 论文中罕见，是视觉亮点

---

## 六、AERIS 需要注意的差距

1. **能耗分析缺失**：同行论文几乎都有 "alive nodes vs rounds" 和 "residual energy" 图，AERIS 仅有 Fig 8 一个子面板
2. **Fig 6 线条过密**：15 条线远超行业惯例上限（6 条）
3. **部分单位标注缺失**：Fig 8 energy 子面板

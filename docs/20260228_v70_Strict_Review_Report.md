# AERIS v70 严格审稿报告（Sensors MDPI 资深审稿人视角）

**审稿日期**: 2026-02-28
**论文版本**: AERIS_Sensors_MDPI_Submission_Draft_20260228_v70.tex
**数据版本**: s10r_4env_merged_descriptive_20260227.csv (360行)
**审稿人角色**: Sensors MDPI 资深审稿人（严格模式）

---

## 总体判定：Minor Revision

v70 在数据完整性、统计严谨性和声明边界控制方面已达到较高水平。S10R 密集矩阵（4环境×3功率×6节点×5协议×1000重复）全部通过验收（raw_results=30000, error_runs=0, run_tier=publication）。但仍存在 1 个 P0 级和若干 P1/P2 级问题需修复。

---

## P0 — 阻塞发布

### [P0-1] Table 5 (regime_map) Sensitivity block 描述与实际 S10R 数据不一致
- **位置**: v70.tex:194
- **问题**: Table 5 第4行声称 Sensitivity block 范围为 `4 env × 3 node counts × 5 protocols, n=600 per cell`，但实际 S10R 数据为 `4 env × 6 node counts × 5 protocols, n=1000 per cell`。
- **证据**: s10r_4env_merged_descriptive_20260227.csv 包含 6 个 node counts (100,200,300,500,800,1000)，每 cell n=1000；Section 5.8 正文（v70.tex:411）也正确写了 "six node counts (100, 200, 300, 500, 800, 1000), and five protocols (n=1000 per cell)"。
- **影响**: Table 5 是全文证据架构的索引表，审稿人会用它交叉验证所有子节。当前描述与正文和数据矛盾，属于元数据错误。
- **修复**: 将 Table 5 第4行改为 `4 env × 6 node counts × 5 protocols | n=1000 per env-node-protocol cell | Tx-power boundary analysis`。

---

## P1 — 应修复

### [P1-1] indoor_factory 500节点 AERIS tx5 > tx15 反转未在正文讨论
- **位置**: v70.tex:445 附近（tx-power 讨论段）
- **问题**: S10R 数据显示 indoor_factory 500节点 AERIS tx5=0.8569 > tx15=0.8471（delta=+0.0098），即提高功率反而降低 PDR。正文仅笼统提到"non-monotonic"，但未针对此具体反转给出解释或标注。
- **证据**: s10r_4env_merged_descriptive_20260227.csv 行 indoor_factory,5.0,500,AERIS=0.856857; indoor_factory,15.0,500,AERIS=0.847065
- **影响**: 审稿人会追问"为什么 500 节点处 tx5 优于 tx15"，缺乏具体讨论会被视为回避异常。
- **修复**: 在 Section 5.8 或 Discussion 中增加 1-2 句针对此反转的具体解释（如碰撞域扩大效应在中等密度下尤为显著）。

### [P1-2] outdoor_urban 800节点 AERIS 方差异常大（std/mean > 0.4）
- **位置**: 数据层面，论文未讨论
- **问题**: outdoor_urban 800节点 AERIS 三个功率下 std 分别为 0.1258、0.1119、0.0884，相对于 mean (0.231, 0.252, 0.263) 的变异系数 CV > 0.33。这意味着该 cell 的 PDR 分布极度分散，可能存在双峰或重尾分布。
- **证据**: s10r_4env_merged_descriptive_20260227.csv
- **影响**: 高 CV 意味着 mean±std 可能不是该 cell 的良好摘要统计量。审稿人可能质疑该区域的结论可靠性。
- **修复**: 在 Discussion 或 Limitations 中增加一句关于 outdoor_urban 高密度区域 PDR 分布特征的说明，或在图表中标注该区域的 CI 宽度。

### [P1-3] Table 8 (tab:s10_aeris_1000) 仅展示 tx5 vs tx15，遗漏 tx10
- **位置**: v70.tex:413-427
- **问题**: Table 8 标题为"AERIS power-sensitivity snapshot at 1000 nodes"，但只展示 tx5 和 tx15 两列，遗漏了 tx10。S10R 矩阵包含三个功率级别，仅展示两端可能误导读者认为功率响应是单调的。
- **证据**: 实际数据 indoor_office tx10=0.6775 介于 tx5=0.8176 和 tx15=0.5404 之间，但 indoor_factory tx10=0.7291 > tx5=0.6633 且 > tx15=0.5032，呈现非单调模式。
- **修复**: 将 Table 8 扩展为三列（tx5/tx10/tx15），或在 caption 中明确说明"仅展示端点对比，完整三功率数据见 Fig 5"。

### [P1-4] PEGASIS patch-control 零 delta 异常的解释力度不足
- **位置**: v70.tex:390, 581
- **问题**: PEGASIS 在 matched patch-control 中 0/24 显著（即完全不受 MAC 碰撞+多跳中继 patch 影响），论文将其归为"implementation-coupling anomaly requiring dedicated implementation audit"。但未给出任何具体的技术假说解释为什么 PEGASIS 的链式转发路径对碰撞模型免疫。
- **影响**: 审稿人会追问：如果 PEGASIS 对碰撞完全免疫，是否说明碰撞模型对链式协议的建模存在缺陷？这直接影响主矩阵中 PEGASIS 排名的可信度。
- **修复**: 在 Discussion 或 Limitations 中增加 2-3 句技术假说（如 PEGASIS 的逐跳转发天然避免了同时多节点竞争信道的场景，因此碰撞模型对其影响有限）。

### [P1-5] 能耗分析缺失——仅有 Fig 7 trade-off panel 一笔带过
- **位置**: v70.tex:480-490
- **问题**: 全文以 PDR 为主指标，能耗仅在 Fig 7 trade-off panel 中以一个子面板出现，无独立表格、无数值报告、无统计检验。对于 WSN 论文，能耗是核心指标之一，完全缺乏定量能耗分析会被审稿人视为重大遗漏。
- **修复**: 至少增加一个能耗摘要表（如 100 节点 4 环境下各协议的 mean energy），或在 Limitations 中明确说明"本文聚焦 PDR，能耗定量分析为 future work"。

---

## P2 — 建议改进

### [P2-1] Fig 4/5 数据源需确认是否已更新至 S10R
- **位置**: v70.tex:433, 440
- **问题**: 论文引用 `fig4_power_sensitivity_maps.pdf`、`fig5_power_sensitivity_absolute.pdf`。需确认这些图表是否已基于最新 S10R 数据（n=1000, 6 node counts）重新生成，还是仍基于旧版 S10 数据（n=600, 3 node counts）。
- **修复**: 确认图表数据源，若仍为旧版则用 S10R 数据重新生成。

### [P2-2] indoor_office PEGASIS tx15 1000节点 PDR=0.9986 需讨论
- **位置**: 数据层面
- **问题**: indoor_office 环境下 PEGASIS 在 tx15/1000节点时 PDR=0.9986（std=0.0023），几乎完美。而同条件下 AERIS 仅 0.5404。这一极端差距（delta=-0.458）在论文中未被充分讨论。
- **修复**: 在 Discussion 中补充说明 indoor_office 高功率下 PEGASIS 链式转发的天然优势，以及 AERIS 在该条件下的劣势原因。

### [P2-3] Abstract 中 "balanced sampling (n=3200)" 与 S10R 的关系不清
- **位置**: v70.tex:31
- **问题**: Abstract 提到 "balanced sampling (n=3200 independent runs per environment-node-protocol cell)"，这是主矩阵的描述。但 S10R 矩阵（n=1000 per cell）的贡献未在 Abstract 中体现。
- **修复**: 在 Abstract 中增加一句关于 tx-power sensitivity 矩阵规模的描述，或保持现状但确保不会让审稿人误以为 S10R 也是 n=3200。

### [P2-4] Section 3 协议描述缺少伪代码的复杂度分析细节
- **位置**: v70.tex:126-142
- **问题**: Round Pseudocode 小节提到了复杂度，但未给出具体的 Big-O 分析。对于 Sensors 审稿人，轻量级声明需要复杂度上界支撑。
- **修复**: 增加 1 行 per-round 复杂度上界（如 O(n log n) 或 O(n·k)）。

---

## 数据一致性交叉验证结果

| 检查项 | 结果 |
|--------|------|
| S10R reconciliation: 12/12 文件存在 | ✅ PASS |
| S10R: 全部 raw_results=30000, error_runs=0 | ✅ PASS |
| S10R: 全部 run_tier=publication, primary_metric=pdr_expected | ✅ PASS |
| descriptive CSV 行数 = 360 (4×3×6×5) | ✅ PASS |
| significance CSV 行数 = 360 | ✅ PASS |
| 4 环境覆盖完整 | ✅ PASS |
| 3 功率 (5/10/15) 覆盖完整 | ✅ PASS |
| 6 节点 (100-1000) 覆盖完整 | ✅ PASS |
| 5 协议覆盖完整 | ✅ PASS |
| Table 8 tx5 数值与 CSV 一致 | ✅ PASS (0.8176/0.6633/0.0582/0.7679) |
| Table 8 tx15 数值与 CSV 一致 | ✅ PASS (0.5404/0.5032/0.1634/0.5431) |
| Table 5 regime_map Sensitivity block 描述 | ❌ FAIL — 见 P0-1 |
| 图表文件存在性 (fig0-fig8) | ✅ PASS — 全部 PDF 存在 |

---

## 优点确认（审稿人正面评价）

1. **声明边界控制出色**：全文严格区分 legacy 100-node 矩阵、primary large-scale 矩阵、stress block、sensitivity block，不做跨 regime 数值混用。
2. **统计方法规范**：Welch t + Holm 校正 + Hedges' g 三件套齐全，且对大样本下 effect size 膨胀有明确脚注警告。
3. **诚实报告 AERIS 劣势**：indoor_office 大规模下 PEGASIS 优于 AERIS 被明确承认并写入 rank-2。
4. **S10R 密集矩阵规模可观**：360,000 次独立运行（360 cells × 1000 seeds），数据量在 WSN 仿真论文中属于上游水平。
5. **PEGASIS 异常被标记而非隐藏**：patch-control 零 delta 被标记为 anomaly 而非忽略，体现了审计意识。

---

## 修复优先级建议

| 优先级 | 编号 | 预计工作量 | 建议 |
|--------|------|-----------|------|
| P0 | P0-1 | 5 min | 改 Table 5 一行文字，必须在 v71 前修复 |
| P1 | P1-1 | 10 min | 增加 1-2 句讨论 |
| P1 | P1-2 | 10 min | 增加 1 句 + 可选图表标注 |
| P1 | P1-3 | 10 min | 扩展 Table 8 或加 caption 说明 |
| P1 | P1-4 | 15 min | 增加技术假说 |
| P1 | P1-5 | 20 min | 增加能耗表或 Limitations 说明 |
| P2 | P2-1 | 视情况 | 确认图表数据源 |
| P2 | P2-2 | 10 min | 增加讨论 |
| P2 | P2-3 | 5 min | 可选 |
| P2 | P2-4 | 5 min | 可选 |

---

## 结论

v70 整体质量已接近 camera-ready 水平。P0-1 是唯一的阻塞项（Table 5 元数据与实际数据不一致），修复后即可进入 v71。P1 项建议在 v71 中一并修复以提高审稿通过率。同意你下一步做"图表严格收口版"+ v71。

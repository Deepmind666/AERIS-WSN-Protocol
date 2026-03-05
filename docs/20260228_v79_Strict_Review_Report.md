# AERIS v79 严格审稿报告（Sensors MDPI 资深审稿人视角）

**审稿日期**: 2026-02-28
**论文版本**: AERIS_Sensors_MDPI_Submission_Draft_20260228_v79.tex
**数据版本**: s10r_4env_merged_descriptive_20260227.csv (360行)
**审稿人角色**: Sensors MDPI 资深审稿人（严格模式）
**对照基准**: v70 审稿报告 + v72 图表审稿报告 + Codex v73 审查

---

## 总体判定：Minor Revision（偏 Accept）

v79 相比 v70/v73 有显著质量提升。v70 的 P0-1（Table 5 元数据错误）已修复，v73 的 P1 级问题（调试句、caption 冗长、字号过小、术语内部化）大部分已修复。当前无 P0 级阻塞项。剩余问题均为 P2 级版面微调。

---

## v70/v73 修复确认

| 编号 | 原始问题 | v79 状态 |
|------|---------|---------|
| P0-1 | Table 5 regime_map 描述错误 | ✅ 已修复 (line 194) |
| P1-3 (v73) | line 152 调试句 | ✅ 已删除 |
| P1-4 (v73) | caption 过长 | ✅ 已精简（4个核心 caption 均≤2行） |
| P1-2 (v73) | 字号 <8pt | ✅ s77 脚本无 fontsize=7.* |
| P1-4 (v70) | PEGASIS 技术假说 | ✅ line 598 已补充 |
| P1-3 (v70) | Table 8 缺 tx10 | ✅ 已扩展为三列 |
| P2-1 (v73) | S10R 术语 | ✅ 已清除，改为 "power-sensitivity matrix" |
| P2-3 (v70) | Abstract S10R 描述 | ✅ 摘要已重写 |

---

## P0 — 阻塞发布

无。

---

## P1 — 应修复

### [P1-1] "patch-control" 术语未完全规范化
- **位置**: v79.tex:192, 194, 362, 394, 399, 449, 452, 469, 474, 549, 557, 597, 598, 606
- **问题**: "patch-control" 在正文中出现 14+ 次，未按 v74 计划替换为 "matched stress comparison"。Table 5 中保留原名可以接受，但正文段落中高频使用内部术语会让审稿人困惑。
- **修复**: 首次出现定义后（如 "matched stress comparison, hereafter patch-control"），后续至少在 Section 标题和 caption 中统一用学术名。正文讨论段可保留括号内原名。

### [P1-2] outdoor_urban 800节点 AERIS 高方差未讨论（v70 P1-2 遗留）
- **位置**: 数据层面，论文未提及
- **问题**: outdoor_urban 800节点 AERIS 三个功率下 CV > 0.33，mean±std 可能不是良好摘要统计量。
- **修复**: 在 Discussion 或 Limitations 中增加一句关于 outdoor_urban 高密度区域 PDR 分布特征的说明。

### [P1-3] indoor_factory 500节点 tx5>tx15 反转未具体讨论（v70 P1-1 遗留）
- **位置**: v79.tex:444 附近
- **问题**: 正文提到 "non-monotonic" 但未针对 indoor_factory 500节点的具体反转（tx5=0.857 > tx15=0.847）给出解释。
- **修复**: 增加 1 句具体说明（如碰撞域扩大效应在中等密度下尤为显著）。

---

## P2 — 建议改进

### [P2-1] Fig 1 (env_pdr_panel) outdoor_urban inset 效果待确认
- **位置**: PDF p.7, Fig 2
- **问题**: v79 caption 提到 "includes an inset (0--0.16 PDR)"，但 PDF 中 inset 区域较小，低值区 baseline 分离度仍不够清晰。
- **修复**: 可选——稍微增大 inset 尺寸或调整 y 轴上限到 0.25。

### [P2-2] Fig 10 (absolute profiles) 下行 baseline 面板 y 轴范围差异大
- **位置**: PDF p.12
- **问题**: 下行 4 个面板使用 environment-specific y-axis zoom，但 outdoor_urban 面板 y 轴范围与其他三个差异极大（0-0.3 vs 0-1），读者需要反复看轴标签才能对比。
- **修复**: 可选——在每个面板右上角标注 y 轴范围提示，或统一为 0-1 并加 inset。

### [P2-3] 能耗分析仍偏弱（v70 P1-5 遗留）
- **位置**: v79.tex:486, 492
- **问题**: 能耗仅有 Fig 7 tradeoff 一个子面板 + Table 10 统计快照。对于 WSN 论文，这仍然偏薄。
- **修复**: Limitations 中已有隐含说明，但建议显式增加一句 "Quantitative energy analysis beyond the summary snapshot is deferred to future work"。

### [P2-4] indoor_office PEGASIS tx15 1000节点 PDR=0.9986 未充分讨论（v70 P2-2 遗留）
- **位置**: 数据层面
- **问题**: PEGASIS 在 indoor_office tx15/1000节点时 PDR=0.9986，AERIS 仅 0.5404，delta=-0.458。Discussion 中提到了 PEGASIS 优势但未针对这个极端差距展开。
- **修复**: 可选——在 Discussion 中补 1 句关于 indoor_office 高功率下链式转发天然优势的说明。

---

## 数据一致性交叉验证结果

| 检查项 | 结果 |
|--------|------|
| S10R reconciliation: 12/12 文件存在 | ✅ PASS |
| S10R: 全部 raw_results=30000, error_runs=0 | ✅ PASS |
| S10R: 全部 run_tier=publication, primary_metric=pdr_expected | ✅ PASS |
| Table 8 tx5 数值与 CSV 一致 | ✅ PASS (0.8176/0.6633/0.0582/0.7679) |
| Table 8 tx10 数值与 CSV 一致 | ✅ PASS (0.6775/0.7291/0.1363/0.7269) |
| Table 8 tx15 数值与 CSV 一致 | ✅ PASS (0.5404/0.5032/0.1634/0.5431) |
| Table 5 regime_map 描述与实际数据一致 | ✅ PASS |
| 图表文件存在性 (12个 s79 PDF) | ✅ PASS |
| 调试句残留 | ✅ PASS — "Figure asset filenames" 已删除 |
| S10R 术语残留 | ✅ PASS — 正文中无 "S10R" |
| fontsize<8pt 残留 | ✅ PASS — s77 脚本无 fontsize=7.* |

---

## 优点确认（审稿人正面评价）

1. **摘要重写质量高**：v79 摘要结构清晰（背景→方法→结果→结论），claim 边界明确，无过度声明。
2. **Contributions 改为编号小节**：符合 Sensors 审稿阅读习惯，比 v70 的段落式更易扫读。
3. **Caption 精简到位**：4 个核心 caption 均≤2行，符合期刊规范。
4. **Table 8 扩展为三列**：tx5/tx10/tx15 完整展示，消除了单调性误导风险。
5. **声明边界控制持续出色**：全文严格区分 regime，不做跨块数值混用。
6. **统计方法规范**：Welch t + Holm + Hedges' g 三件套齐全。
7. **PEGASIS 异常处理成熟**：line 598 的技术假说（serialized chain forwarding）比 v70 更具体。
8. **Discussion 新增 "Interpretation of Matched Degradation Block" 段**：主动解释 ranking vs absolute-level 的区别，预防审稿人追问。

---

## 修复优先级建议

| 优先级 | 编号 | 预计工作量 | 建议 |
|--------|------|-----------|------|
| P1 | P1-1 | 15 min | patch-control 术语首处定义+后续替换 |
| P1 | P1-2 | 5 min | outdoor_urban 高方差加 1 句说明 |
| P1 | P1-3 | 5 min | indoor_factory 500节点反转加 1 句说明 |
| P2 | P2-1 | 10 min | Fig 1 inset 微调（可选） |
| P2 | P2-2 | 10 min | Fig 10 下行 y 轴提示（可选） |
| P2 | P2-3 | 5 min | 能耗分析 future work 显式声明 |
| P2 | P2-4 | 5 min | PEGASIS indoor_office 极端差距讨论 |

---

## 结论

v79 已达到可送审水平。P0=0，P1 仅剩 3 项（术语规范化 + 两处数据异常讨论），均为 5-15 分钟的文字补充。P2 全部为可选微调。

与 v70 相比，v79 的主要进步：摘要重写、caption 精简、调试句清除、Table 8 扩展、字号提升、术语规范化（S10R 已清除）。

建议：修完 P1-1/P1-2/P1-3 后即可作为 v80 送审稿。P2 项可在审稿人反馈后视情况处理。

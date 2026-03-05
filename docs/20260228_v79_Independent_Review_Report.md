# AERIS v79 独立审查报告（Sensors MDPI 资深审稿人视角）

**审查日期**: 2026-02-28
**论文版本**: AERIS_Sensors_MDPI_Submission_Draft_20260228_v79.tex (623行)
**审查者**: 独立会话全量复核
**数据版本**: s10r_4env_merged_descriptive_20260227.csv (360行) + scalability_4env_v50rigor_20260222_descriptive.csv + env_sensitivity_20260207_205317.json (W8)
**对照基准**: v79 审稿报告 (20260228) + v70/v73 历史审查

---

## 检查项逐项结果

### 1. 结构确认（623行，8 section，12 table，12 figure）

| Section | 行范围 | 状态 |
|---------|--------|------|
| Introduction + Contributions | 37-49 | PASS — 编号小节，3条贡献清晰 |
| Related Work | 51-55 | PASS — 引用覆盖 2024-2026 近期文献 |
| System Model and Protocol | 57-151 | PASS — 含 PDR 定义、CAS/Gateway 公式、伪代码、复杂度 |
| Experimental Setup | 153-220 | PASS — 含 regime map、统计方法、可复现性控制 |
| Results | 221-538 | PASS — 7 个子节，覆盖全部证据块 |
| Discussion | 540-590 | PASS — 含 matched degradation 解释段、部署指导表 |
| Limitations | 592-601 | PASS — 7 条边界声明 |
| Conclusion | 603-608 | PASS — 环境分域结论，无过度声明 |

---

## 总体判定：Accept（条件性）

v79 已达到 Sensors MDPI 可送审水平。P0=0，P1 仅剩术语规范化和两处数据异常讨论（均为文字补充），P2 全部为可选微调。数据一致性交叉验证全部通过。

---

### 2. 调试句残留检查

搜索模式: `debug|TODO|FIXME|HACK|XXX|TEMP|Figure asset|print(`

结果: **0 匹配** — PASS

---

### 3. Caption 精简检查

共 23 个 `\caption{}`，其中核心图表 caption 检查：

| 图表 | 行号 | 长度 | 状态 |
|------|------|------|------|
| Fig 1 (env_pdr_panel) | 248 | 2行 | PASS |
| Fig 2 (ablation_panel) | 278 | 1行 | PASS |
| Fig 3 (scalability_panel) | 330 | 2行 | PASS |
| Fig 4 (tradeoff_panel) | 486 | 2行 | PASS |
| Fig 5 (patch_control_delta) | 474 | 2行 | PASS |
| Fig 6 (delta_maps) | 433 | 2行 | PASS |
| Fig 7 (ns3_trend_panel) | 533 | 2行 | PASS |
| Fig 10 (absolute_profiles) | 440 | 2行 | PASS |

所有核心 caption 均 ≤2行。PASS。

---

### 4. 绘图脚本字号检查（build_sensors_figures_s77.py）

| 参数 | 值 | 状态 |
|------|-----|------|
| axes.labelsize | 12.8pt | PASS |
| xtick.labelsize | 11.0pt | PASS |
| ytick.labelsize | 11.0pt | PASS |
| legend.fontsize | 10.6pt | PASS |
| 正文标注最小字号 | 8.2pt (inset tick) | PASS (≥8pt) |
| fontsize<8pt 残留 | 0处 | PASS |

全部字号 ≥8pt，主体字号 ≥9pt。PASS。

---

### 5. 数据一致性交叉验证

#### Table 2 (100节点 legacy 矩阵, n=30)
- 数据源: W8 `env_sensitivity_20260207_205317.json`
- 验证: 4环境 × 5协议 = 20 cells，mean±std 全部 4位小数匹配
- 结果: **20/20 PASS**

#### Table 3 (scalability 1000节点, n=3200)
- 数据源: `scalability_4env_v50rigor_20260222_descriptive.csv`
- 验证: 4环境 × 5协议 = 20 cells，mean 全部 4位小数匹配
- 结果: **20/20 PASS**

#### Table 8 (tx-power sensitivity AERIS 1000节点)
- 数据源: `s10r_4env_merged_descriptive_20260227.csv`
- 验证: 4环境 × 3功率 = 12 cells，mean 全部 4位小数匹配
- 结果: **12/12 PASS**

总计: **52/52 cells 全部匹配**。PASS。

---

### 6. 术语规范化检查

| 术语 | 搜索结果 | 状态 |
|------|---------|------|
| S10R / s10r | 0 匹配 | PASS — 已清除 |
| patch-control | 约 18 处 | 注意 — 见 P1-1 |
| debug/TODO/FIXME | 0 匹配 | PASS |

---

### 7. 图表质量审查

| 图表 | 文件 | 存在性 | 脚本字号 |
|------|------|--------|---------|
| Fig 0 (workflow) | fig0_aeris_workflow_20260228_s79.pdf | PASS | ≥8.2pt |
| Fig 1 (env_pdr_panel) | fig1_env_pdr_panel_20260228_s79.pdf | PASS | ≥8.2pt |
| Fig 2 (ablation) | fig2_ablation_panel_20260228_s79.pdf | PASS | ≥9.0pt |
| Fig 3 (scalability) | fig3_scalability_panel_20260228_s79.pdf | PASS | ≥9.0pt |
| Fig 4 (tradeoff) | fig4_tradeoff_panel_20260228_s79.pdf | PASS | ≥9.0pt |
| Fig 5 (patch-control) | fig5_s11_patch_control_delta_20260228_s79.pdf | PASS | ≥9.0pt |
| Fig 6 (delta maps) | fig6_s10_delta_maps_20260228_s79.pdf | PASS | ≥9.0pt |
| Fig 7 (NS-3) | fig7_ns3_trend_panel_20260228_s79.pdf | PASS | ≥9.0pt |
| Fig 8 (significance) | fig8_s8_significance_heatmap_20260228_s79.pdf | PASS | ≥8.3pt |
| Fig 9 (consistency) | fig9_s9_s11_consistency_20260228_s79.pdf | PASS | ≥9.0pt |
| Fig 10 (profiles) | fig10_s10_absolute_profiles_20260228_s79.pdf | PASS | ≥9.0pt |
| Fig 11 (sig panel) | fig11_s11_significance_panel_20260228_s79.pdf | PASS | ≥8.5pt |

12/12 图表文件齐全，SVG 备份同步存在。PASS。

---

## P0 — 阻塞发布

无。

---

## P1 — 应修复

### [P1-1] "patch-control" 术语未完全规范化
- **位置**: v79.tex 约 18 处（line 192, 194, 362, 385, 387, 389, 399, 449, 452, 469, 473-478, 549, 597, 598, 606）
- **问题**: "patch-control" 在正文中高频出现，未按学术惯例首处定义后统一用规范名。Table 5 中保留原名可接受，但正文段落中审稿人可能困惑。
- **修复**: 首次出现处定义 "matched stress comparison (hereafter patch-control)"，后续 Section 标题和 caption 中统一用学术名。
- **工作量**: 15 min

### [P1-2] outdoor_urban 800节点 AERIS 高方差未讨论
- **位置**: 数据层面，论文未提及
- **问题**: outdoor_urban 800节点 AERIS 三个功率下 CV > 0.33，mean±std 可能不是良好摘要统计量。
- **修复**: Discussion 或 Limitations 中增加一句关于 outdoor_urban 高密度区域 PDR 分布特征的说明。
- **工作量**: 5 min

### [P1-3] indoor_factory 500节点 tx5>tx15 反转未具体讨论
- **位置**: v79.tex:445 附近
- **问题**: 正文提到 "non-monotonic" 但未针对 indoor_factory 500节点的具体反转给出解释。
- **修复**: 增加 1 句具体说明（如碰撞域扩大效应在中等密度下尤为显著）。
- **工作量**: 5 min

---

## P2 — 建议改进（全部可选）

### [P2-1] Fig 1 inset 区域偏小
- **位置**: fig1_env_pdr_panel outdoor_urban 面板
- **修复**: 可选——稍微增大 inset 尺寸或调整 y 轴上限到 0.25

### [P2-2] Fig 10 下行 baseline 面板 y 轴范围差异大
- **位置**: fig10_s10_absolute_profiles 下行面板
- **修复**: 可选——在每个面板右上角标注 y 轴范围提示

### [P2-3] 能耗分析仍偏弱
- **位置**: v79.tex:486, 492
- **修复**: 建议在 Limitations 中显式增加一句 future work 声明

### [P2-4] indoor_office PEGASIS tx15 1000节点极端差距未充分讨论
- **位置**: 数据层面（PEGASIS PDR=0.9986 vs AERIS=0.5404）
- **修复**: 可选——Discussion 中补 1 句关于链式转发天然优势的说明

---

## 修复优先级汇总

| 优先级 | 编号 | 预计工作量 |
|--------|------|-----------|
| P1 | P1-1 术语规范化 | 15 min |
| P1 | P1-2 高方差说明 | 5 min |
| P1 | P1-3 反转讨论 | 5 min |
| P2 | P2-1~P2-4 | 各 5-10 min，全部可选 |

---

## 结论

v79 通过全部 7 项检查（结构/调试句/caption/字号/数据一致性/术语/图表）。P0=0，P1=3（均为文字补充），P2=4（全部可选）。52/52 数据 cells 交叉验证匹配。建议修完 P1-1/P1-2/P1-3 后作为 v80 送审。

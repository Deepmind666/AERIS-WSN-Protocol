# AERIS v44 严苛审稿报告 (2026-02-21)

> 审稿组: Claude 4.6 (Opus) 五角色模拟
> 审稿文件: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260221_v44.tex` (490 lines)
> 图表版本: S43 (8 figures)
> 基准版本: v42 审稿报告 (`docs/20260220_v42_Strict_Review_Report.md`)

---

## Overall Verdict: Accept (with minor editorial suggestions)

---

## v42 P1 修复验证 (5/5 CLOSED)

| # | v42 Finding | v44 修复位置 | 状态 |
|---|---|---|---|
| P1-1 | NS3_ALIGNMENT_EVIDENCE L134: 26/28 | L134 now reads "25/28" | CLOSED |
| P1-2 | NS3_ALIGNMENT_EVIDENCE Section 3.1 data mismatch | Section 3.1 values now match `ns3_scale_ext_1000_stats.csv` (0.602530, 0.533557, 0.692123) | CLOSED |
| P1-3 | Related Work 缺 baseline 选择理由 | v44 L55: "The baseline set ... is selected to span representative WSN routing paradigms (cluster-head rotation, chain forwarding, residual-energy clustering, and threshold-triggered event reporting) under a comparable low-overhead control budget." | CLOSED |
| P1-4 | CAS Eq.3 权重来源不明 | v44 L83: "The CAS coefficients ... are fixed per environment family from an offline grid search and then held constant across all publication-tier runs." | CLOSED |
| P1-5 | tab:pdr100 缺显著性标注 | v44 L169: "All pairwise AERIS-versus-baseline comparisons in Table~\ref{tab:pdr100} are significant after Holm correction ($p<0.001$)." | CLOSED |

---

## v42 P2 状态追踪

| # | v42 Finding | v44 状态 | 备注 |
|---|---|---|---|
| P2-1 | Gateway α/β/γ 未给数值 | OPEN (optional) | 低风险，可在 camera-ready 补 |
| P2-2 | Ablation 缺 CAS/Skeleton/Safety 总结 | PARTIAL | L197 提及 CAS 不一致，但未明确说 Skeleton/Safety 无效果 |
| P2-3 | 摘要词数 | OK | 177 词 (≤200) |
| P2-4 | fig0 信息密度偏低 | OPEN (optional) | 仍为 0.97\textwidth |
| P2-5 | subsection 标题误导 | PARTIAL | L228 改为 "Statistical Robustness Snapshot (AERIS vs LEACH)"，但表中实际比较对象是 PEGASIS 和 TEEN，标题仍不准确 |
| P2-6 | Conclusion 冗余 | OK | 三段结构合理，无明显冗余 |
| P2-7 | Data Availability 缺 URL | OPEN (optional) | 仍为 "upon final publication" |

---

## 新发现 (v44 独有)

### P0 (阻断级): 0 项

无禁写断言命中。无数据伪造。无统计方法滥用。

### P1 (必修级): 0 项

所有 v42 P1 均已修复，无新增 P1。

### P2 (建议级): 2 项

**P2-NEW-1: tab:robust_snapshot subsection 标题仍不准确**
- 证据: v44 L228 写 "Statistical Robustness Snapshot (AERIS vs LEACH)"
- 但 L237 比较对象是 PEGASIS (indoor\_office)，L238-240 比较对象是 TEEN
- 这是 v42 P2-5 的遗留问题，codex 修复时将 "AERIS vs LEACH" 保留在括号中
- 最小修复: 改为 "Statistical Robustness Snapshot (AERIS vs Strongest Baseline)" 或删除括号内容

**P2-NEW-2: 图表文件名后缀从 s42 更新为 s43，但内容未变**
- 证据: 所有 8 个 figure 引用均改为 `_20260221_s43.pdf`
- 风险: 无——文件名更新是正常版本管理
- 最小修复: 无需修改，仅记录

---

## 数据交叉验证汇总

| 检查项 | 数据源 | 结果 |
|---|---|---|
| S8 tab:scale1000 (L216-219) vs CSV | `scalability_4env_s8_unified_20260215_descriptive.csv` | 20/20 OK |
| NS3 tab:ns3_trend (L399-402) vs CSV | `ns3_scale_ext_1000_stats.csv` | 8/8 OK |
| NS3 25/28 声明 (L414) vs CLAIM_GATE | `NS3_CLAIM_GATE.md` L16/18 | 一致 (25/28) |
| NS3 25/28 vs ALIGNMENT_EVIDENCE | `NS3_ALIGNMENT_EVIDENCE.md` L99/103/134 | 全部一致 (25/28) — P1-1 已修复 |
| NS3 Section 3.1 数据 vs CSV | `ns3_scale_ext_1000_stats.csv` | 4/4 OK — P1-2 已修复 |
| 禁写断言扫描 | v44 tex 全文 | 0 命中 |
| 摘要词数 | v44 tex L31 | 177 词 (≤200) |
| 文献完整性 | bibliography.bib | 17/17 cite keys 全部存在 |

---

## v42→v44 Diff 摘要

v44 相对 v42 的变更为 5 处定向修复 + 图表文件名更新：

1. **L55** (新增): baseline 选择理由句 (P1-3)
2. **L83** (新增): CAS 权重来源说明 "offline grid search" (P1-4)
3. **L169** (新增): tab:pdr100 显著性声明 (P1-5)
4. **L228** (修改): subsection 标题加括号 "(AERIS vs LEACH)" (P2-5 部分修复)
5. **NS3_ALIGNMENT_EVIDENCE.md**: L134 "26/28"→"25/28" + Section 3.1 数据同步 (P1-1, P1-2)
6. **图表引用**: 8 处 `s42` → `s43` 文件名更新

数据表内容未变。无结构性重组。无新增/删除段落。

---

## Gate 判定

### 是否允许投稿？

**允许。** v44 无 P0、无 P1。v42 的 5 项 P1 全部 CLOSED。数据交叉验证全部通过。禁写断言扫描 0 命中。

### 是否需要补实验？

**不需要。** 证据链与 v42 相同，完整且一致。

### 剩余可选修改清单 (按优先级)

| # | 优先级 | 内容 | 涉及文件 | 类型 |
|---|---|---|---|---|
| 1 | camera-ready | tab:robust_snapshot subsection 标题: "(AERIS vs LEACH)" → "(AERIS vs Strongest Baseline)" | v44.tex L228 | tex 修改 |
| 2 | camera-ready | Ablation 段补一句 Skeleton/Safety 无效果说明 | v44.tex L197 | tex 修改 |
| 3 | camera-ready | Gateway α/β/γ 默认值脚注 | v44.tex L74 | tex 修改 |
| 4 | camera-ready | fig0 缩至 0.7\textwidth 或移至附录 | v44.tex L90 | 版面优化 |
| 5 | camera-ready | Data Availability 加仓库 URL | v44.tex L485 | tex 修改 |

以上均为 P2 级建议，不阻断投稿。

---

*报告生成: Claude 4.6 (Opus), 2026-02-21*
*审稿文件: AERIS_Sensors_MDPI_Submission_Draft_20260221_v44.tex (490 lines, 8 figures)*
*数据验证: S8 20/20, NS3 8/8, NS3-doc 4/4, bib 17/17, 禁写 0/10*
*v42 P1 修复: 5/5 CLOSED*

# AERIS v47 严苛审稿报告 (2026-02-21)

> 审稿组: Claude 4.6 (Opus) 五角色模拟
> 审稿文件: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260221_v47.tex` (518 lines)
> 图表版本: S45 (12 figures, 12 tables)
> 基准版本: v44 审稿报告 (`docs/20260221_v44_Strict_Review_Report.md`)

---

## Overall Verdict: Accept (conditional on P1-V47-1 fix)

---

## v44 遗留 P2 修复验证

| # | v44 Finding | v47 状态 | 备注 |
|---|---|---|---|
| P2-NEW-1 | subsection 标题 "(AERIS vs LEACH)" 不准确 | **CLOSED** | L228 now reads "Statistical Robustness Snapshot (AERIS vs Strongest Baseline)" |
| P2-1 | Gateway α/β/γ 未给数值 | OPEN (optional) | 低风险，可 camera-ready 补 |
| P2-2 | Ablation 缺 Skeleton/Safety 总结 | OPEN (optional) | L197 仍仅提 CAS |
| P2-4 | fig0 信息密度偏低 | OPEN (optional) | 仍为 0.97\textwidth |
| P2-7 | Data Availability 缺 URL | OPEN (optional) | 仍为 "upon final publication" |

---

## v44→v47 Diff 摘要

v47 相对 v44 的变更为 1 处 P2 修复 + 4 张新图 + 图表文件名更新：

1. **L228** (修改): subsection 标题 "(AERIS vs LEACH)" → "(AERIS vs Strongest Baseline)" (P2-NEW-1 CLOSED)
2. **L256-261** (新增): fig8 — S8 significance heatmap
3. **L313-318** (新增): fig9 — S9-S11 consistency check
4. **L362-367** (新增): fig10 — S10 absolute tx-power profiles
5. **L399-404** (新增): fig11 — S11 significance & effect-size panel
6. **图表引用**: 12 处 `s43` → `s45` 文件名更新

数据表内容与 v44 完全一致。无文字段落增删。总行数 490→518 (+28 lines，全部来自新图插入)。

---

## 新发现 (v47 独有)

### P0 (阻断级): 0 项

无禁写断言命中。无数据伪造。无统计方法滥用。

### P1 (必修级): 1 项

**P1-V47-1: 21/24 个图表标签无正文交叉引用**
- 证据: 自动扫描发现 24 个 `\label{fig:...}` / `\label{tab:...}`，但仅 3 个有对应 `\ref{}`
- 已引用: `fig:aeris_workflow` (L86), `tab:pdr100` (L169), `tab:s9_pegasis_1000` (L311)
- 未引用: fig1-fig11 中的 11 张图 + tab:ablation_gateway, tab:scale1000, tab:robust_snapshot, tab:rigor_patch, tab:s9_patch_control, tab:s10_aeris_1000, tab:s11_aeris_delta, tab:ns3_trend, tab:regime_map, tab:deployment_summary 共 10 张表
- 风险: MDPI 投稿指南要求 "All figures and tables should be cited in the main text"。审稿人或编辑部会直接退回
- 注意: 这是 v44 以来的遗留问题（v44 为 17/20 未引用），v47 新增 4 张图后变为 21/24
- 最小修复: 在每个图/表首次出现前的正文段落中加入 `Figure~\ref{...}` 或 `Table~\ref{...}` 引用。大部分只需在已有描述句末加引用即可

### P2 (建议级): 3 项

**P2-V47-1: 浮动体密度偏高 (12 figures + 12 tables = 24 个浮动体)**
- 证据: v47 共 12 张图 + 12 张表 = 24 个浮动体，518 行 tex
- v44 为 8 图 + 11 表 = 19 个浮动体，490 行
- 风险: Sensors 无硬性限制，但审稿人可能觉得图表过密。平均每 21.6 行一个浮动体
- 最小修复: 可选——将 fig8/fig10/fig11 中的 1-2 张移至 Supplementary Materials 或合并为复合图

**P2-V47-2: fig9 (S9-S11 consistency) 与正文叙述脱节**
- 证据: fig9 插入在 L313-318（S9 段落中间），但正文未引用 fig9 或解读其内容
- L311 末尾提到 "report that matrix separately below"，但未提及 fig9 的 consistency check
- 风险: 审稿人会问 "这张图说明了什么？"
- 最小修复: 在 L311 附近加一句引用 fig9 并说明 S9-S11 delta 一致性（如 "Figure~\ref{fig:s9_s11_consistency} confirms that S9 and S11 deltas are directionally consistent across protocols and scales."）

**P2-V47-3: fig10 (S10 absolute profiles) 与正文叙述脱节**
- 证据: fig10 插入在 L362-367（S10 段落末尾），但正文未引用 fig10
- L369 的 S10 总结段仅引用了 fig6 的 delta map
- 风险: 同上——无引用的图会被审稿人质疑
- 最小修复: 在 L369 附近加一句引用 fig10（如 "Figure~\ref{fig:s10_absolute_profiles} shows the absolute PDR profiles under both power levels, confirming the non-monotonic direction reversals."）

---

## 新增图表审稿 (Fig8-Fig11)

### fig8: S8 Significance Heatmap (50.2KB)
- 可读性: 双面板 (a) delta + (b) -log10(p) with effect-size annotations
- 审美: 热力图配色清晰，注释字号适中
- 信息密度: 高——同时展示 delta 和显著性强度
- 结论承载力: 补充 tab:robust_snapshot，可视化 S8 全矩阵显著性分布
- 建议: 与 fig3 (scalability panel) 信息有重叠，可考虑合并或移至附录

### fig9: S9-S11 Consistency (35.4KB)
- 可读性: 双面板 (a) cell-wise agreement + (b) AERIS trajectories
- 审美: 简洁，配色与主图系列一致
- 信息密度: 中——核心信息是 S9/S11 方向一致
- 结论承载力: 支持 S9→S11 过渡叙述的可信度
- 问题: 正文无引用（P2-V47-2）

### fig10: S10 Absolute Profiles (35.1KB)
- 可读性: 多面板，dashed/solid 区分 tx5/tx15
- 审美: 线型区分清晰
- 信息密度: 中——补充 fig6 的 delta 视角
- 结论承载力: 直接展示非单调方向反转
- 问题: 正文无引用（P2-V47-3）

### fig11: S11 Significance Panel (54.6KB)
- 可读性: 双面板 (a) Hedges' g map + (b) significant cell count
- 审美: 效应量色阶直观
- 信息密度: 高——24 cell 全矩阵效应量可视化
- 结论承载力: 直接支持 L390 的 "24/24 significant" 声明
- 建议: 与 fig5 信息有重叠，可考虑合并

---

## 数据交叉验证汇总

| 检查项 | 数据源 | 结果 |
|---|---|---|
| S8 tab:scale1000 (L216-219) vs CSV | `scalability_4env_s8_unified_20260215_descriptive.csv` | 20/20 OK (与 v44 一致) |
| NS3 tab:ns3_trend (L427-430) vs CSV | `ns3_scale_ext_1000_stats.csv` | 8/8 OK (与 v44 一致) |
| NS3 25/28 声明 (L442) | `NS3_CLAIM_GATE.md` + `NS3_ALIGNMENT_EVIDENCE.md` | 一致 (25/28) |
| 禁写断言扫描 | v47 tex 全文 | 0 命中 |
| 摘要词数 | v47 tex L31 | 177 词 (≤200) |
| 文献完整性 | bibliography.bib | 17/17 cite keys 全部存在 |
| 图表文件存在性 | for_submission/figures/ | 12/12 s45 PDF 全部存在 |

---

## Gate 判定

### 是否允许投稿？

**有条件允许。** v47 无 P0，数据交叉验证全部通过，禁写断言 0 命中。但 P1-V47-1（21/24 图表无正文交叉引用）必须在投稿前修复——MDPI 编辑部会因此退回。

### 是否需要补实验？

**不需要。** 证据链与 v44 相同，完整且一致。

### 投稿前必须完成 (P1)

| # | 内容 | 涉及文件 |
|---|---|---|
| 1 | 为 21 个未引用的图/表在正文中加 `\ref{}` 交叉引用 | v47.tex 全文 |

### Camera-ready 可选优化 (P2)

| # | 内容 | 涉及文件 |
|---|---|---|
| 1 | 考虑将 fig8/fig10/fig11 中 1-2 张移至 Supplementary | v47.tex |
| 2 | Gateway α/β/γ 默认值脚注 | v47.tex L74 |
| 3 | Ablation 段补 Skeleton/Safety 无效果说明 | v47.tex L197 |
| 4 | fig0 缩至 0.7\textwidth | v47.tex L90 |
| 5 | Data Availability 加仓库 URL | v47.tex L513 |

---

*报告生成: Claude 4.6 (Opus), 2026-02-21*
*审稿文件: AERIS_Sensors_MDPI_Submission_Draft_20260221_v47.tex (518 lines, 12 figures, 12 tables)*
*数据验证: S8 20/20, NS3 8/8, bib 17/17, 禁写 0/10, 图文件 12/12*
*交叉引用审计: 3/24 referenced (P1-V47-1)*

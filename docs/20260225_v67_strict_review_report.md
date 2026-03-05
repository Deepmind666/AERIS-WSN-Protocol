# v67 严格复审报告
# 审稿人：Claude (Sensors/MDPI 视角)
# 日期：2026-02-25
# 稿件：AERIS_Sensors_MDPI_Submission_Draft_20260225_v67.tex
# 基线：v64 两轮审稿报告（strict + harsh，合计 0P0/3P1/10P2）

---

## 0. 审稿范围

本轮基于 v67 tex 全文，核查以下内容：
1. v64 报告中 3 个 P1 的关闭状态
2. v64 报告中 10 个 P2 的关闭状态
3. Table 1/2/3/4 数据一致性（逐值核对源文件）
4. 引用/图文件完整性
5. v66→v67 变更引入的新风险
6. 综合门控判定

v66 变更：系数表 caption 措辞强化、开发语言替换为出版语言、图文件名归一化（fig0-fig8 别名）。
v67 变更：集成用户定稿的 AERIS 流程图（fig0_aeris_workflow.pdf），无文本/数据变更。

---

## 1. v64 P1 关闭状态（3/3 已关闭）

| # | 问题 | 状态 | 证据行号 |
|---|------|------|----------|
| P1-1 | Table 1 caption 未显式标注 collision/relay flags 状态 | **已关闭** | L224: caption 含 "strict collision/relay flags disabled"；L243 补充说明不与 primary 矩阵混用 |
| P1-2 | legacy→primary 多混杂变量未充分讨论 | **已关闭** | L533: Discussion 新增段落，显式指出两个同时变化因素（collision/relay + suppressor 移除），声明 cross-tier 仅用于 scope positioning |
| P1-3 | patch-control 方向与 legacy-primary 方向矛盾未调和 | **已关闭** | L539-540: "Interpretation of Matched Degradation Block" 显式区分 ranking vs absolute-level，结论句 "AERIS keeps relative advantages...while absolute reliability drops" |

---

## 2. v64 P2 关闭状态（5/10 已关闭，5/10 未关闭）

| # | 问题 | 状态 | 证据 |
|---|------|------|------|
| P2-1 | `tab:cas_gateway_coeffs` 未被 `\ref` 引用 | **已关闭** | L83: `Table~\ref{tab:cas_gateway_coeffs}` |
| P2-2 | `Kandris2020` bib key 与 year=2009 不一致 | **未关闭** | bib 中 key 仍为 `Kandris2020`，year=2009 |
| P2-3 | bib 中 37 个未引用条目未清理 | **未关闭** | 73 条目中仅 36 被引用，37 条冗余 |
| P2-4 | Figure 编号跳过 fig4 | **已关闭** | fig0→fig1→...→fig8 连续，v66 已修复 |
| P2-5 | Gateway δ=-0.60 物理含义未解释 | **已关闭** | L83: "negative distance coefficient penalizes long CH-to-BS distance" |
| P2-6 | Table 2 caption 说明数据与 Table 1 共享 | **已关闭** | Table 2 现为 ablation 表，自描述清晰，不再需要共享说明 |
| P2-7 | env_sensitivity JSON 缺少 collision/relay flags 元数据 | **未关闭** | JSON 中无 mac_collision/multihop_relay 字段 |
| P2-8 | 辅助表格移至补充材料 | **未关闭** | 13 表全在正文，未移动 |
| P2-9 | Discussion 中 `\subsection*` 改为编号 | **未关闭** | L42/539/542/571 仍为 `\subsection*` |
| P2-10 | Limitations 补充 PEGASIS 零差异机制假说 | **已关闭** | L580: 定性为 "simulator-coupling artifact requiring dedicated implementation audit"，并限定影响范围 |

---

## 3. 数据一致性核对（Table 1/2/3/4）

### 3.1 Table 1 (tab:pdr100) — Legacy 100-node comparability matrix
- 源文件：`results/mega_experiments/env_sensitivity_20260207_205317.json`
- 核对范围：4 env × 5 protocols × (mean + std) = 40 个数值
- 结果：**40/40 全部匹配**
- 备注：std 使用 ddof=0（总体标准差），caption 已注明 "minor last-digit differences come from rounding conventions"

### 3.2 Table 2 (tab:ablation_gateway) — Full vs no_gateway
- 源文件：`results/mega_experiments/ablation_diag_multi_20260207_205448.json`
- 核对范围：4 env × 3 列 (Full±std, no_gateway±std, Delta) = 12 个数值
- 结果：**12/12 全部匹配**

### 3.3 Table 3 (tab:scale1000) — Primary large-scale matrix at 1000 nodes
- 源文件：`results/mega_experiments/scalability_4env_v50rigor_20260222_descriptive.csv`
- 核对范围：4 env × 5 protocols mean + 4 rank = 24 个数值
- 结果：**24/24 全部匹配**

### 3.4 Table 4 (tab:robust_snapshot) — Significance snapshot at 1000 nodes
- 源文件：`results/mega_experiments/scalability_4env_v50rigor_20260222_significance.csv`
- 核对范围：4 env × (ΔPDR, Hedges' g, Holm p, sig) = 16 个数值
- 结果：**16/16 全部匹配**

### 3.5 汇总
- 四张表格共 92 个数值单元格，**全部与源文件精确匹配**
- P0 数据错配数：**0**

---

## 4. 引用/图文件完整性核查

### 4.1 引用完整性
| 检查项 | 结果 |
|--------|------|
| tex cite keys 总数 | 36 |
| bib 条目总数 | 73 |
| cite key 在 bib 中缺失 | 0 ✅ |
| bib 中未被引用条目 | 37（P2-3 未关闭） |

### 4.2 图文件完整性
| 文件名 | 存在 |
|--------|------|
| fig0_aeris_workflow.pdf | ✅ |
| fig1_env_pdr_panel.pdf | ✅ |
| fig2_ablation_panel.pdf | ✅ |
| fig3_scalability_panel.pdf | ✅ |
| fig4_power_sensitivity_maps.pdf | ✅ |
| fig5_power_sensitivity_absolute.pdf | ✅ |
| fig6_patch_control_delta.pdf | ✅ |
| fig7_tradeoff_panel.pdf | ✅ |
| fig8_ns3_trend_panel.pdf | ✅ |

9/9 图文件全部存在，编号 fig0→fig8 连续无缺口。

### 4.3 交叉引用完整性
- `\label{}` 定义：22 个
- 每个 label 均有对应 `\ref{}` 引用：22/22 ✅
- 未被引用的 label：0
- 未定义的引用：0

### 4.4 TODO/FIXME 残留扫描
- 全文搜索 TODO/FIXME/XXX/HACK/PLACEHOLDER：**0 命中** ✅

---

## 5. v67 新风险扫描

v66→v67 的唯一变更是集成用户定稿的 AERIS 流程图（fig0_aeris_workflow.pdf）。文本和数据均未改动。

| # | 风险点 | 级别 | 说明 |
|---|--------|------|------|
| N1 | 流程图 PDF 是否可正常嵌入 | 低 | fig0_aeris_workflow.pdf 存在且 v67 编译成功，无 fatal error。需人工确认 PDF 中图片渲染质量 |
| N2 | 无新文本/数据风险 | — | v67 相对 v66 仅换图，无新增文字或数值变更 |

**v67 未引入新的 P0/P1/P2 问题。**

---

## 6. 综合门控判定

### 6.1 v64→v67 问题追踪总表

| 维度 | v64 原始 P0/P1/P2 | v67 已关闭 | v67 残留 |
|------|-------------------|-----------|----------|
| 数据一致性 | 0/0/0 | — | 0/0/0 |
| 引用/DOI | 0/0/2 (P2-2,P2-3) | 0 | 0/0/2 |
| 方法严谨性 | 0/2/1 (P1-2,P1-3,H5) | 2P1 关闭 | 0/0/1 |
| 图文一致性 | 0/0/2 (P2-4,P2-6) | 2 关闭 | 0/0/0 |
| 版式/投稿 | 0/1/3 (P1-1,H3,H6,P2-1) | P1-1+P2-1+P2-4 关闭 | 0/0/2 |
| **总计** | **0/3/10** | **3P1+5P2** | **0/0/5** |

### 6.2 残留 5 个 P2 明细

| # | 问题 | 修复难度 | 是否阻塞投稿 |
|---|------|----------|-------------|
| P2-2 | Kandris2020 bib key 与 year=2009 不一致 | 低（改 key 或加 note） | 否 |
| P2-3 | bib 中 37 个未引用条目 | 低（脚本清理） | 否（BibTeX 不输出未引用条目） |
| P2-7 | env_sensitivity JSON 缺 collision/relay flags 元数据 | 低（生成 patched 文件） | 否（不影响论文内容） |
| P2-8 | 13 表全在正文，未移至补充材料 | 中（需作者判断） | 否（编辑可能建议精简） |
| P2-9 | Discussion 中 3 个 `\subsection*` 未编号 | 低（改为 `\subsection` 或 `\paragraph`） | 否（MDPI 模板可能接受） |

### 6.3 门控结论

| 指标 | 值 |
|------|-----|
| P0 | 0 |
| P1 | 0（全部已关闭） |
| P2 残留 | 5（均为格式/元数据类，不影响科学结论） |
| 数据错配 | 0/92 单元格 |
| 引用缺失 | 0/36 cite keys |
| 图文件缺失 | 0/9 |
| 交叉引用缺失 | 0/22 labels |

**综合判定：Accept with Minor Revision**

判定理由：
- v64 的 3 个 P1 全部在 v66/v67 中关闭，核心方法论弱点已修补
- 数据层零错配（92/92 单元格精确匹配源文件）
- 引用/图文件/交叉引用全部完整
- 残留 5 个 P2 均为格式/元数据类，不影响科学结论，不阻塞投稿

---

## 7. 建议修复路径（投稿前可选）

### 快速修复（<1h，建议执行）
1. **P2-3**: 清理 bib 中 37 个未引用条目（脚本化对比 tex cite keys vs bib keys，删除差集）
2. **P2-9**: Discussion 中 3 个 `\subsection*` 改为 `\paragraph{}`（MDPI 模板更友好）

### 可选修复（不阻塞投稿）
3. **P2-2**: 将 bib key `Kandris2020` 改为 `Kandris2009`，同步更新 tex 中的 `\cite{}`
4. **P2-7**: 为 env_sensitivity JSON 生成 patched 文件，补充 collision/relay flags 元数据
5. **P2-8**: 评估是否将 Table 5 (rigor patch pilot) 和 Table 8 (PEGASIS snapshot) 移至补充材料

---

## 8. 版本对比总结

| 版本 | P0 | P1 | P2 | 判定 |
|------|----|----|-----|------|
| v64 (两轮合并) | 0 | 3 | 10 | Minor Revision（偏 Major 边界） |
| v67 (本轮) | 0 | 0 | 5 | **Accept with Minor Revision** |

v64→v67 净关闭：3 个 P1 + 5 个 P2。残留 5 个 P2 全部为非阻塞性格式/元数据问题。

---

*报告结束*

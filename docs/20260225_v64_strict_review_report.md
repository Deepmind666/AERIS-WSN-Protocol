# v64 严格复审报告
# 审稿人：Claude (Sensors/MDPI 视角)
# 日期：2026-02-25
# 稿件：AERIS_Sensors_MDPI_Submission_Draft_20260225_v64.tex

---

## 1. 文件读取清单

| # | 文件 | 状态 |
|---|------|------|
| 1 | `.claude/RULES.md` | ✅ |
| 2 | `.codex/RULES.md` | ✅ |
| 3 | `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260225_v64.tex` | ✅ |
| 4 | `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260225_v64.pdf` | ⚠️ 未单独可视审阅（无 PDF 渲染工具），基于 tex 源码审阅 |
| 5 | `for_submission/bibliography.bib` | ✅ |
| 6 | `docs/20260225_v64_change_log.md` | ✅ |
| 7 | `docs/20260225_v64_table1_source_check.md` | ✅ |
| 8 | `docs/20260224_v62_cited_doi_validation.csv` | ✅ |
| 9 | `docs/20260224_v63_workflow_section3_consistency_checklist.md` | ✅ |
| 10 | `results/mega_experiments/scalability_4env_v50rigor_20260222_descriptive.csv` | ✅ |
| 11 | `results/mega_experiments/scalability_4env_v50rigor_20260222_significance.csv` | ✅ |
| 12 | `results/mega_experiments/env_sensitivity_20260207_205317.json` | ✅ (通过 table1_source_check 间接验证) |

---

## 2. 本次完成

1. Table 1 (legacy 100-node) 20/20 数值逐值核对 — 全部匹配
2. Table 3 (primary 1000-node) 20/20 数值逐值核对 — 全部匹配
3. Table 4 (significance snapshot) 4/4 行核对 (diff, g, p_holm) — 全部匹配
4. 引用完整性核对：36 cite keys vs bib，缺失=0
5. DOI 可解析性核对：36/36 条目 HTTP 200 OK
6. v64 新修5项逐项核实 — 全部已关闭
7. 图文件存在性核对：9/9 PDF 文件存在
8. 交叉引用完整性核对：22 labels，21 有 \ref 引用，1 未引用
9. 摘要/结论术语一致性核对 — 通过

---

## 3. 仍需核对

1. PDF 可视审阅（图表渲染质量、字体嵌入、页面溢出）— 需人工用 PDF 阅读器检查
2. Table 2 (ablation gateway) 数值溯源 — 源 JSON 路径未在白名单文件中明确标注，建议补充
3. Table 5-8 (rigor patch, patch-control, tx-power, matched delta) 的逐值源文件核对 — 本次仅核对了核心 Table 1/3/4，其余表格建议后续补充核对

---

## 4. 问题总表（P0/P1/P2）

| # | 问题 | 严重级别 | 文件路径:行号 | 证据摘要 | 修复建议 |
|---|------|----------|--------------|----------|----------|
| 1 | `tab:cas_gateway_coeffs` 未被正文 `\ref` 引用 | P2 | v64.tex:85 | label 定义存在但全文无 `\ref{tab:cas_gateway_coeffs}` | 在 Section 3 首次提及系数表处添加 `Table~\ref{tab:cas_gateway_coeffs}` |
| 2 | `Kandris2020` bib 条目年份标注为 2009 但 cite key 含 2020 | P2 | bibliography.bib:101-111 | `year={2009}` 但 key 为 `Kandris2020`；DOI 验证文件也标注 `year=2009` | 将 key 改为 `Kandris2009` 或在 bib 中添加 note 说明引用年份差异 |
| 3 | bib 中 37 个未被引用的条目 | P2 | bibliography.bib | 73 个 bib 条目中仅 36 个被 tex 引用 | 投稿前清理未引用条目，减少编辑审查负担 |
| 4 | Figure 编号跳过 fig4 | P2 | v64.tex:432 | fig0→fig1→fig2→fig3→fig5，缺少 fig4 | 建议重新编号为连续序列，或在文中说明跳号原因 |
| 5 | Table 1 caption 标注 "legacy" 但未显式说明 collision/relay flags 状态 | P1 | v64.tex:223 | caption 写 "strict collision/relay flags disabled" 但正文第242行才解释 | 建议在 caption 中直接标注 "(collision/relay disabled)" |
| 6 | Eq.(2) Gateway scoring 的 $\delta$ 系数为负值 (-0.60) 但未解释物理含义 | P2 | v64.tex:102 | $\delta(\hat{D})=-0.60$ 意味着距离越远得分越低，但未在正文中解释 | 在系数表后添加一句解释负系数的物理含义 |

---

## 5. 三张摘要表

### 5.1 数据一致性核对表

| 表格 | 核对范围 | 源文件 | 匹配数/总数 | 结果 |
|------|----------|--------|-------------|------|
| Table 1 (tab:pdr100) | 20 cells (4 env × 5 proto, mean±std) | env_sensitivity_20260207_205317.json | 20/20 | ✅ 通过 |
| Table 3 (tab:scale1000) | 20 cells (4 env × 5 proto, mean) | scalability_4env_v50rigor_20260222_descriptive.csv | 20/20 | ✅ 通过 |
| Table 4 (tab:robust_snapshot) | 4 rows (diff, g, p_holm) | scalability_4env_v50rigor_20260222_significance.csv | 4/4 | ✅ 通过 |

P0 数据错配数：0

### 5.2 引用/DOI 核对表

| 检查项 | 结果 | 详情 |
|--------|------|------|
| tex cite keys 总数 | 36 | — |
| bib 条目总数 | 73 | — |
| cite key 在 bib 中缺失 | 0 | ✅ 全部命中 |
| DOI 可解析性 (HTTP 200) | 36/36 | ✅ 全部通过 |
| bib 中未被引用条目 | 37 | P2：建议投稿前清理 |
| key-year 不一致 | 1 | P2：Kandris2020 实际 year=2009 |

P0 引用缺失数：0

### 5.3 图文一致性核对表

| 检查项 | 结果 | 详情 |
|--------|------|------|
| Figure 1 workflow 与 Section 3 语义一致 | ✅ | CAS 三模式、first-hit 顺序、输入要素均匹配 (v63 checklist 已确认) |
| 图文件全部使用 *_s62.pdf 后缀 | ✅ | 9/9 文件统一后缀 |
| 图文件全部存在于 figures/ 目录 | ✅ | 9/9 PDF 文件存在 |
| 交叉引用完整性 (label→ref) | ⚠️ | 22 labels 中 21 有 ref，`tab:cas_gateway_coeffs` 未被引用 (P2) |
| Figure 编号连续性 | ⚠️ | fig0→fig1→fig2→fig3→fig5，跳过 fig4 (P2) |

P1 图文问题数：0

---

## 6. 可投门控结论

| 维度 | P0 | P1 | P2 |
|------|----|----|-----|
| 数据一致性 | 0 | 0 | 0 |
| 引用/DOI | 0 | 0 | 2 |
| 方法严谨性 | 0 | 0 | 1 |
| 图文一致性 | 0 | 0 | 2 |
| 版式/投稿 | 0 | 1 | 1 |
| **合计** | **0** | **1** | **6** |

判定规则：
- P0 = 0 → 未触发 Reject 门槛
- P1 = 1 → 在 ≤2 阈值内
- 综合判定：**Minor Revision (Accept with minor fixes)**

触发理由：无 P0 阻塞项；唯一 P1 是 Table 1 caption 中 collision/relay 状态标注位置不够显眼（信息已在正文中存在，仅需前移至 caption）。6 个 P2 均为格式/风格类改进，不影响科学结论。

---

## 7. 最小修复路径

### 24h 内可完成（推荐投稿前修复）
1. 在 Section 3 系数表首次出现处添加 `Table~\ref{tab:cas_gateway_coeffs}` 引用（P2 #1）
2. Table 1 caption 末尾追加 "(collision/relay flags disabled)"（P1 #5）
3. 清理 bib 中 37 个未引用条目（P2 #3）

### 72h 内可完成（建议但非阻塞）
4. 统一 Kandris key 命名或添加 note（P2 #2）
5. 重新编号 fig4→fig5 序列或添加跳号说明（P2 #4）
6. 在系数表后补充 $\delta=-0.60$ 的物理含义解释（P2 #6）

---

## 8. 给 Codex 的执行建议清单

1. **立即可执行**：修复 P1 #5（Table 1 caption 补充 flags 状态），修复 P2 #1（添加系数表交叉引用）— 均为 tex 单行编辑
2. **批量清理**：用脚本对比 tex cite keys 与 bib keys，删除未引用的 37 个 bib 条目
3. **不建议执行**：不需要重跑任何实验，不需要修改 src/，不需要改数据口径
4. **后续压力测试**：建议对 Table 2 (ablation) 和 Table 5-8 (stress/sensitivity) 做同等精度的源文件逐值核对，作为第二轮审计补充
5. **PDF 可视审阅**：建议人工打开 v64.pdf 检查图表渲染质量、页面溢出、字体嵌入

---

*报告结束*

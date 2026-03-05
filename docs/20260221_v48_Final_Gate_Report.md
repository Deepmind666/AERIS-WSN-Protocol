# AERIS v48 最终拒稿视角复核 (2026-02-21)

> 审稿模式: 仅挑可拒稿项 (P0/P1)，不提润色
> 审稿文件: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260221_v48.tex` (550 lines)
> 图表版本: S45 (12 figures, 12 tables)
> 基准版本: v47 审稿报告 (`docs/20260221_v47_Strict_Review_Report.md`)

---

## Verdict: Accept (conditional on P1-V48-1 micro-fix)

---

## v47 P1 修复验证

| v47 Finding | v48 状态 |
|---|---|
| P1-V47-1: 21/24 图表无交叉引用 | **CLOSED** — 自动扫描 24/24 labels 全部有 `\ref{}` |

---

## 自动化扫描结果

| 检查项 | 结果 |
|---|---|
| 交叉引用覆盖 | 24/24 OK |
| 禁写断言扫描 | 0 命中 |
| 摘要词数 | 165 词 (≤200) |
| 文献完整性 | 17/17 cite keys in bib |
| NS3 数据一致性 | 8/8 OK (vs `ns3_scale_ext_1000_stats.csv`) |
| S8 数据一致性 | 与 v47 一致 (v47 已验证 20/20 OK) |
| 图文件存在性 | 12/12 s45 PDF 存在 |
| 总行数 | 550 (v47: 518, +32 lines from cross-ref sentences) |

---

## P0 (阻断级): 0 项

无数据伪造。无禁写断言。无统计方法滥用。

## P1 (可拒稿级): 1 项

**P1-V48-1: L354 跨节引用错位**
- 证据: L354 写 `Table~\ref{tab:s10_aeris_1000} reports the 1000-node AERIS tx-power snapshot used in this section.`
- 但 L354 位于 S9 PEGASIS 表 (L338-352) 之后、S10 subsection 标题 (L356) 之前
- 这句引用了一个尚未出现的 S10 表，且 "this section" 指代不明（此时读者仍在 S9）
- 风险: 审稿人会标记为 "disorganized presentation"，不会直接拒稿但会要求 major revision
- 最小修复: 将 L354 整行移至 L357 之后（S10 subsection 内部），或删除该句（L390 已有引用覆盖 fig:s10）

---

## Gate 判定

**P0: 0 | P1: 1 (micro-fix) | 可拒稿风险: 极低**

P1-V48-1 是一句话的位置错误，修复耗时 < 30 秒。修复后即可锁版投稿。

### 修复后是否需要重新编译？

是。移动或删除 L354 后需重新 pdflatex+bibtex。

### 是否需要补实验？

不需要。

---

*报告生成: Claude 4.6 (Opus), 2026-02-21*
*审稿文件: v48.tex (550 lines, 12 fig, 12 tab)*
*扫描: xref 24/24, 禁写 0, bib 17/17, NS3 8/8, abstract 165w*

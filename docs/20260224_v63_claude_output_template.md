# AERIS v63 审稿输出模板（固定结构）

> **使用要求**：逐节填写，不得删减任何区块。每条问题必须给证据路径与行号。
> 缺少任何区块 = 报告不合格。

---

## 1) 文件读取清单

| 序号 | 文件 | 读取状态 |
|:----:|------|:--------:|
| 1 | `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260224_v63.tex` | ✅/❌ |
| 2 | `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260224_v63.pdf` | ✅/❌ |
| 3 | `for_submission/bibliography.bib` | ✅/❌ |
| 4 | `docs/20260224_v62_cited_doi_validation.csv` | ✅/❌ |
| 5 | `docs/20260224_v63_workflow_section3_consistency_checklist.md` | ✅/❌ |
| 6 | `docs/20260224_v63_change_log.md` | ✅/❌ |
| 7 | `results/mega_experiments/scalability_4env_v50rigor_20260222_descriptive.csv` | ✅/❌ |
| 8 | `results/mega_experiments/scalability_4env_v50rigor_20260222_significance.csv` | ✅/❌ |

---

## 2) 本次完成

1. 数据-文本一致性核对（Table 1/3/4 vs 源 CSV/JSON）
2. cite-bib 完整性核对（被引键 vs bibliography.bib）
3. DOI 可解析性核对（被引键 vs v62_cited_doi_validation.csv）
4. Workflow 与 Section 3 对齐核对
5. 图版本（s62）一致性核对
6. 摘要结论边界核对
7. 门控结论与评分

---

## 3) 仍需核对

- [ ] （如无，写"无"）

---

## 4) 问题总表（P0/P1/P2）

> P0 必须优先列出。每条问题必须包含全部 5 个字段，缺一不可。

| # | 问题 | 严重级别 | 文件路径:行号 | 证据摘要 | 修复建议 |
|:-:|------|:--------:|---------------|----------|----------|
| 1 | （示例：Table 3 indoor_office AERIS PDR 与源 CSV 不一致） | P0 | `v63.tex:245` vs `descriptive.csv:row12` | 论文值 0.9847，源文件值 0.9842 | 更正为源文件值 |
| 2 | （示例：摘要未区分 legacy 与 primary 口径） | P1 | `v63.tex:31` | 文本直接合并两种矩阵结论 | 在摘要增加 legacy 限定语 |
| ... | ... | ... | ... | ... | ... |

> 若无问题，写"未发现 P0/P1/P2 问题"。

**问题统计**：P0 = ___ 条，P1 = ___ 条，P2 = ___ 条

---

## 5) 三张摘要表（强制）

### 5.1 数据一致性核对表

> 至少覆盖 Table 1/3/4 的关键数值。每行必须同时给出论文值和源文件值。

| 表格 | 项目 | 论文值 | 源文件值 | 一致性 | 证据路径 |
|------|------|--------|----------|:------:|----------|
| Table 3 | indoor_office AERIS PDR (n=1000) | | | ✅/❌ | `v63.tex:行号` vs `descriptive.csv:行号` |
| Table 3 | outdoor_suburban AERIS PDR (n=1000) | | | ✅/❌ | |
| Table 4 | AERIS vs PEGASIS delta | | | ✅/❌ | `v63.tex:行号` vs `significance.csv:行号` |
| Table 4 | Hedges' g | | | ✅/❌ | |
| Table 1 | （legacy 100 节点关键值） | | | ✅/❌ | |
| ... | ... | ... | ... | ... | ... |

**小结**：___ / ___ 项一致

### 5.2 引用/DOI 核对表

| 指标 | 数值/状态 | 证据路径 |
|------|-----------|----------|
| tex 被引键总数 | | `v63.tex` 全文 `\cite{}` 提取 |
| bib 缺失键数 | | `bibliography.bib` 交叉比对 |
| DOI 校验通过数 | | `v62_cited_doi_validation.csv` |
| DOI 异常数 | | `v62_cited_doi_validation.csv` |
| 近 5 年引用比例 (2022+) | | `bibliography.bib` year 字段统计 |

**小结**：缺失键 = ___，DOI 异常 = ___

### 5.3 图文一致性核对表

| 检查项 | 状态 | 证据路径 |
|--------|:----:|----------|
| Workflow 图与 Section 3 语义对齐 | ✅/❌ | `v63_workflow_section3_consistency_checklist.md` |
| CAS 三模式（Direct/Chain/Two-hop）在图中体现 | ✅/❌ | `v63.tex:Section 3` vs workflow 图 |
| 级联顺序匹配 Section 3 文本 | ✅/❌ | |
| 正文图引用全部为 `*_s62.*` | ✅/❌ | `v63.tex` 全文 `\includegraphics` 搜索 |
| Figure 1 文件存在且可读 | ✅/❌ | `for_submission/figures/fig0_aeris_workflow_20260224_s62.pdf` |

**小结**：___ / 5 项通过

---

## 6) 可投门控结论

- **结论**：`Reject / Major / Minor / Accept`
- **总分（100）**：___
- **P0/P1/P2 计数**：P0 = ___，P1 = ___，P2 = ___

**触发原因**：
1. （列出决定评级的关键因素，引用检查项编号）
2. ...

**维度得分明细**：

| 维度 | 权重 | 得分 | 扣分原因 |
|------|-----:|-----:|----------|
| 数据一致性与可复现性 | 30 | | |
| 方法与逻辑严谨性 | 20 | | |
| 图表质量与图文一致 | 15 | | |
| 统计与显著性表达 | 10 | | |
| 引用与文献规范 | 10 | | |
| 写作清晰与结构 | 10 | | |
| 投稿合规性 | 5 | | |
| **合计** | **100** | | |

---

## 7) 最小修复路径（24h / 72h）

> 仅在结论为 Reject/Major/Minor 时填写。Accept 时写"无需修复"。

### 24h 必修（阻塞投稿项）
1. ...
2. ...

### 72h 完整收口
1. ...
2. ...

---

## 8) 给 Codex 的执行建议清单

> 按优先级排列，每条建议必须可直接执行（给出文件路径和具体操作）。

1. ...
2. ...
3. ...

---

## 9) 审稿纪律自检（必须回答）

| 自检项 | 回答 |
|--------|:----:|
| 是否存在无证据断言 | 是/否 |
| 是否每条问题均绑定 `路径:行号` | 是/否 |
| 是否在有 P0 时避免了 Accept | 是/否 |
| 是否包含全部 9 个区块 | 是/否 |
| 是否包含全部 3 张摘要表 | 是/否 |

> 若任何自检项为"否"，报告自动降级为"不合格"，需说明原因并补充。

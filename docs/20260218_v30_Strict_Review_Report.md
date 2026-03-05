# AERIS v30 严格审稿报告

> 审稿日期：2026-02-18
> 审稿对象：`AERIS_Sensors_MDPI_Submission_Draft_20260217_v30.tex`
> 审稿依据：`20260218_Claude_Strict_Reviewer_Prompt_v1.md`
> 审稿人：Claude Opus 4.6（四角色模拟）

---

## 1) 总体结论

- 当前稿件总体建议：**Major Revision**
- 一句话理由：S9 表格数值与实际数据文件存在不可追溯的偏差（P0），NS3 CLAIM_GATE 文件中 26/28 与论文 25/28 不一致（P1），S8 非物理上升趋势的讨论虽已标注但仍缺乏根因解释（P1），整体叙事防御性重复过多导致可读性下降（P1）。

---

## 2) 四位审稿人意见

### R1 方法学保守派

- **结论：Major Revision**

三条最关键问题：
1. **S8 非物理 PDR 上升趋势未给出根因解释**（tex:197）。论文承认 indoor_factory/outdoor_urban/outdoor_suburban 中 AERIS PDR 随节点数增加而上升，但仅标注为"known S8 baseline limitation"，未解释物理机制。审稿人会追问：是路由算法 bug、信道模型缺陷、还是聚类密度效应？缺乏根因分析会被视为方法学缺陷。
2. **S9 表格（tab:s9_patch_control）数值来源不可追溯**。indoor_factory 100 节点 patch=0.9543 与实际 S9 patch 文件（`scalability_indoor_factory_local_s9_20260216_023118.json`）中 mean=0.9281 不符；outdoor_urban 500 节点 patch=0.4067 与实际文件 mean=0.7299 严重偏离。这意味着 S9 表格引用了一个未在文件清单中出现的旧数据源。
3. **PEGASIS 在 S11 中 patch=control（delta=0.000）**。indoor_factory 全部 6 个节点规模下 PEGASIS patch 与 control 完全相同（t=0, p=1.0），这在物理上不合理——开启 mac-collision 和 multihop-relay 不应对任何协议完全无影响。这暗示 PEGASIS 实现可能未正确接入碰撞/中继模块。

两条优点：
1. 证据分层设计（S8/S9/S10/S11/NS3）是 WSN 仿真论文中罕见的严谨做法。
2. 明确声明"不池化跨 regime 数据"，避免了常见的统计误用。

必改项（P0）：
- 修复 S9 表格数值来源，确保每个数值可追溯到具体 JSON 文件。
- 解释 PEGASIS S11 delta=0 的原因（是否未接入碰撞模块）。

次改项（P1）：
- 为 S8 非物理上升趋势提供至少一段根因假设。

---

### R2 统计学严格派

- **结论：Minor Revision**

三条最关键问题：
1. **Hedges' g 绝对值过大（最高 336.7）的解释不充分**。tex:219 提到"should be interpreted as magnitude indicators"，但 g>50 在任何学科都是异常值。应明确说明这是 n=1000 大样本 + 极低组内方差的数学产物，而非真实效应量的物理含义。
2. **S9 样本量不对称（patch n=1000 vs control n=600）的处理**。虽然使用了 Welch t-test，但论文未讨论不等样本量对效应量估计的影响。S11 已修复此问题，但 S9 表格仍保留在论文中，应加注释说明 S11 是 S9 的样本量匹配版本。
3. **S10 中 59/60 显著的表述过于简洁**。唯一不显著的 LEACH indoor_office 1000 节点应给出具体数值（delta、p 值），否则审稿人无法判断是边界不显著还是完全无差异。

两条优点：
1. Holm-Bonferroni 校正在所有比较族中一致使用。
2. 明确声明"大样本下联合解释 p 值、效应量和绝对 delta"。

必改项（P0）：
- 无。

次改项（P1）：
- 为 Hedges' g > 50 的情况增加一句方法学脚注。
- S10 唯一不显著单元给出具体数值。

---

### R3 工程复现派

- **结论：Major Revision**

三条最关键问题：
1. **S9 表格数据来源断裂**（P0）。tex:260-271 中 indoor_factory 和 outdoor_urban 的 patch 值无法从任何已知 S9 JSON 文件中复现。具体偏差：
   - indoor_factory n=100: tex=0.9543, 文件=0.9281（差 0.026）
   - indoor_factory n=500: tex=0.8413, 文件=0.9060（差 0.065）
   - outdoor_urban n=100: tex=0.7943, 文件=0.7482（差 0.046）
   - outdoor_urban n=500: tex=0.4067, 文件=0.7299（差 0.323）
   outdoor_urban n=500 的偏差高达 0.323，这不是四舍五入误差，而是完全不同的数据源。
2. **NS3 CLAIM_GATE.md 与论文不一致**。CLAIM_GATE.md 第 17 行写"26/28 统计显著"，但实际数据和论文均为 25/28。CLAIM_GATE 是门控文件，必须与数据一致。
3. **bibliography.bib 未审查**。提示词要求抽查核心引用的 DOI 真实性，但 bib 文件未在本次交付清单中列出为"已更新"。

两条优点：
1. 每个实验 JSON 文件包含 git_commit、timestamp、run_tier 等溯源元数据。
2. provenance sidecar 机制（.provenance.json）为每个结果文件提供了审计链。

必改项（P0）：
- S9 表格必须重新生成，从明确的 JSON 文件中提取数值。
- CLAIM_GATE.md 中 26/28 → 25/28。

次改项（P1）：
- 在论文 Data Availability 段落中列出关键 JSON 文件名。

---

### R4 应用价值导向派

- **结论：Minor Revision**

三条最关键问题：
1. **防御性重复表述过多**。"bounded to the tested baseline set"、"regime-specific"、"intentionally not claimed" 等限定语在摘要、结果、讨论、结论中反复出现超过 15 次。这种写法虽然学术上安全，但严重降低可读性，审稿人可能认为作者对自己的结果缺乏信心。
2. **缺乏与 SOTA 的定量对比**。论文仅与 LEACH/PEGASIS/HEED/TEEN 比较，这些都是 2000-2004 年的经典协议。缺少与近 5 年 adaptive/learning-based 方法的定量对比（Related Work 仅定性引用了 Ren2024、Okine2024）。
3. **实际部署场景缺失**。论文声称"lightweight hierarchical protocol"但未给出计算复杂度分析、内存占用、或在真实硬件（如 TelosB、CC2530）上的可行性讨论。

两条优点：
1. 环境分类（4 类信道）贴近实际部署场景。
2. 部署指导表（tab:deployment_summary）提供了可操作的工程建议。

必改项（P0）：
- 无。

次改项（P1）：
- 精简防御性重复表述，每个限定语只在首次出现时完整表述，后续引用缩写。
- 在 Related Work 或 Discussion 中增加一段与近年方法的定性对比。

---

## 3) 问题清单（按严重度排序）

| ID | 严重度 | 文件:行号 | 问题描述 | 证据文件 | 修复建议 |
|----|--------|-----------|----------|----------|----------|
| 1 | P0 | v30.tex:260-271 | S9 表格 indoor_factory/outdoor_urban patch 值无法从已知 JSON 复现（最大偏差 0.323） | `scalability_indoor_factory_local_s9_20260216_023118.json`, `scalability_outdoor_urban_local_s9_20260216_023118.json` | 从明确的 JSON 文件重新提取 S9 表格数值，或标注数据来源文件名 |
| 2 | P0 | v30.tex:276 / S11 significance CSV | PEGASIS 在 indoor_factory S11 中 patch=control（delta=0.000, t=0, p=1.0），6/6 节点规模全部如此 | `s11_matched_4env_patch_vs_control_20260217_significance.csv` 行 44-49 | 检查 PEGASIS 实现是否正确接入 mac-collision/multihop-relay 模块；若确认未接入，在论文中明确说明 |
| 3 | P1 | `NS3_CLAIM_GATE.md`:17 | 写"26/28 统计显著"，实际数据为 25/28（3 个不显著：indoor_office n=100/200/1000） | `ns3_scale_ext_1000_significance.csv` | 修改为 25/28 |
| 4 | P1 | v30.tex:197 | S8 非物理 PDR 上升趋势仅标注为"known limitation"，未给出根因假设 | S8 descriptive CSV | 增加一段根因分析（如：无 MAC 碰撞建模导致密集部署下信道利用率被高估） |
| 5 | P1 | v30.tex:219 | Hedges' g 最高达 336.7，仅说"magnitude indicators"，解释不充分 | `s11_matched_4env_patch_vs_control_20260217_significance.csv` | 增加脚注：大样本+低方差的数学产物，不应跨研究比较 |
| 6 | P1 | v30.tex:31 | 摘要过长（约 250 词），包含过多实验细节（S9/S10/S11 具体数字） | — | 压缩至 200 词以内，将 S9/S10/S11 细节移至正文 |
| 7 | P1 | v30.tex:全文 | 防御性限定语重复 >15 次（"bounded to", "intentionally not claimed", "regime-specific"） | — | 首次完整表述，后续用缩写或引用首次定义 |
| 8 | P1 | v30.tex:250 | S9 节未说明 S11 已替代 S9 的样本量匹配问题 | — | 在 S9 节末尾加一句："S11 provides the matched-sample confirmation of this block." |
| 9 | P2 | v30.tex:52 | Related Work 缺少与近 5 年 adaptive/learning-based WSN 方法的定量对比 | — | 增加 1-2 句定性对比说明为何选择经典 baseline |
| 10 | P2 | v30.tex:423 | Data Availability 未列出关键 JSON 文件名 | — | 列出 S8/S9/S10/S11 核心 JSON 文件名 |

---

## 4) 决策邮件模板

### Reject 邮件

尊敬的作者：

经审稿专家组评审，您的稿件"AERIS: Environment-Aware Hierarchical Routing for Reliable Wireless Sensor Networks under Realistic Channel Conditions"未达到本刊发表标准，决定退稿。

主要理由：
1. S9 实验表格中多个关键数值无法从提交的数据文件中复现，存在数据溯源断裂。
2. PEGASIS 在 S11 碰撞/中继测试中 patch 与 control 完全相同（delta=0.000），暗示实验实现存在缺陷。
3. S8 基线矩阵中 PDR 随节点数增加而上升的非物理趋势未给出根因解释。

建议作者在修复上述问题后重新投稿。

### Major Revision 邮件

尊敬的作者：

您的稿件经审稿专家组评审，建议大修后重新提交。

稿件在证据分层设计和统计方法上展现了较高的学术严谨性，但存在以下必须修复的问题：
1. **[P0]** S9 表格（Table 5）中 indoor_factory 和 outdoor_urban 的 patch 数值无法从已知 JSON 文件复现，最大偏差达 0.323。请从明确的数据文件重新提取并标注来源。
2. **[P0]** PEGASIS 在 S11 indoor_factory 中 patch=control（delta=0.000），请检查 PEGASIS 实现是否正确接入碰撞模块，并在论文中说明。
3. **[P1]** NS3 CLAIM_GATE.md 中 26/28 与实际 25/28 不一致，请修正。
4. **[P1]** 摘要过长，防御性限定语重复过多，请精简。

请在修改稿中逐条回复上述意见。

### Minor Revision 邮件

尊敬的作者：

您的稿件经审稿专家组评审，建议小修后接收。

稿件整体质量较好，证据分层和统计处理规范。请修复以下问题：
1. 确认 S9 表格数值来源并补充溯源标注。
2. 精简摘要和正文中的重复限定语。
3. 补充 PEGASIS S11 delta=0 的简要说明。
4. NS3 CLAIM_GATE.md 中 26/28 → 25/28。

### Accept 邮件

尊敬的作者：

您的稿件"AERIS: Environment-Aware Hierarchical Routing..."经审稿专家组评审，达到本刊发表标准，决定接收。

稿件在 WSN 可靠性路由领域提供了系统性的多环境评估证据，证据分层设计和统计方法规范，结论边界清晰。建议在校样阶段确认所有表格数值与数据文件的一致性。

---

## 5) 导师视角修改意见

### 3 天内必须完成

1. **修复 S9 表格数据来源**。逐行核对 tab:s9_patch_control 中每个数值，标注其来源 JSON 文件名。如果来源文件已丢失，用当前可追溯的 S9 patch 文件重新生成表格。这是投稿的硬性前提。
2. **调查 PEGASIS S11 delta=0 问题**。在 `src/baseline_protocols/pegasis_protocol.py` 中检查 mac-collision 和 multihop-relay 的接入点。如果 PEGASIS 确实未接入这两个模块，在论文 S11 节加一句说明："PEGASIS uses a fixed chain topology that bypasses the collision and relay modules; its near-zero deltas are therefore expected."
3. **修正 NS3 CLAIM_GATE.md**。将 26/28 改为 25/28，与实际数据和论文一致。

### 7 天内可完成

4. **精简摘要**。当前约 250 词，目标 ≤200 词。将 S9/S10/S11 的具体数字（"59/60 cells"、"24/24 cells"）移至正文，摘要只保留结论性表述。
5. **压缩防御性重复**。全文搜索 "bounded to"、"intentionally not claimed"、"regime-specific"，每个限定语只在首次出现时完整表述，后续用 "as noted in Section X" 替代。
6. **为 S8 非物理趋势增加根因假设**。在 tex:197 附近增加一段："We hypothesize that the upward PDR trend under S8 arises from the absence of MAC-layer contention modeling: as node density increases, the simulator allows concurrent transmissions without collision penalties, artificially inflating delivery rates."
7. **Hedges' g 脚注**。在 Statistical Methods 节增加："When both sample size and between-group separation are large while within-group variance is small, Hedges' g can exceed conventional benchmarks by orders of magnitude; such values reflect the specific experimental design rather than generalizable effect magnitudes."

### 14 天进阶任务

8. **补充 PEGASIS 碰撞模块接入**。如果确认 PEGASIS 未接入碰撞模块，考虑在 v31 中补跑一次 PEGASIS-only 的碰撞测试，或在 Limitations 中明确标注。
9. **Related Work 扩展**。增加 1-2 段与近年 adaptive/learning-based WSN 方法的定性对比，说明选择经典 baseline 的理由（可复现性、公平性、社区认可度）。
10. **语言润色**。请母语审校者通读全文，重点检查摘要和结论的表述流畅度。

---

## 6) 最终门控判定

- 可提交给老师阶段汇报：**是**（S11 数据完整、后处理通过、图表已生成，整体进展可汇报）
- 可直接投稿 Sensors：**否**

最小补强路径（5 条）：

1. 修复 S9 表格数值来源（P0，预计 1-2 小时）
2. 调查并说明 PEGASIS S11 delta=0 原因（P0，预计 1 小时）
3. 修正 CLAIM_GATE.md 26/28 → 25/28（P1，5 分钟）
4. 精简摘要至 ≤200 词（P1，30 分钟）
5. 为 S8 非物理趋势增加根因假设段落（P1，30 分钟）

---

## 附录：数据交叉验证结果

### S11 表格 vs 数据文件（全部通过）

| 环境 | 节点 | tex delta | 数据 delta | 状态 |
|------|------|-----------|------------|------|
| indoor_office | 100 | -0.0234 | -0.023356 | OK |
| indoor_office | 1000 | -0.3120 | -0.311986 | OK |
| indoor_factory | 100 | -0.0052 | -0.005170 | OK |
| indoor_factory | 1000 | -0.2435 | -0.243527 | OK |
| outdoor_urban | 100 | -0.0077 | -0.007700 | OK |
| outdoor_urban | 1000 | -0.7474 | -0.747410 | OK |
| outdoor_suburban | 100 | -0.0191 | -0.019126 | OK |
| outdoor_suburban | 1000 | -0.2616 | -0.261626 | OK |

### S9 表格 vs 数据文件（4 处不匹配）

| 环境 | 节点 | tex patch | 文件 patch | 偏差 | 状态 |
|------|------|-----------|------------|------|------|
| indoor_factory | 100 | 0.9543 | 0.9281 | 0.026 | MISMATCH |
| indoor_factory | 500 | 0.8413 | 0.9060 | 0.065 | MISMATCH |
| outdoor_urban | 100 | 0.7943 | 0.7482 | 0.046 | MISMATCH |
| outdoor_urban | 500 | 0.4067 | 0.7299 | 0.323 | MISMATCH |

### NS3 显著性计数

| 来源 | 数值 | 状态 |
|------|------|------|
| 实际数据 | 25/28 | 基准 |
| v30 论文 | 25/28 | OK |
| CLAIM_GATE.md | 26/28 | MISMATCH |

### PEGASIS S11 异常

| 环境 | 全部 6 节点规模 delta | t-stat | p-value | 状态 |
|------|----------------------|--------|---------|------|
| indoor_factory | 0.000000 | 0.0 | 1.0 | 需调查 |
| outdoor_urban | 近零（-0.005~+0.001） | 近零 | ~1.0 | 需调查 |

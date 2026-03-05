# AERIS v31 严格审稿报告

> 审稿日期：2026-02-19
> 审稿对象：`AERIS_Sensors_MDPI_Submission_Draft_20260218_v31.tex`
> 审稿依据：`20260218_Claude_Strict_Review_Prompt_v31.md`
> 审稿人：Claude Opus 4.6（四角色模拟）

---

## 0) 强制核验项（逐条回答）

### 1. S9 表格可复现性
- 抽查 12 行（AERIS patch，4 环境 × 3 节点数）
- 对照文件：`s9_matched_4env_patch_vs_control_20260216_merged.csv`
- 结果：**12/12 全部匹配**（diff=0.0000）
- v30 中 4 处 MISMATCH 已全部修正

### 2. S11 PEGASIS 异常解释
- tex:276 写道："PEGASIS shows exact zero deltas in indoor_factory (all six node counts), which is treated as an implementation-coupling anomaly pending further code-path audit rather than as physical invariance evidence."
- 判定：**诚实且不误导**。明确标注为实现耦合异常，不当作物理结论。
- 代码审计（`20260218_PEGASIS_S11_ZeroDelta_Audit.md`）给出了三点根因：`pegasis_chain_exempt=True`、`uplink_factor(1)=1.0`、未接入 multihop relay。

### 3. 摘要长度与信息密度
- 实测词数：**172 词**（≤200 目标达成）
- v30 为 ~255 词，压缩了 83 词
- 摘要中仍保留了 S9/S10/S11 的关键结论性表述，但已移除具体数字细节
- 判定：**通过**

### 4. S8 非物理趋势处理
- tex:197 写道："A plausible cause is that the original S8 simulator path omits explicit MAC-layer contention penalties, so increasing node density can artificially increase effective forwarding opportunities instead of inducing collision loss."
- 判定：**合格**。给出了合理根因假设，并明确 S8 为 frozen baseline regime。

### 5. NS-3 边界是否越界
- tex:338 写道："NS-3 validation is used as a cross-platform trend check rather than a numerical-equivalence proof."
- tex:356 写道："25/28 AERIS-versus-LEACH comparisons are significant; the non-significant cells are indoor_office at node counts 100, 200, and 1000."
- 判定：**未越界**。严格保持 trend-level，未出现 numerical equivalence 暗示。

### 6. 图表质量
- 5 张图均为 PDF 矢量格式，配色一致（低饱和科学色板）
- fig3 scalability panel 的 indoor_office 子图使用了窄 y 轴窗口（图注已说明）
- fig5 S11 patch-control delta 图为新增，双面板布局清晰
- 判定：**合格**，无误导性视觉缩放

### 7. 引用真实性风险

| 引用 | 标题关键词 | 风险等级 | 理由 |
|------|-----------|---------|------|
| Heinzelman2000LEACH | Energy-Efficient Communication Protocol | 低 | WSN 经典论文，IEEE HICSS |
| Lindsey2002PEGASIS | Power-Efficient Gathering | 低 | IEEE Aerospace Conference 经典 |
| Younis2004HEED | Hybrid Energy-Efficient Distributed | 低 | IEEE TPDS 经典 |
| Manjeshwar2001TEEN | Routing Protocol for Enhanced Efficiency | 低 | IEEE ICDCS Workshop 经典 |
| Akyildiz2002Survey | A Survey on Sensor Networks | 低 | IEEE Comm Magazine 高引 |
| Kandris2020 | Power Conservation through Energy Efficient Routing | 低 | Sensors MDPI |
| Ren2024 | MeFi: Mean Field RL for Cooperative Routing | 中 | 2024 年新文献，无法确认 DOI |
| Okine2024 | Multi-Agent DRL for Packet Routing | 中 | 2024 年新文献，无法确认 DOI |
| Bhukya2025Hybrid | Hybrid routing | 中 | 2025 年新文献，无法确认 DOI |
| Dan2024EMRAR | EMRAR | 中 | 2024 年新文献，无法确认 DOI |

- 经典引用（7 篇）：风险低
- 近年引用（4 篇 Ren2024/Okine2024/Bhukya2025/Dan2024）：**证据不足**，无法在本地验证 DOI 真实性，建议作者逐一核实

---

## A) Findings（按严重度排序）

| ID | 严重度 | 问题描述 | 证据 | 修复建议 |
|----|--------|----------|------|----------|
| 1 | P1 | NS3 CLAIM_GATE.md 第 16 行仍写"26 个统计显著"，与实际 25/28 不一致；第 18 行推荐措辞已正确写 25/28 | `NS3_CLAIM_GATE.md`:16 vs `:18` | 将第 16 行的"26"改为"25"，同时将第 17 行"2 个不显著"改为"3 个不显著"并补充 n=200 |
| 2 | P1 | 防御性限定语仍有重复："bounded to" 出现 5 次，"intentionally not" 出现 3 次 | v31.tex 全文搜索 | 首次完整表述后，后续改为"as noted in Section X"或删除 |
| 3 | P1 | S9 表格（tab:s9_patch_control）仅展示 AERIS 的 patch/control，但 tex:276 讨论了"PEGASIS 在所有 24 个 PEGASIS cells 中近零 delta"——读者无法从表格验证此声明 | v31.tex:252-274 vs :276 | 要么在 S9 表格中增加 PEGASIS 行，要么在文中标注数据来源文件名 |
| 4 | P1 | Hedges' g 解释（tex:219）虽已改善，但仍缺少 v30 审稿建议的方法学脚注 | v31.tex:219 | 在 Statistical Methods 节增加脚注："When both sample size and between-group separation are large while within-group variance is small, Hedges' g can exceed conventional benchmarks by orders of magnitude." |
| 5 | P1 | S9 节（tex:249-276）未说明 S11 是 S9 的样本量匹配版本 | v31.tex:276 末尾 | 加一句："S11 provides the matched-sample confirmation of this block (n=1000 in both arms)." |
| 6 | P2 | S10 唯一不显著单元（LEACH indoor_office 1000 nodes）未给出具体数值 | v31.tex:297 | 补充 delta、p 值、g 值 |
| 7 | P2 | Data Availability（tex:423）未列出关键 JSON 文件名，仅给出分组描述 | v31.tex:423 | 至少列出 S8/S9/S11 核心 CSV 文件名 |
| 8 | P2 | 近年引用（Ren2024/Okine2024/Bhukya2025/Dan2024）DOI 真实性未验证 | bibliography.bib | 作者逐一核实 DOI 并确认可访问 |

---

## B) 四角色结论

### R1 方法学保守派 — Minor Revision

3 条核心理由：
1. **S8 非物理趋势已给出根因假设（tex:197），但未提供定量佐证**。仅说"omits explicit MAC-layer contention penalties"是定性假设，缺少一个简单的数值论证（如：在 n=1000 时理论碰撞概率估算）。不过这不构成 P0，因为 S8 已被明确降级为 frozen baseline。
2. **PEGASIS S11 delta=0 的处理是诚实的**（tex:276），代码审计给出了三点根因。但论文正文未引用审计文档的具体发现（`pegasis_chain_exempt=True`），审稿人可能追问"为什么是零"而得不到足够细节。
3. **S9 表格数据溯源已修复**（12/12 匹配），v30 的 P0 问题已关闭。

必改项（P0）：无。
次改项（P1）：在 S9/S11 讨论段落中补充 PEGASIS 豁免机制的一句话说明。

---

### R2 统计学严格派 — Minor Revision

3 条核心理由：
1. **Hedges' g 方法学脚注仍缺失**（tex:219）。v30 审稿明确建议在 Statistical Methods 节增加脚注解释大样本+低方差下 g 值膨胀的数学原因。v31 仅在 tex:219 写了"magnitude indicators within this experiment design"，未达到脚注级别的方法学说明。
2. **S9 样本量不对称（patch n=1000 vs control n=600）的讨论已在 tex:250 提及**，但未明确说明 S11 是其样本量匹配版本。建议在 S9 节末尾加一句衔接。
3. **S10 唯一不显著单元未给出具体数值**（tex:297 仅说"LEACH at indoor_office, 1000 nodes"）。审稿人需要 delta/p/g 才能判断是边界不显著还是完全无差异。

必改项（P0）：无。
次改项（P1）：Hedges' g 脚注、S9→S11 衔接句。

---

### R3 工程复现派 — Minor Revision

3 条核心理由：
1. **CLAIM_GATE.md 第 16 行仍写"26 个统计显著"**。v31 fix log 声称"NS3_CLAIM_GATE.md currently already reports 25/28"，但实际第 16 行仍为 26。第 18 行推荐措辞确实写了 25/28，造成文件内部自相矛盾。这是一个遗漏修复。
2. **S9 表格仅展示 AERIS**，但正文讨论了 PEGASIS 的 24 个 cells 结果。读者无法从论文内部验证 PEGASIS 声明。建议至少标注数据来源文件名。
3. **Data Availability 段落（tex:423）改为分组描述后，可读性提升**，但仍未列出具体文件名。对于强调可复现性的论文，这是一个遗憾。

必改项（P0）：无（CLAIM_GATE 是辅助文件，不在论文正文中）。
次改项（P1）：修正 CLAIM_GATE 第 16-17 行。

---

### R4 应用价值导向派 — Accept (with minor notes)

3 条核心理由：
1. **防御性限定语已有改善但仍偏多**。"bounded to" 5 次、"intentionally not" 3 次。v30 有 >15 次，v31 降至 ~10 次，方向正确但仍可进一步精简。
2. **部署指导表（tab:deployment_summary）实用性好**，给出了环境分类的工程建议和主要注意事项。
3. **Related Work 已补充近年方法引用**（Ren2024/Okine2024/Chen2023Survey/ElFouly2023/Dan2024/Bhukya2025），并说明了选择经典 baseline 的理由（"rule-based protocol with explicit reproducibility constraints"）。

必改项（P0）：无。
次改项（P1）：继续精简防御性重复。

---

## C) 决策邮件（中文）

### Reject 邮件

尊敬的作者：

经审稿专家组评审，您的稿件未达到本刊发表标准，决定退稿。

主要理由：
1. S8 基线矩阵中 PDR 随节点数增加而上升的非物理趋势虽已给出假设，但缺乏定量验证。
2. PEGASIS 在 S11 碰撞/中继测试中 patch 与 control 完全相同，暗示实验框架对部分协议无效。
3. 仅与 2000-2004 年经典协议比较，缺少与近年方法的定量对比。

建议作者在修复上述问题后重新投稿。

### Major Revision 邮件

尊敬的作者：

您的稿件经审稿专家组评审，建议大修后重新提交。

稿件在证据分层设计和统计方法上展现了较高的学术严谨性，但存在以下需修复的问题：
1. **[P1]** NS3 CLAIM_GATE.md 内部自相矛盾（第 16 行 26/28 vs 第 18 行 25/28），请统一修正。
2. **[P1]** S9 表格仅展示 AERIS，但正文讨论了 PEGASIS 结果，读者无法验证。
3. **[P1]** Hedges' g 方法学脚注仍缺失，大样本下 g>100 的解释不充分。
4. **[P1]** 防御性限定语仍有约 10 处重复，影响可读性。

请在修改稿中逐条回复上述意见。

### Minor Revision 邮件

尊敬的作者：

您的稿件经审稿专家组评审，建议小修后接收。

稿件整体质量较好，v30 中的 P0 问题（S9 表格数据溯源断裂、PEGASIS delta=0 解释缺失）已在 v31 中修复。请修复以下问题：
1. 修正 CLAIM_GATE.md 第 16 行 26→25。
2. 在 Statistical Methods 节增加 Hedges' g 方法学脚注。
3. 在 S9 节末尾加一句 S11 衔接说明。
4. 精简剩余防御性重复表述。

### Accept 邮件

尊敬的作者：

您的稿件经审稿专家组评审，达到本刊发表标准，决定接收。

稿件在 WSN 可靠性路由领域提供了系统性的多环境评估证据，证据分层设计规范，S9 数据溯源问题已修复，PEGASIS 异常已诚实标注。建议在校样阶段精简防御性限定语并确认近年引用的 DOI 可访问性。

---

## D) 导师视角修订路线图

### 3 天内必须完成

1. **修正 CLAIM_GATE.md 第 16-17 行**。将"26 个统计显著"改为"25"，将"2 个不显著"改为"3 个不显著（indoor_office n=100, 200, 1000）"。（5 分钟）
2. **Hedges' g 方法学脚注**。在 Statistical Methods 节（tex:118 附近）增加："When both sample size and between-group separation are large while within-group variance is small, Hedges' g can exceed conventional benchmarks by orders of magnitude; such values reflect the specific experimental design rather than generalizable effect magnitudes."（15 分钟）
3. **S9→S11 衔接句**。在 tex:276 末尾加："S11 provides the matched-sample confirmation of this block with n=1000 in both arms; see Section 5.7."（5 分钟）

### 7 天内建议完成

4. **精简防御性重复**。全文搜索 "bounded to"（5 次）、"intentionally not"（3 次），每个限定语只在首次出现时完整表述，后续用 "as noted in Section X" 替代。目标：总计 ≤5 次。（30 分钟）
5. **S9 表格补充 PEGASIS 行或标注数据来源**。在 tab:s9_patch_control 下方加脚注："Full five-protocol S9 data are available in `s9_matched_4env_patch_vs_control_20260216_merged.csv`."（10 分钟）
6. **S10 不显著单元补充数值**。在 tex:297 补充 LEACH indoor_office 1000 nodes 的 delta、p、g 值。（10 分钟）
7. **核实近年引用 DOI**。逐一检查 Ren2024/Okine2024/Bhukya2025/Dan2024/ElFouly2023/Chen2023Survey 的 DOI 可访问性。（30 分钟）

### 14 天增强项

8. **PEGASIS 碰撞豁免验证实验**。按 `20260218_Claude_PEGASIS_Sanity_Prompt.md` 跑 PEGASIS-only 小矩阵（`pegasis_chain_exempt=False`），验证关闭豁免后是否出现显著负 delta。
9. **语言润色**。请母语审校者通读全文，重点检查摘要和结论的表述流畅度。
10. **Data Availability 补充文件名**。列出 S8/S9/S11 核心 CSV 文件名。

---

## E) 最终门控判定

- 是否可用于老师阶段汇报：**是**
- 是否可直接投稿 Sensors：**是（条件性）**

与 v30 对比：v30 有 2 个 P0（S9 数据溯源断裂、PEGASIS delta=0 未解释），判定为"不可投稿"。v31 已修复全部 P0，降级为 0 个 P0 + 5 个 P1 + 3 个 P2。

**条件性投稿的前提**（3 项，均可在 1 小时内完成）：
1. 修正 CLAIM_GATE.md 第 16-17 行（虽非论文正文，但属于审计链文件）
2. 增加 Hedges' g 方法学脚注
3. 增加 S9→S11 衔接句

完成以上 3 项后，v31 可作为 v32 提交。

---

## 附录：v30→v31 修复验证

| v30 问题 | 严重度 | v31 状态 | 验证方法 |
|----------|--------|----------|----------|
| S9 表格 4 处 MISMATCH | P0 | **已修复** | 12/12 行与 CSV 精确匹配 |
| PEGASIS S11 delta=0 未解释 | P0 | **已修复** | tex:276 标注为实现耦合异常 |
| CLAIM_GATE 26/28 | P1 | **部分修复** | 第 18 行已改为 25/28，第 16 行仍为 26 |
| 摘要过长（~255 词） | P1 | **已修复** | 实测 172 词 |
| S8 非物理趋势无根因 | P1 | **已修复** | tex:197 给出 MAC contention 假设 |
| Hedges' g 脚注缺失 | P1 | **未修复** | tex:219 仅有简短说明，未达脚注级别 |
| 防御性重复 >15 次 | P1 | **部分改善** | 降至 ~10 次，仍可进一步精简 |
| S9→S11 衔接缺失 | P1 | **未修复** | S9 节末尾无衔接句 |

---

文件清单（本次审查涉及）：
- `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260218_v31.tex`（主稿件，428 行）
- `docs/20260218_v31_MajorFix_Log.md`（修订日志）
- `docs/20260218_PEGASIS_S11_ZeroDelta_Audit.md`（PEGASIS 代码审计）
- `docs/20260218_Claude_Strict_Review_Prompt_v31.md`（审稿提示词）
- `results/mega_experiments/s9_matched_4env_patch_vs_control_20260216_merged.csv`（S9 数据）
- `results/mega_experiments/s11_matched_4env_patch_vs_control_20260217_delta.csv`（S11 delta）
- `results/mega_experiments/s11_matched_4env_patch_vs_control_20260217_significance.csv`（S11 显著性）
- `ns3_validation/results/NS3_CLAIM_GATE.md`（NS3 门控文件）
- `for_submission/bibliography.bib`（参考文献）
- `docs/20260219_v31_Strict_Review_Report.md`（本报告）

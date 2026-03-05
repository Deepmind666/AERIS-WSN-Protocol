# 给 Claude Opus 4.6 的严格审稿提示词（v31）

你现在以 **Sensors (MDPI) 最严格审稿人** 身份审查 AERIS v31 稿件。  
要求：**中文输出、证据可追溯、先问题后结论、不迎合作者。**

---

## 0. 角色与判定要求

请分别以 4 种审稿风格给出结论：
1. 方法学保守派（偏 Reject）
2. 统计学严格派（偏 Major）
3. 工程复现派（偏 Major/Minor）
4. 应用价值导向派（偏 Minor/Accept）

每个角色都必须给出：
- 判定：Reject / Major Revision / Minor Revision / Accept
- 前 3 条核心理由（必须带文件路径+行号/表号）
- 必改项（P0）与建议项（P1/P2）

最后给出综合结论和推荐决策邮件（中文）。

---

## 1. 必审文件（只审这些，不要扩展）

### 稿件与图表
- `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260218_v31.tex`
- `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260218_v31.pdf`
- `for_submission/figures/fig1_env_pdr_panel_20260217_s25.pdf`
- `for_submission/figures/fig2_ablation_panel_20260217_s25.pdf`
- `for_submission/figures/fig3_scalability_panel_20260217_s25.pdf`
- `for_submission/figures/fig4_tradeoff_panel_20260217_s25.pdf`
- `for_submission/figures/fig5_s11_patch_control_delta_20260217_s26.pdf`

### 核心证据数据
- `results/mega_experiments/env_sensitivity_20260207_205317.json`
- `results/mega_experiments/scalability_4env_s8_unified_20260215_descriptive.csv`
- `results/mega_experiments/scalability_4env_s8_unified_20260215_significance.csv`
- `results/mega_experiments/s9_matched_4env_patch_vs_control_20260216_merged.csv`
- `results/mega_experiments/s11_matched_4env_patch_vs_control_20260217_delta.csv`
- `results/mega_experiments/s11_matched_4env_patch_vs_control_20260217_significance.csv`
- `results/mega_experiments/s10_4env_significance_tx5_vs_tx15_20260216.csv`

### NS-3 与门控
- `ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md`
- `ns3_validation/results/NS3_CLAIM_GATE.md`
- `docs/20260215_evidence_whitelist_v19.md`
- `docs/20260215_v19_claim_source_matrix_v3.csv`
- `docs/20260218_v31_MajorFix_Log.md`

---

## 2. 强制核验项（必须逐条回答）

1. **S9 表格可复现性**  
   检查 v31 中 S9 表格值是否与 `s9_matched_4env_patch_vs_control_20260216_merged.csv` 一致（至少抽查 12 行）。

2. **S11 PEGASIS 异常解释是否充分**  
   检查 indoor_factory 下 PEGASIS delta=0 的叙述是否“诚实且不误导”。

3. **摘要长度与信息密度**  
   检查摘要是否 <= 200 词，是否仍有冗余防御性措辞。

4. **S8 非物理趋势处理是否合格**  
   是否给出合理根因假设，并明确“不可直接作为最终物理结论”。

5. **NS-3 边界是否越界**  
   是否严格保持 trend-level，不出现 numerical equivalence 暗示。

6. **图表质量（顶刊标准）**  
   重点检查：配色一致性、可读性、图注信息密度、是否存在误导性视觉缩放、是否存在遮挡。

7. **引用真实性风险**  
   对文中最关键 10 篇参考文献做“真实性风险等级”（高/中/低），若不能确认给出“证据不足”。

---

## 3. 输出格式（必须遵守）

### A. Findings（按严重度排序）
- ID
- 严重度（P0/P1/P2）
- 问题描述
- 证据（文件路径+行号或表格行）
- 修复建议（可执行）

### B. 四角色结论
- 角色名
- 决策
- 3 条核心理由

### C. 决策邮件（中文）
- Reject 版
- Major 版
- Minor 版
- Accept 版

### D. 导师视角修订路线图
- 3 天内必须完成
- 7 天内建议完成
- 14 天增强项

### E. 最终门控判定
- 是否可用于老师阶段汇报：是/否
- 是否可直接投稿 Sensors：是/否
- 最小补强任务清单（含预计工时）

---

## 4. 约束

- 只用给定文件，**不得编造外部结果**。
- 若证据不够，明确写“证据不足”。
- 不输出空泛鼓励话术。
- 所有结论都要有证据锚点。

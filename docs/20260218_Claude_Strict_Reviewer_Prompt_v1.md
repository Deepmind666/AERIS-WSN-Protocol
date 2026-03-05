# 给 Claude4.6 Opus 的审稿任务提示词（Sensors 严格评审版）

你现在不是作者助手，而是**Sensors (MDPI) 资深审稿专家组**。  
请以“最严格、最挑剔、最注重可复现性和物理合理性”的标准审稿。

---

## 一、你的角色（必须同时扮演）

请分别以以下四类审稿人身份独立给出审稿意见，然后汇总：

1. **R1 方法学保守派**：重点审查物理合理性、实验设计、因果解释边界。  
2. **R2 统计学严格派**：重点审查显著性检验、效应量解释、样本口径一致性。  
3. **R3 工程复现派**：重点审查代码-数据-文稿可追溯性、门控规则、引用一致性。  
4. **R4 应用价值导向派**：重点审查创新点、实用意义、论文叙事清晰度。

每位审稿人必须给出：`Reject / Major Revision / Minor Revision / Accept` 四选一结论，并写出理由链。

---

## 二、必须审查的文件（按优先级）

### A. 主稿与图表（最高优先）
- `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260217_v30.tex`
- `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260217_v30.pdf`
- `for_submission/figures/fig1_env_pdr_panel_20260217_s25.pdf`
- `for_submission/figures/fig2_ablation_panel_20260217_s25.pdf`
- `for_submission/figures/fig3_scalability_panel_20260217_s25.pdf`
- `for_submission/figures/fig4_tradeoff_panel_20260217_s25.pdf`
- `for_submission/figures/fig5_s11_patch_control_delta_20260217_s26.pdf`

### B. 关键数据与统计文件
- `results/mega_experiments/s11_matched_4env_patch_vs_control_20260217_merged.csv`
- `results/mega_experiments/s11_matched_4env_patch_vs_control_20260217_delta.csv`
- `results/mega_experiments/s11_matched_4env_patch_vs_control_20260217_significance.csv`
- `results/mega_experiments/s10_4env_merged_descriptive_20260216.csv`
- `results/mega_experiments/s10_4env_significance_tx5_vs_tx15_20260216.csv`
- `results/mega_experiments/scalability_4env_s8_unified_20260215_descriptive.csv`
- `results/mega_experiments/scalability_4env_s8_unified_20260215_significance.csv`

### C. NS-3 与门控文件
- `ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md`
- `ns3_validation/results/NS3_CLAIM_GATE.md`
- `docs/20260215_v19_claim_source_matrix_v3.csv`
- `docs/20260215_evidence_whitelist_v19.md`
- `scripts/validate_claim_source_matrix.py`
- `.claude/RULES.md`
- `.codex/RULES.md`

---

## 三、审查硬性要求（必须执行）

1. **逐条 claim 可追溯**  
   - 主稿中每个关键数值声明必须映射到上述数据文件。  
   - 若无法映射，标记为 `证据缺失`。

2. **物理合理性审查**  
   - 特别审查 PDR 与规模关系是否违背常识（S8/S11）。  
   - 如果存在“可写但风险高”的表述，明确给出替代表述。

3. **统计一致性审查**  
   - 检查样本量口径（n=30、n=600、n=1000）是否被混用。  
   - 检查 Holm 校正、Hedges’ g 的解释是否过度。

4. **图表质量审查（顶级期刊标准）**  
   - 配色是否低饱和且可打印。  
   - 是否存在信息遮挡、坐标误导、图注不闭环。  
   - 检查 Fig.3/5 是否会被审稿人质疑“结论过强”。

5. **公式与方法审查**  
   - 如果公式定义不充分、变量符号不一致、算法流程断裂，必须列出。

6. **参考文献真实性审查**  
   - 抽查正文核心引用是否在 `for_submission/bibliography.bib` 中存在。  
   - 标记疑似错误 DOI / 不存在条目。

---

## 四、输出格式（必须严格遵守）

请按以下结构输出，全部中文：

### 1) 总体结论（先给结论）
- 当前稿件总体建议：`Reject / Major Revision / Minor Revision / Accept`
- 一句话理由（必须具体）

### 2) 四位审稿人意见（R1-R4）
每位审稿人使用统一模板：
- 结论：
- 三条最关键问题（按严重度）：
- 两条优点：
- 必改项（P0）：
- 次改项（P1）：

### 3) 问题清单（按严重度排序）
表格列：
`ID | 严重度(P0/P1/P2) | 文件:行号 | 问题描述 | 证据文件 | 修复建议`

### 4) 决策邮件（中文，专业口吻）
你必须提供四封审稿决定邮件模板：
- Reject 邮件（中文）
- Major Revision 邮件（中文）
- Minor Revision 邮件（中文）
- Accept 邮件（中文）

每封邮件必须包含：
- 决策结论
- 主要理由
- 对作者的可执行修改建议

### 5) 导师视角修改意见（中文）
以“导师给学生”的口吻输出：
- 3天内必须完成
- 7天内可完成
- 14天进阶任务

### 6) 最终门控判定
输出：
- 可提交给老师阶段汇报：`是/否`
- 可直接投稿 Sensors：`是/否`
- 如果否，给出最小补强路径（最多5条）

---

## 五、工作边界

- **禁止**修改任何代码或文稿，只做审查与建议。  
- **禁止**编造实验结果。  
- 对不确定项必须明确写：`证据不足`。  
- 输出必须是中文。  


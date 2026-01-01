# 1月1日 AERIS 项目深度评估（MDPI Sensors 视角）

日期: 2026-01-01
评估对象: C:\AERIS-WSN-Protocol
目标期刊: Sensors (MDPI)
主要论文: C:\AERIS-WSN-Protocol\for_submission\final_paper.tex
参考旧稿: C:\AERIS-WSN-Protocol\for_submission\final_paper.pdf

## 一、项目概览与研究价值（当前结论）
AERIS 的核心价值在于“真实环境校准 + MCU 级确定性路由”，与传统 LEACH/HEED/PEGASIS/TEEN 的理想化假设形成差异。结合 Intel Lab 环境日志与 CC2420 能耗模型，对所有协议统一建模与评估，是本项目真正有学术贡献的方向。该价值在 Sensors 中可成立，但前提是“数据一致、叙述克制、实验闭环”。

结论: 目前论文尚未达到可稳投 Sensors 的程度，主要问题是数据一致性、图表来源混乱、结果叙述与图表对不上、以及“过度宣称”。这些需要系统修复后才具备投稿基础。

## 二、已确认的核心创新（应保留并强化）
1) Trace-calibrated 信道与能耗建模: 使用 Intel Lab 环境数据驱动阴影衰落与能耗参数，避免自由空间假设。
2) MCU 级确定性协同栈: CAS/Skeleton/Gateway/Safety 组合，逻辑轻量、可解释、适合资源受限节点。
3) 可复现评估链路: 统一脚本 + JSON 指标 + 统计检验 + 图表生成，理论上具备复现闭环。

## 三、证据链状态与风险评估
### 1) 数据一致性风险（重大）
- 存在两个消融实验数据集: `results/intel_ablation.json` 与 `results/intel_ablation_parallel.json`，数值差异极大（详见 `DATA_CONSISTENCY_ISSUE.md`）。
- 统计验证文件曾使用错误数据源，导致效应量误报（g=10.09 vs g=4.48）。
- 当前状态文档之间互相矛盾:
  - `CURRENT_STATUS.md` (2024-12-31) 显示“Related Work 缺失/文献仅5篇”。
  - `PAPER_STATUS.md` (2026-01-01) 直接标记“Ready for Submission”。
  - `REVIEWER_CRITIQUE.md` 则指出图表编号混乱、数据来源不明。

结论: 数据和状态的单一事实源尚未建立，导致论文可信度不足。

### 2) 实验完整性风险（高）
- `EXPERIMENT_RIGOR_ANALYSIS.md` 明确指出：基线对比、可扩展性、长期稳定性实验缺失或不完整。
- 动态场景（corridor/moving/dropout）结果偏弱，AERIS 在单网关配置下明显落后，应当明确为“压力测试失败场景”，避免夸大。
- 大规模 300/500-node 结果在“是否启用可靠性 overlay”上存在叙述冲突，容易被审稿人质疑选择性呈现。

### 3) 图表资产风险（高）
- 图表文件在多个目录重复存在（results/plots、results/publication_figures、results/real_data_figures 等），难以确定“论文最终引用哪一套”。
- 多数图表缺少来源标注/样本量/统计标记，或者图例遮挡问题反复出现。
- `FIGURE_GENERATION_CHECKLIST.md` 明确禁止伪造，但当前图表链路尚未统一执行。

## 四、论文内容的关键问题（Sensors 视角）
1) Claims 过大: “优于经典协议”的叙述在动态/大规模条件下不成立，应转为“条件性优势 + 明确边界”。
2) 结果与图表不一致: Fig.11 与 Table 4 叙述冲突风险高，审稿人最敏感。
3) 文献真实度: 参考文献中存在 TODO 或 AI 痕迹，必须全部清理或标注待核验。
4) 语言风格偏 IEEE: 需要更少术语堆叠、更多可读性句式，符合 Sensors 风格。

## 五、当前可用的“可信证据”
- 消融实验（n=50×5=250）: `results/intel_ablation.json` 已验证（Gateway g≈4.48, Safety g≈3.48）。
- 参数敏感性（n=40×9=360）: `results/intel_sensitivity.json` 已验证。
- E0 环境相关性（n≈399,485）: `results/prior_experiments/e0_env_link_correlation.json` 已验证。

这些是现阶段最可靠的证据，应作为论文的“硬支撑”。

## 六、必须建立的“单一事实源”
1) 统一数据源: 明确只使用 `intel_ablation.json` 或彻底解释 `intel_ablation_parallel.json`。
2) 统一图表输出路径: 建议锁定 `results/plots/` 为论文唯一引用目录。
3) 统一论文入口: 以 `for_submission/final_paper.tex` 为唯一主稿，禁止并行版本。

## 七、短期修复优先级（必须完成）
1) 建立“结果-图表-文本一致性”清单: 对每张图标注其 JSON 来源、样本量、脚本路径。
2) 完成基线对比的可重复实验: 至少 4 协议 + AERIS-E/R，重复次数与图表一致。
3) 重写结果叙述: 动态场景明确为失败边界，大规模结果明确是否启用可靠性 overlay。
4) 清理参考文献: 去重、补全、删除 TODO 或标注核验。
5) 统一图表质量: 解决图例遮挡/标注缺失/标题位置错误等反复问题。

## 八、对项目的整体评价（当前阶段）
- 科研逻辑链存在，证据链正在形成，但尚未闭合。
- 代码和图表资产过多，重复与混乱严重，导致“看起来像未完成项目”。
- 若以 Sensors 审稿人视角，目前稿件仍可能被以“结果不一致/证据不足/图表混乱”拒稿。

## 九、下一阶段目标（我将持续记录于本文件）
- 目标: 形成“可复现 + 数据一致 + 叙述克制 + 图表干净”的 Sensors 版本。
- 策略: 用真实数据支撑最强贡献，弱化不稳场景，压低夸张 Claims。

## 十、重要信息备忘（防遗忘）
- 主稿: `for_submission/final_paper.tex`
- 图表主目录: `results/plots/`（建议确定为唯一来源）
- 数据一致性警告: `DATA_CONSISTENCY_ISSUE.md`
- 实验严谨性评估: `EXPERIMENT_RIGOR_ANALYSIS.md`
- 图表强制清单: `FIGURE_GENERATION_CHECKLIST.md`
- 科研逻辑链: `RESEARCH_LOGIC.md`

---

### 更新日志
- 2026-01-01: 创建本评估文档，完成项目深度评估与核心风险定位。

# Requirements Document

## Introduction

本规范旨在将AERIS（Adaptive Environment-aware Routing for IoT Sensors）项目从当前状态提升至MDPI Sensors期刊发表标准。核心目标是"能过审的Sensors水平"，而非"看起来像论文"。

基于深度批判性分析，本规范遵循两条硬约束：
1. **不动流程图**：流程图已定稿，只优化"实验数据相关图表 + 实验本身"
2. **指标/特征必须有先验证据 + 理论/统计支撑**：每个用于讲故事的指标，都要能回答"为什么选它、它和性能因果链的关系是什么、证据在哪里、统计上站不站得住"

科研故事采用"三段式证据链"：
- **机理/先验（Why）**：来自物理/协议/概率论的可解释假设
- **先验实验（Evidence-0）**：在主实验之前做"小而硬"的验证，让指标选择有根
- **统计验证（Evidence-1）**：主实验里把不确定性讲清楚（CI、效应量、多重比较校正）

## Glossary

- **AERIS**: Adaptive Environment-aware Routing for IoT Sensors，本项目提出的WSN路由协议
- **PDR**: Packet Delivery Ratio，数据包投递率，端到端可靠性核心指标
- **CAS**: Context-Adaptive Switching，上下文自适应切换模块
- **Gateway**: 网关节点，负责远距离簇头的两跳中继
- **Skeleton**: 骨干路由，基于PCA的高连通性节点选择
- **Cohen's d**: 效应量指标，衡量两组数据差异的实际显著性
- **先验实验**: 用于支撑指标/特征选择合理性的预备性实验
- **ETX**: Expected Transmission Count，预期传输次数，链路质量指标
- **PRR**: Packet Reception Ratio，数据包接收率
- **MDPI Sensors**: 目标期刊，IF=3.9，Q2级别
- **Bernoulli试验**: 链路传输成功/失败的概率模型
- **Beta-Binomial**: 用于safety阈值标定的贝叶斯后验分布
- **Cliff's δ / Hedges g**: 非参数/参数效应量指标
- **SHAP**: SHapley Additive exPlanations，特征重要性解释方法
- **Gini/Jain's fairness**: 负载均衡度量指标
- **plotenv**: 项目指定的绘图conda环境，避免NumPy崩溃

## Requirements

### Requirement 1: 先验实验E0 - 环境→链路解释力验证

**User Story:** As a 审稿人, I want 看到环境上下文（温度/湿度）确实能解释/预测链路退化, so that 我能相信环境感知路由的科学基础。

#### Acceptance Criteria

1. WHEN 分析Intel Lab trace数据, THE AERIS_System SHALL 计算humidity/temperature与PRR/ETX/pdr_hop_level的相关性（Pearson r或Spearman ρ），并报告p值。

2. WHEN 进行环境-链路关系分析, THE AERIS_System SHALL 使用滞后相关（cross-correlation）检测环境变化对链路质量的延迟影响。

3. WHEN 验证环境特征的预测能力, THE AERIS_System SHALL 使用回归或分类模型预测链路成功/失败，报告AUC和Brier score。

4. WHEN 确认统计显著性, THE AERIS_System SHALL 使用置换检验（permutation test）验证相关性非偶然。

5. IF 环境特征解释力弱（AUC < 0.6）, THEN THE AERIS_System SHALL 在论文中明确说明"环境作为可观测上下文的proxy，而非直接因果"。

### Requirement 2: 先验实验E1 - CAS特征贡献度验证

**User Story:** As a 审稿人, I want 看到CAS使用的特征（能量、距离、密度、公平性、链路质量、波动性）不是随便凑的, so that 我能相信特征选择的合理性。

#### Acceptance Criteria

1. WHEN 定义oracle mode, THE AERIS_System SHALL 在同一轮计算哪种模式带来更高的效用U = PDR - λ·Energy（λ可扫描）。

2. WHEN 分析特征贡献, THE AERIS_System SHALL 使用可解释模型（logistic regression / GAM）从特征预测oracle mode。

3. WHEN 报告特征重要性, THE AERIS_System SHALL 提供系数符号、显著性、以及permutation importance或SHAP值。

4. WHILE 进行特征分析, THE AERIS_System SHALL 确保模型可解释性强，便于写进论文。

5. IF 某特征系数不显著（p > 0.1）, THEN THE AERIS_System SHALL 考虑移除该特征或在论文中说明其辅助作用。

### Requirement 3: 先验实验E2 - Safety阈值/窗口概率论标定

**User Story:** As a 审稿人, I want 看到safety阈值θ和窗口T不是拍脑袋定的, so that 我能相信这是"风险控制"而非任意参数。

#### Acceptance Criteria

1. WHEN 设计safety机制, THE AERIS_System SHALL 将每轮delivery建模为Bernoulli试验，窗口内成功数服从Binomial分布。

2. WHEN 标定阈值, THE AERIS_System SHALL 使用Beta-Binomial得到后验P(p < θ | data)，仅当超过置信水平才触发safety。

3. WHEN 报告阈值选择, THE AERIS_System SHALL 提供误触发概率（false positive rate）的理论计算和实验验证。

4. WHILE 运行safety机制, THE AERIS_System SHALL 记录触发次数、触发条件、以及触发后的PDR变化。

5. IF 误触发率> 10%, THEN THE AERIS_System SHALL 调整阈值或窗口大小以控制风险。

### Requirement 4: 先验实验E3 - 负载不均衡/调度指标验证

**User Story:** As a 审稿人, I want 看到gateway/CH负载分布与可靠性/能耗的关系, so that 我能相信负载均衡机制的必要性。

#### Acceptance Criteria

1. WHEN 分析负载分布, THE AERIS_System SHALL 计算gateway/CH的负载Gini系数和Jain's fairness index。

2. WHEN 验证负载影响, THE AERIS_System SHALL 证明负载不均衡会带来可靠性下降/能耗上升（效应量+CI）。

3. WHEN 报告调度效果, THE AERIS_System SHALL 提供gateway并发、uplink抑制、CH负载分布、重传压力的统计。

4. WHILE 运行实验, THE AERIS_System SHALL 记录每轮的负载分布和对应的性能指标。

5. IF 负载均衡对性能无显著影响（Cohen's d < 0.2）, THEN THE AERIS_System SHALL 在论文中调整负载均衡的重要性声称。

### Requirement 5: 先验实验E4 - MCU-grade决策时延验证

**User Story:** As a 审稿人, I want 看到AERIS决策时延在MCU预算内, so that 我能相信"轻量级"的声称。

#### Acceptance Criteria

1. WHEN 测量决策时延, THE AERIS_System SHALL 提供ECDF分布图和随规模增长的scaling curve。

2. WHEN 声称MCU兼容, THE AERIS_System SHALL 证明决策时延< 25ms（TelosB级别）且内存< 50KB。

3. WHEN 对比ML/RL方法, THE AERIS_System SHALL 引用具体文献的计算开销数据（65-600ms）作为对照。

4. WHILE 运行实验, THE AERIS_System SHALL 使用benchmark_decision_time.json记录详细时延数据。

5. IF 决策时延随规模超线性增长, THEN THE AERIS_System SHALL 在论文中说明适用规模上限。

### Requirement 6: 扩展实验矩阵

**User Story:** As a 审稿人, I want 看到实验覆盖多场景、多规模、多负载, so that 我能相信结论的泛化性。

#### Acceptance Criteria

1. WHEN 设计实验矩阵, THE AERIS_System SHALL 覆盖以下场景：Intel replay（真实痕迹）、合成室内（uniform/corridor/cluster/obstacle-like）、动态压力（moving BS/dropout/phase shift）。

2. WHEN 设计实验矩阵, THE AERIS_System SHALL 覆盖以下规模：100/300/500/1000节点（每个点至少20-50 seeds）。

3. WHEN 设计实验矩阵, THE AERIS_System SHALL 覆盖至少3档发送率（低/中/高）+ 可选bursty负载。

4. WHILE 运行实验, THE AERIS_System SHALL 记录完整指标体系：可靠性（hop-level + e2e PDR、tail risk p05/p01）、能耗（total、per-delivered-packet、分布）、开销（控制包/重传/ARQ触发）、公平性（Jain/Gini）。

5. IF 某场景/规模组合无法完成, THEN THE AERIS_System SHALL 在论文中说明实验范围限制。

### Requirement 7: 大规模网络PDR问题诊断与修复

**User Story:** As a 研究者, I want 理解并修复300节点PDR仅2.9%的问题, so that AERIS能够声称支持大规模网络或明确其适用范围。

#### Acceptance Criteria

1. WHEN 运行大规模网络实验（>100节点）, THE AERIS_System SHALL 记录详细的跳数分布、网关负载、骨干使用率等诊断信息。

2. WHEN 诊断发现路径过长（平均跳数>4）, THE AERIS_System SHALL 自动调整网关数量（k_gw）和骨干节点数量（k_sk）以控制跳数。

3. IF 大规模网络PDR无法修复至>50%, THEN THE AERIS_System SHALL 在论文中明确声明"当前版本支持≤100节点网络"。

4. WHEN 进行大规模实验, THE AERIS_System SHALL 提供参数敏感性分析，展示k_gw、k_sk对PDR的影响曲线。

5. WHILE 运行任何规模的实验, THE AERIS_System SHALL 记录每跳PDR统计，用于验证PDR衰减模型。

### Requirement 8: 动态场景性能分析与定位调整

**User Story:** As a 论文作者, I want 诚实地报告动态场景的性能限制, so that 论文定位准确且不误导读者。

#### Acceptance Criteria

1. WHEN AERIS在动态场景（走廊移动、移动BS、随机失联）PDR低于基线50pp以上, THE AERIS_System SHALL 在论文中删除"面向动态环境优化"的声称。

2. WHEN 动态场景实验完成, THE AERIS_System SHALL 提供详细的失败模式分析，包括路径重建延迟、网关失效率等诊断指标。

3. IF 动态场景无法改善, THEN THE AERIS_System SHALL 将论文定位调整为"静态部署环境的轻量级协议"。

4. WHEN 报告动态场景结果, THE AERIS_System SHALL 使用Cohen's d效应量和Welch t检验量化与基线的差异。

5. WHILE 进行动态场景实验, THE AERIS_System SHALL 记录每阶段的拓扑变化率和AERIS响应延迟。

### Requirement 9: CAS模块效应量评估与重新定位

**User Story:** As a 审稿人, I want 看到CAS模块的真实贡献, so that 我能评估其作为"核心创新"的合理性。

#### Acceptance Criteria

1. WHEN CAS模块效应量（Cohen's d）< 0.5, THE AERIS_System SHALL 将论文重点从CAS转移至Gateway/Safety机制。

2. WHEN 进行消融实验, THE AERIS_System SHALL 报告所有模块的效应量，并按贡献度排序。

3. IF CAS在所有场景效应量均< 0.3, THEN THE AERIS_System SHALL 将CAS定位为"辅助模块"而非"核心创新"。

4. WHEN 报告CAS效果, THE AERIS_System SHALL 提供模式使用统计（DIRECT/CHAIN/TWO_HOP比例）和触发条件分析。

5. WHILE CAS模块运行, THE AERIS_System SHALL 记录每次模式切换的决策依据和环境特征值。

### Requirement 10: 统计严谨性与可重复性

**User Story:** As a 审稿人, I want 验证实验结果的统计显著性和可重复性, so that 我能信任论文的结论。

#### Acceptance Criteria

1. WHEN 报告任何性能对比, THE AERIS_System SHALL 提供Welch t检验p值、效应量（Cliff's δ或Hedges g）、95%置信区间（bootstrap/BCa）。

2. WHEN 进行多重比较, THE AERIS_System SHALL 使用Holm-Bonferroni校正控制FWER。

3. WHEN 发布实验结果, THE AERIS_System SHALL 提供完整的复现脚本、随机种子、配置文件。

4. WHILE 运行任何实验, THE AERIS_System SHALL 记录JSON格式的原始数据，支持独立验证。

5. IF 某对比p值> 0.05, THEN THE AERIS_System SHALL 在论文中明确说明"差异不显著"。

### Requirement 11: 图表质量与MDPI规范

**User Story:** As a 投稿者, I want 所有图表符合MDPI Sensors规范, so that 论文不会因格式问题被退回。

#### Acceptance Criteria

1. WHEN 生成图表, THE AERIS_System SHALL 确保宽度>=1200px（或等效pt），字体统一，颜色一致。

2. WHEN 导出SVG, THE AERIS_System SHALL 保留文字（svg.fonttype='none'），避免文字转路径。

3. WHEN 使用plotenv环境, THE AERIS_System SHALL 确保所有绘图脚本在该环境下可运行。

4. WHILE 生成图表, THE AERIS_System SHALL 运行validate_figures.py检查质量。

5. IF 图表不符合规范, THEN THE AERIS_System SHALL 修复后重新导出。

### Requirement 12: 代码质量与冗余清理

**User Story:** As a 开发者, I want 清理冗余代码并提升代码质量, so that 项目可维护且易于审稿人验证。

#### Acceptance Criteria

1. WHEN 清理代码, THE AERIS_System SHALL 移除所有未使用的函数、类、导入。

2. WHEN 重构代码, THE AERIS_System SHALL 确保核心协议文件（aeris_protocol.py）行数< 1000行。

3. WHEN 提交代码, THE AERIS_System SHALL 通过ruff/flake8静态检查，无E/W级别错误。

4. WHILE 维护代码, THE AERIS_System SHALL 保持README与实际功能同步。

5. IF 存在遗留路径（如EEHFR命名）, THEN THE AERIS_System SHALL 完成迁移或在文档中明确说明。

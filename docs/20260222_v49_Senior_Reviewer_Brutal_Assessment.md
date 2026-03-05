# AERIS v49 资深审稿人严苛评审报告

> 审稿人角色: 资深WSN领域审稿人，仅关注可拒稿级问题
> 审稿文件: `AERIS_Sensors_MDPI_Submission_Draft_20260221_v49.tex` (549 lines)
> 源代码审查: `src/aeris_protocol.py`, `src/benchmark_protocols.py`, `src/baseline_protocols/`, `src/cas_selector.py`, `src/gateway_selector.py`, `src/skeleton_selector.py`, `src/mac_collision_model.py`, `src/realistic_channel_model.py`
> 数据审查: `results/final_baseline_compare.json`, S8/S9/S10/S11 matrices
> 审稿日期: 2026-02-22
> 审稿标准: 不讨好作者，不回避问题，实事求是

---

## 总体判定: Major Revision (倾向 Reject)

本文在格式规范性和数据自洽性方面做得较好，但在**算法创新性、仿真方法学严谨性、Baseline公平性**三个核心维度存在严重问题。以下逐一展开。

---

## 一、算法创新性评估：不足以支撑独立论文

### 1.1 AERIS的"三组件"本质上是工程拼装，非算法创新

论文声称三个贡献组件：CAS (Context-Adaptive Switching)、Skeleton backbone、Gateway coordination。

**CAS (cas_selector.py)**：本质是一个6维线性加权打分器，对三种模式(DIRECT/CHAIN/TWO_HOP)分别计算加权分数，取最高分。权重通过"offline grid search"确定后固定。这不是"context-adaptive switching"——这是一个静态线性分类器。没有在线学习，没有上下文感知的动态调整（代码中的`set_stage_weights`虽然存在但论文未讨论）。与论文标题中"Environment-Aware"的暗示不符。

**Gateway (gateway_selector.py)**：本质是对CH按距离、中心性、链路质量的加权排序，选top-k。这是最基本的加权评分选择，任何WSN教科书都有类似方法。公式(2) `Score_i = αE_i + βC_i + γL_i` 是三项线性组合——这不构成方法论贡献。

**Skeleton (skeleton_selector.py)**：用PCA找主轴方向，选靠近主轴的CH作为backbone。这是一个简单的几何启发式，且论文消融实验(Table 2)显示Skeleton的贡献在所有环境中都不显著（论文自己也承认"CAS is not consistently positive across environments"）。

**核心问题**：三个组件都是已知技术的简单组合（线性加权 + PCA + 模糊逻辑），没有任何一个组件具有独立的算法贡献。论文没有提出新的理论框架、新的优化目标、或新的算法范式。这是一个**系统集成工作**，不是算法创新工作。

### 1.2 模糊逻辑系统是装饰性的

代码中(`aeris_protocol.py` L248-261)，当`skfuzzy`库不可用时，模糊逻辑系统退化为一个5项线性加权：`0.35*re + 0.25*ce + 0.15*nd + 0.10*db + 0.15*lq`。这意味着所谓的"模糊逻辑决策"在实际运行中可能就是简单加权求和。论文完全没有讨论这一点。

### 1.3 与现有工作的区分度不足

论文Related Work仅用一段话概括了"recent adaptive or learning-based methods"，没有与任何具体的环境自适应WSN协议进行详细对比。2020年后有大量基于强化学习、图神经网络的WSN路由工作，论文完全没有讨论AERIS相对于这些方法的定位。

---

## 二、仿真方法学：存在根本性缺陷

### 2.1 自研Python仿真器的可信度问题

这是本文最大的方法学风险。所有核心结论（100-node matrix、S8 scalability）都来自一个自研Python仿真器，而非NS-3、OMNeT++等经过同行验证的仿真平台。

**具体问题**：

**(a) 无MAC层建模（S8主矩阵）**

论文自己承认（L233）："the original S8 simulator path omits explicit MAC-layer contention penalties"。这意味着S8矩阵——论文的核心大规模证据——**完全没有MAC碰撞建模**。在WSN仿真中，忽略MAC碰撞等于假设所有节点可以同时无冲突地传输，这在物理上是不可能的。

论文试图用S9/S10/S11作为"calibration blocks"来弥补，但这些block的结论是**AERIS在patch模式下PDR显著下降**（S11: 24/24 cells negative and significant）。这实际上证明了：一旦加入更真实的物理模型，AERIS的优势大幅缩水。

**(b) PDR随节点数增加而上升——物理不合理**

论文自己承认（L233）："AERIS also shows a non-physical PDR-scale increase in three environments"。在真实WSN中，节点密度增加必然导致更多碰撞和干扰，PDR应该下降或至少不上升。PDR随节点数上升说明仿真器存在根本性建模错误。

论文将此归因于"omits explicit MAC-layer contention penalties"，但这不是一个可以轻描淡写的"limitation"——这是仿真器的根本性缺陷，直接影响所有大规模结论的可信度。

**(c) 信道模型过于简化**

`realistic_channel_model.py`使用log-normal shadowing模型，每次链路传输独立采样。没有时间相关性（temporal correlation）、没有多径衰落（multipath fading）、没有节点间干扰建模。对于声称"realistic channel conditions"的论文，这个信道模型远不够"realistic"。

### 2.2 NS-3验证的局限性被低估

NS-3验证仅覆盖AERIS vs LEACH（2个协议），且在indoor_office环境中多个节点数下不显著。论文将NS-3定位为"trend-level evidence"，但这实际上意味着：**论文无法证明其Python仿真器的结果在经过验证的仿真平台上可以复现**。

更关键的是：NS-3验证中AERIS的PDR（如indoor_office 0.9202）与Python仿真器的PDR（0.9739）差距明显。论文没有解释这个差距的来源。

---

## 三、Baseline比较公平性：存在结构性偏差

### 3.1 AERIS拥有Baseline不具备的架构优势

通过代码审查发现，AERIS相对于Baseline拥有以下**架构级**优势：

| 能力 | AERIS | LEACH | PEGASIS | HEED | TEEN |
|---|---|---|---|---|---|
| 多跳中继(Gateway/Skeleton) | ✓ | 后加(可选) | ✗ | ✗ | ✗ |
| 链路层重传(intra_link_retx) | ✓ (最多3次) | 后加(可选) | ✗ | ✗ | ✗ |
| 功率递增重传 | ✓ | 后加(可选) | ✗ | ✗ | ✗ |
| Safety fallback直传 | ✓ | ✗ | ✗ | ✗ | ✗ |
| 冗余上行链路 | ✓ | ✗ | ✗ | ✗ | ✗ |
| 自适应CH比例 | ✓ | ✗ | N/A | ✗ | ✗ |
| 邻居PRR/ETX追踪 | ✓ | ✗ | ✗ | ✗ | ✗ |

这意味着AERIS的PDR优势很大程度上来自**架构差异**（多跳+重传+冗余），而非**算法创新**。如果给LEACH/HEED/TEEN加上相同的多跳中继和链路层重传，PDR差距会大幅缩小。

### 3.2 Baseline实现存在可疑的性能抑制

**(a) benchmark_protocols.py中的LEACH有5%随机跳过数据传输**

`benchmark_protocols.py` L451: `self.data_transmission_probability = 0.95`——LEACH有5%的概率跳过整轮数据传输。这在原始LEACH论文中不存在，是人为引入的性能抑制。

**(b) LEACH的cluster_formation有radio_range限制**

`leach_protocol.py` L226: `if min_distance <= self.radio_range and min_distance < distance_to_bs`——节点只有在CH距离小于radio_range且小于到BS距离时才加入簇。这个双重条件在原始LEACH中不存在（原始LEACH是无条件加入最近CH）。这会导致大量节点成为"unclustered"，被迫直传BS，在恶劣信道下PDR极低。

**(c) PEGASIS的leader→BS使用固定5dBm功率**

`benchmark_protocols.py` L764: `tx_power_dbm=5.0`——PEGASIS leader向BS传输时使用固定5dBm，而AERIS使用10dBm。这是一个直接的不公平比较。

### 3.3 PDR差距不合理

在outdoor_urban 1000节点下，AERIS PDR=0.8846，而LEACH=0.0622、PEGASIS=0.0497、HEED=0.0341。**14-26倍的PDR差距**在同一信道模型下是不合理的。即使AERIS有架构优势，这种量级的差距更可能来自Baseline实现的缺陷，而非AERIS的算法优越性。

---

## 四、统计方法学问题

### 4.1 Hedges' g值异常

Table 5报告的Hedges' g值：indoor_factory +138.53，outdoor_urban +180.04。这些值在统计学上是荒谬的——正常的效应量很少超过5.0。论文虽然加了脚注解释"large-n comparisons with very small within-group variance"，但这恰恰说明：**实验设计有问题**。

当n=1000且组内方差极小时，任何微小差异都会产生巨大的效应量。这不是统计方法的问题，而是实验设计的问题——组内方差极小意味着仿真器的随机性不足，或者仿真参数过于确定性。

### 4.2 S11的自我否定

S11是论文自己设计的"matched patch-control"实验。结果是：**AERIS在所有24个cell中PDR都显著下降**（delta range: -0.0052 to -0.7474）。论文将此解释为"stricter physics assumptions reduce reliability"，但更直接的解释是：**一旦仿真器更接近真实物理，AERIS的优势就消失了**。

outdoor_urban 1000节点下，patch模式AERIS PDR从0.8846降到0.1372——降幅84.5%。这不是"calibration evidence"，这是对核心结论的根本性否定。

---

## 五、论文写作与结构问题

### 5.1 过度防御性写作

论文充斥着"scope boundary"、"regime-specific"、"trend-level evidence"等限定词。几乎每个结论后面都跟着一个caveat。这种写作风格给审稿人的印象是：**作者自己也不确信结论的稳健性**。

### 5.2 Evidence regime分离是掩盖问题的策略

将S8（无MAC碰撞）和S9/S10/S11（有MAC碰撞）分为不同"evidence regimes"，然后声称"cross-regime numerical pooling is avoided"——这实际上是在说：**我们知道加入MAC碰撞后结果会变差，所以我们把两组结果分开报告，避免直接比较**。

一个严谨的做法应该是：修复仿真器，在统一的物理模型下重跑所有实验，报告一组一致的结果。而不是保留一个已知有缺陷的"frozen baseline"作为核心证据。

### 5.3 图表过密

549行tex中包含12张图+12张表=24个浮动体。平均每22.9行一个浮动体。这导致论文读起来像一个实验报告而非学术论文，缺乏深入的分析和讨论。

---

## 六、具体可拒稿风险清单

| # | 级别 | 问题 | 审稿人可能的措辞 |
|---|---|---|---|
| R1 | **Fatal** | 核心S8矩阵无MAC碰撞建模，PDR随节点数上升 | "The simulator lacks fundamental MAC-layer modeling, producing physically impossible trends. All large-scale conclusions are unreliable." |
| R2 | **Fatal** | Baseline比较不公平（架构差异+实现抑制） | "The comparison is structurally unfair. AERIS has multi-hop relay, retransmission, and safety fallback that baselines lack. The 14x PDR gap is an artifact of implementation asymmetry." |
| R3 | **Major** | 算法创新不足（线性加权+PCA+模糊逻辑的拼装） | "The proposed components (CAS, Gateway, Skeleton) are straightforward combinations of known techniques without algorithmic novelty." |
| R4 | **Major** | S11自我否定核心结论 | "The authors' own S11 experiment shows AERIS PDR drops significantly under realistic physics. This undermines the paper's central claim." |
| R5 | **Major** | NS-3验证不充分（仅2协议，多cell不显著） | "NS-3 validation covers only AERIS vs LEACH and shows non-significance in multiple cells. This is insufficient cross-platform validation." |
| R6 | **Minor** | Hedges' g值异常（>100），暴露实验设计问题 | "Effect sizes exceeding 100 indicate either a flawed experimental design or insufficient stochasticity in the simulator." |
| R7 | **Minor** | Related Work缺乏与近年自适应WSN协议的对比 | "The related work section does not adequately position the contribution against recent adaptive/learning-based WSN routing methods." |

---

## 七、修改建议（如果选择修改而非撤稿）

### 必须修改（否则必被拒）

1. **修复仿真器**：在统一的物理模型（含MAC碰撞）下重跑所有实验。不能保留"frozen S8"作为核心证据。
2. **公平化Baseline**：给所有Baseline加上相同的多跳中继和链路层重传能力，然后重新比较。
3. **修复PEGASIS的5dBm功率问题**：所有协议使用相同的tx_power。
4. **删除LEACH的5%随机跳过**：恢复原始LEACH行为。
5. **重新定位论文贡献**：如果修复后AERIS优势缩小到合理范围（2-5倍而非14倍），将论文定位为"系统集成与实验评估"而非"算法创新"。

### 建议修改

6. 补充与近年自适应WSN协议的详细对比。
7. 减少图表数量（24个浮动体→12-16个），增加分析深度。
8. 删除过度防御性的"scope boundary"措辞，直接报告结论。
9. 在信道模型中加入时间相关性和多径衰落。

---

## 八、最终判定

**Recommendation: Major Revision (倾向 Reject)**

本文在实验规模和数据管理方面投入了大量工作，但在三个核心维度（创新性、仿真严谨性、比较公平性）存在根本性问题。如果不修复仿真器和公平化Baseline，论文的核心结论不可信。

如果作者愿意：(1) 在统一物理模型下重跑全部实验，(2) 公平化Baseline实现，(3) 重新定位论文贡献，则有可能在Major Revision后达到发表水平。但这需要大量的重新实验和重写工作。

---

*报告生成: Claude 4.6 (Opus), 2026-02-22*
*源代码审查: aeris_protocol.py (2700+ lines), benchmark_protocols.py (1228 lines), leach_protocol.py (438 lines), pegasis_protocol.py (466 lines), cas_selector.py (150+ lines), gateway_selector.py (150+ lines), skeleton_selector.py (80+ lines), mac_collision_model.py (80+ lines), realistic_channel_model.py (150+ lines)*
*论文审查: v49.tex (549 lines, 12 figures, 12 tables, 17 references)*

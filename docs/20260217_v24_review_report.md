# AERIS v24 外部审稿专家组评审报告

> 审稿对象: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260217_v24.tex`
> 审稿日期: 2026-02-17
> 审稿标准: Sensors (MDPI) 投稿门槛，严格、可追溯、可执行

---

## 1. 审稿总览

**总体判断**: 本稿在边界声明和统计严谨性方面做了大量工作，远超多数WSN论文的自我约束水平。作者明确区分了三个证据块（100节点、大规模S8、NS-3趋势），避免了跨块数值池化，并在摘要和结论中反复强调scope boundary。这是值得肯定的。

**核心问题**: 然而，文稿存在一个根本性的物理合理性缺陷——S8冻结矩阵中AERIS的PDR在3/4环境下随节点数增加而上升（indoor_factory: 0.93→0.97, outdoor_urban: 0.76→0.88, outdoor_suburban: 0.97→0.99），这违反了基本的无线信道竞争物理规律。虽然作者在S9 patch实验中展示了修复后的单调递减趋势，但核心scalability claims仍然基于物理不合理的S8矩阵。这是审稿人最可能攻击的致命弱点。

**次要问题**: S9/S10数据已在v24中报告，但这些数据的白名单状态不明确（v19白名单未覆盖S9/S10文件）。NS-3 CLAIM_GATE.md中的26/28与实际数据25/28存在不一致。文稿结构偏重于防御性声明，导致可读性下降。

---

## 2. 证据一致性问题清单

| # | 严重级别 | 位置 | 问题描述 | 证据路径 | 改写建议 |
|---|---------|------|---------|---------|---------|
| E1 | **High** | v24 L165-168, tab:scale1000 | S8矩阵中AERIS PDR随节点数上升（3/4环境），物理不合理。indoor_factory 100n=0.9333→1000n=0.9726，outdoor_urban 100n=0.7559→1000n=0.8846。核心scalability claim基于此数据。 | `claim_source_matrix_v3.csv` C77/C78/C79 标注为 CRITICAL ANOMALY | 必须在tab:scale1000附近添加显式警告："Note: In three of four environments, AERIS PDR increases with scale under the original simulator (no MAC contention model). This trend is physically implausible and is addressed by the S9 patch experiment below." |
| E2 | **High** | v24 L173 | "AERIS is first in all four environments in the current balanced S8 matrix" — 虽然数值正确，但基于物理不合理的数据做排名声明，审稿人会质疑排名的有效性。 | `scalability_4env_s8_unified_20260215_descriptive.csv` | 改为："Under the original simulator settings (without MAC contention), AERIS ranks first in all four environments. However, this ranking should be interpreted with caution given the physically implausible PDR-scale trend noted above." |
| E3 | **Medium** | v24 L251 | "AERIS remains above LEACH/HEED/TEEN under patch mode in all four tested environments at 1000 nodes" — 需核实。S9 CSV: outdoor_urban 1000n AERIS patch=0.1372, PEGASIS patch=0.0987, LEACH patch=0.0041, HEED=0.0016, TEEN=0.0025。AERIS确实高于LEACH/HEED/TEEN，但PEGASIS=0.0987也低于AERIS。声明正确。 | `s9_matched_4env_patch_vs_control_20260216_merged.csv` L127,133,139,145,151 | 无需修改，但建议补充PEGASIS的具体数值以增强可信度。 |
| E4 | **Medium** | v24 L254 | "59/60 cells are significant after Holm correction" — 与S10 CSV一致（LEACH indoor_office 1000n p=0.4812 为唯一non-significant）。 | `s10_4env_significance_tx5_vs_tx15_20260216.csv` L7 | 数值正确，无需修改。 |
| E5 | **Medium** | v24 L303 | "25/28 AERIS-versus-LEACH comparisons are significant" — 与NS-3 significance CSV一致（3个non-sig: indoor_office n=100/200/1000）。但NS3_CLAIM_GATE.md §1第1条写"26/28"，存在内部文档不一致。 | `ns3_scale_ext_1000_significance.csv`: 25 YES / 3 NO; `NS3_CLAIM_GATE.md` L16: "26个" | v24正文25/28正确。需修复CLAIM_GATE.md中的26/28→25/28。 |
| E6 | **Medium** | v24 L285 | "AERIS is directionally not lower than LEACH in all tested environment-scale cells" — 实际上indoor_office n=200 AERIS < LEACH by 0.0004。claim_source_matrix C75标注为PARTIAL/WARNING。 | `ns3_scale_ext_1000_significance.csv` indoor_office n=200: diff=-0.000417 | 改为："AERIS is directionally not lower than LEACH in 27 of 28 tested cells; the single exception is indoor\_office at 200 nodes (diff = −0.0004, not significant)." |
| E7 | **Low** | v24 L225 | S9 patch n=1000 vs control n=600，样本量不匹配。虽然Welch t-test不要求等样本，但审稿人可能质疑为何不统一。 | `s9_matched_4env_patch_vs_control_20260216_merged.csv` | 建议在正文中添加一句解释："The unequal sample sizes (patch n=1000, control n=600) reflect the sequential experiment design; Welch's t-test accommodates unequal variances and sample sizes." |
| E8 | **Low** | v24 Abstract L31 | 摘要提到"AERIS remains the highest-mean protocol in all 24 environment-scale cells under the original simulator settings" — 这是S8数据，但摘要未提及S8的物理不合理性。 | `claim_source_matrix_v3.csv` C76 | 建议在摘要中添加qualifier："under the original simulator settings (without MAC contention modeling)" |

---

## 3. 统计口径风险清单

### 3.1 可写 / 慎写 / 禁写 三栏清单

| 类别 | 声明内容 | 依据路径 | 判定 |
|------|---------|---------|------|
| **可写** | AERIS在100节点4环境中PDR均值最高（n=30） | `env_sensitivity_20260207_205317.json` via `claim_source_matrix_v3.csv` C01-C21, 全部 match=YES | ✅ 可写 |
| **可写** | Gateway在3/4环境中正向，indoor_office近中性（+0.0002） | `ablation_diag_multi_20260207_205448.json` via C22-C30, match=YES | ✅ 可写 |
| **可写** | NS-3趋势级验证：25/28显著（Holm α=0.05） | `ns3_scale_ext_1000_significance.csv` 实际计数=25 YES | ✅ 可写 |
| **可写** | S10 59/60 cells显著，唯一non-sig为LEACH indoor_office 1000n | `s10_4env_significance_tx5_vs_tx15_20260216.csv` L7: p=0.4812 | ✅ 可写 |
| **可写** | AERIS hop count 1.97-1.99，PEGASIS >31 hops | `latency_hop_fix_20260209_074608_stats.csv` via C60-C61 | ✅ 可写 |
| **慎写** | S8矩阵AERIS在1000节点4环境排名第一 | `scalability_4env_s8_unified_20260215_descriptive.csv` C31-C51 数值正确，但C77-C79标注CRITICAL ANOMALY | ⚠️ 慎写：必须附带物理不合理性警告 |
| **慎写** | Hedges' g 值在harsh环境极大（138, 180） | `scalability_4env_s8_unified_20260215_significance.csv` C55/C57 | ⚠️ 慎写：v24 L194已有解释，但审稿人仍可能质疑g>100的实际意义 |
| **慎写** | S9 patch下AERIS在所有环境1000n仍高于LEACH/HEED/TEEN | `s9_merged.csv` 数值正确 | ⚠️ 慎写：需明确这是stress-test而非替代S8的正式结论 |
| **禁写** | "AERIS PDR随规模提升"或暗示规模越大PDR越高 | C77/C78/C79 CRITICAL ANOMALY | ❌ 禁写：物理不合理 |
| **禁写** | "NS-3数值验证了Python结果" | `NS3_CLAIM_GATE.md` 禁写条目1: Python 97.4% vs NS-3 92.0% | ❌ 禁写 |
| **禁写** | "AERIS在所有环境所有规模显著优于LEACH（NS-3）" | 3个non-sig cells: indoor_office n=100/200/1000 | ❌ 禁写 |
| **禁写** | "NS-3验证了所有消融结论" | `NS3_CLAIM_GATE.md` 禁写条目4 | ❌ 禁写 |

### 3.2 统计方法审查

| # | 问题 | 严重级别 | 位置 | 说明 |
|---|------|---------|------|------|
| S1 | ddof不一致 | Medium | v24 L103, L131 | tab:pdr100 caption说"std follows the result-file convention with ddof=0"，tab:ablation_gateway说"std reported as population standard deviation, ddof=0"。但S9/S10 CSV中的std是ddof=1（样本标准差）。同一篇论文中混用ddof=0和ddof=1，审稿人会困惑。建议统一为ddof=1并在Methods中说明。 |
| S2 | Hedges' g 极端值 | Low | v24 L185-187 | g=138.5, 180.0, 98.9 — 这些值在统计学文献中极为罕见。v24 L194已有解释（"magnitude indicators within this experiment design"），但建议进一步说明这是因为n=1000导致within-group variance极小，而非效应本身异常。 |
| S3 | S9 patch vs control 样本量不等 | Low | v24 L225 | patch n=1000, control n=600。Welch t-test可处理，但建议说明原因。 |
| S4 | S10 delta方向解释 | Low | v24 L264-267 | Delta列标注为"Delta (5--15)"，即tx5 - tx15。outdoor_urban delta=-0.1019意味着tx15更好。但v24 L272说"raising tx-power does not guarantee higher PDR"，这与outdoor_urban的数据一致（tx15=0.1606 > tx5=0.0587）。逻辑正确但表述可更清晰。 |

---

## 4. NS-3 边界问题清单

| # | 严重级别 | 位置 | 问题句子 | 风险分析 | 改写建议 |
|---|---------|------|---------|---------|---------|
| N1 | **High** | v24 L285 | "AERIS is directionally not lower than LEACH in all tested environment-scale cells" | **误导**：indoor_office n=200 AERIS < LEACH (diff=-0.0004)。审稿人若核查会发现此声明不准确。 | 改为 "in 27 of 28 tested cells" 并注明例外。 |
| N2 | **Medium** | v24 L285 | "significance is confirmed in 3/4 environments" (100节点) | 正确，但紧接着说"while indoor\_office remains non-significant"时未给出具体p值和g值。 | 补充 "(Holm p=1.00, g=0.35)" |
| N3 | **Medium** | v24 L303 | "25/28 AERIS-versus-LEACH comparisons are significant; the non-significant cells are indoor\_office at node counts 100, 200, and 1000" | 正确。但内部文档 `NS3_CLAIM_GATE.md` L16 写"26个"，需修复文档一致性。 | v24正文无需改，修复CLAIM_GATE.md。 |
| N4 | **Low** | v24 L284-285 | "NS-3 validation is used as a cross-platform trend check rather than a numerical-equivalence proof" | 表述正确且反复强调，符合CLAIM_GATE要求。 | 无需修改，保持。 |
| N5 | **Low** | v24 L330 | "NS-3 trend-level publication gate is completed; numerical equivalence is intentionally not claimed" | 正确，与 `NS3_ALIGNMENT_EVIDENCE.md` §7 一致。 | 无需修改。 |

**NS-3边界总评**: v24在NS-3使用边界上做得较好，反复强调trend-level，未越界声称数值对齐。唯一需修复的是N1（方向性声明不准确）和N3（内部文档不一致）。

---

## 5. 图表评审（按Sensors可发表标准）

### Fig 1: PDR Comparison Panel (100 nodes, 4 environments)
- **配色**: 7/10 — 5协议配色可区分，但部分颜色在灰度打印下可能混淆
- **层级**: 8/10 — 4子图布局清晰，环境标签明确
- **线条/标注**: 7/10 — 误差棒可见，但bar chart密集时标注可能重叠
- **图例**: 8/10 — 协议名称完整
- **信息密度**: 7/10 — 适中
- **可读性**: 7/10 — 整体可读，outdoor_urban子图中低PDR协议的差异难以辨认
- **误导风险**: 3/10（低风险）— y轴从0开始，无截断
- **达投稿线**: ✅ 是
- **改进建议**: (1) 考虑在outdoor_urban子图中添加数值标注（低PDR区域差异小）；(2) 确保灰度打印下5种颜色仍可区分（加hatching pattern）

### Fig 2: Ablation Panel (heatmap + marginal effects)
- **配色**: 8/10 — heatmap色阶合理
- **层级**: 7/10 — heatmap + marginal effects双面板布局合理
- **标注**: 7/10 — heatmap cell内数值可读
- **图例**: 7/10 — 色阶bar存在
- **信息密度**: 8/10 — 紧凑但不过载
- **可读性**: 7/10 — 消融变体名称（full, no_gateway, no_cas等）需确保字号足够
- **误导风险**: 2/10（低风险）
- **达投稿线**: ✅ 是
- **改进建议**: (1) 确保heatmap中的数值标注字号≥8pt；(2) marginal effects子图中添加零线参考

### Fig 3: Scalability Panel (100-1000 nodes, 4 environments)
- **配色**: 7/10
- **层级**: 8/10 — 4环境子图布局清晰
- **线条**: 7/10 — 95% CI bands可见
- **图例**: 8/10
- **信息密度**: 8/10
- **可读性**: 7/10 — indoor_office子图使用narrowed y-axis（caption已说明），合理
- **误导风险**: **7/10（高风险）** — **这是最大问题**：图中展示的S8数据在3/4环境中AERIS PDR随节点数上升，但图表本身未标注此趋势的物理不合理性。审稿人看到此图会立即质疑。
- **达投稿线**: ⚠️ 有条件 — 需添加物理不合理性标注
- **改进建议**: **(P0必改)** 在图caption或图内添加注释："Note: Under the original simulator (no MAC contention), AERIS PDR increases with scale in 3/4 environments — a known limitation addressed in Section 5.4 (S9 patch)." 或者考虑将S9 patch趋势作为overlay/inset展示。

### Fig 4: Trade-off Panel (reliability, energy, hop-based latency, lifetime)
- **配色**: 7/10
- **层级**: 8/10 — 4维度子图布局合理
- **标注**: 7/10 — hop panel使用log x-axis（caption已说明）
- **图例**: 8/10
- **信息密度**: 8/10 — 4维度在一张图中，紧凑
- **可读性**: 7/10
- **误导风险**: 3/10（低风险）
- **达投稿线**: ✅ 是
- **改进建议**: (1) 确保log轴刻度标签清晰；(2) 考虑在energy子图中标注"lower is not always better if lifetime is shorter"的caveat

---

## 6. 四位审稿人结论

### 6.1 方法学保守派审稿人

**结论: Major Revision**

**三条最关键理由:**
1. S8冻结矩阵中PDR随规模上升（3/4环境）违反物理直觉，核心scalability claim基于此数据，方法学上不可接受。
2. 仿真器缺少MAC碰撞模型，所有协议的绝对PDR值缺乏物理校准基础，协议间的相对排名可能因碰撞建模而改变。
3. S9 patch实验已展示修复方向，但未完成全量替换——论文同时报告两套矛盾的scalability数据（S8上升 vs S9下降），读者无法判断哪套可信。

**必改项（按优先级）:**
1. (P0) 要么用S9 patch数据替换S8作为核心矩阵，要么在S8表格和图表中添加显式物理不合理性警告
2. (P0) 在Methods中明确说明仿真器不含MAC碰撞模型，并讨论其对结果的影响
3. (P1) 统一ddof标准（ddof=0 vs ddof=1）

### 6.2 统计学严格派审稿人

**结论: Minor Revision**

**三条最关键理由:**
1. 统计方法选择合理（Welch t-test + Holm + Hedges' g），且作者明确区分了统计显著性与工程意义，这在WSN文献中少见。
2. Hedges' g 极端值（>100）虽然数学正确，但缺乏充分的统计学解释——应说明这是大样本+低方差的数学后果，而非效应量的常规解读。
3. ddof不一致（tab:pdr100用ddof=0，S9/S10 CSV用ddof=1）是一个小但可修复的问题。

**必改项:**
1. (P1) 在Methods中添加一段关于极端Hedges' g值的解释
2. (P1) 统一所有表格的ddof标准
3. (P2) S9 patch vs control样本量不等的原因说明

### 6.3 工程复现派审稿人

**结论: Major Revision**

**三条最关键理由:**
1. 仿真器是Python自研实现，无MAC层、无真实调度，与NS-3的PDR差距达5-30%（AERIS indoor_office: Python 97.4% vs NS-3 92.0%；LEACH差距更大）。仿真器的物理保真度不足以支撑定量结论。
2. 代码和数据虽承诺公开，但当前未提供公开仓库链接，复现性无法验证。
3. S8矩阵中PDR随规模上升的异常趋势表明仿真器存在系统性偏差，仅靠S9 patch的"sanity check"不足以修复——需要完整的MAC碰撞模型重跑。

**必改项:**
1. (P0) 在Methods中详细描述仿真器的信道模型、能耗模型、调度假设，使读者可独立判断保真度
2. (P0) 提供代码仓库链接或明确的复现指令
3. (P1) 完成MAC碰撞模型修复后的全量重跑，或将当前结果明确标注为"pre-calibration"

### 6.4 应用价值导向审稿人

**结论: Minor Revision**

**三条最关键理由:**
1. AERIS的设计理念（环境感知+轻量级规则）对实际WSN部署有价值，论文清晰地展示了不同环境下的性能差异，这对工程选型有参考意义。
2. 多环境对比（4种信道条件）和模块消融（Gateway/CAS/Fairness）的实验设计合理，为实际部署提供了模块选择依据。
3. 论文过度防御性的写法（反复声明scope boundary）降低了可读性，工程读者可能在大量caveat中迷失核心贡献。

**必改项:**
1. (P1) 精简防御性声明，将重复的scope boundary合并到一处（如Discussion开头），正文中用简短引用
2. (P1) 添加一个"Practical Deployment Guidance"段落，总结在不同环境下应启用/禁用哪些模块
3. (P2) 考虑添加一个简化的决策流程图，帮助工程读者快速判断AERIS是否适用于其场景

---

## 7. 四封审稿邮件（中文）

### 7.1 Reject 视角邮件

**致编辑/作者：**

感谢作者提交本稿。本文提出了AERIS协议并在多环境下进行了评估，实验规模和统计方法值得肯定。

然而，本稿存在以下根本性问题，导致我建议拒稿：

1. **仿真器物理保真度不足**：核心scalability矩阵（S8）中，AERIS的PDR在3/4环境下随节点数增加而上升（indoor_factory: 0.93→0.97），这违反了无线信道竞争的基本物理规律。作者自己的NS-3实验也证实了PDR应随规模下降。基于物理不合理数据得出的排名结论不可信。

2. **两套矛盾数据并存**：S8（无碰撞模型，PDR上升）和S9 patch（有碰撞模型，PDR下降）同时出现在论文中，但核心claim仍基于S8。这表明作者已知问题但未完成修复。

3. **仿真器缺少MAC碰撞建模**：所有协议的绝对PDR值缺乏物理校准基础，协议间的相对排名可能因碰撞建模而根本改变。

**建议**：完成MAC碰撞模型修复和全量重跑后重新投稿。当前的S9 patch实验方向正确，但需要成为核心矩阵而非附录。

### 7.2 Major Revision 视角邮件

**致编辑/作者：**

本文在WSN多环境路由评估方面做了扎实的工作，统计方法规范（Welch + Holm + Hedges' g），边界声明清晰，这在同类文献中较为突出。建议大修后重审。

**关键问题：**

1. **S8矩阵物理合理性**：核心scalability数据中PDR随规模上升（3/4环境），必须在正文和图表中添加显式警告，或用S9 patch数据替换核心矩阵。这是大修的核心要求。

2. **仿真器描述不足**：Methods中缺少对信道模型、能耗模型、调度假设的详细描述。读者无法独立评估仿真保真度。

3. **NS-3方向性声明不准确**：L285声称"directionally not lower in all cells"，但indoor_office n=200 AERIS < LEACH。需修正。

**下一步要求**：(1) 修复S8物理合理性问题（添加警告或替换数据）；(2) 补充仿真器详细描述；(3) 修正NS-3方向性声明；(4) 统一ddof标准。

### 7.3 Minor Revision 视角邮件

**致编辑/作者：**

本文展示了一个设计合理的多环境WSN路由评估框架，统计严谨性和边界声明的自律程度在Sensors同类稿件中属于上游水平。建议小修后接收。

**需修复的问题：**

1. L285 NS-3方向性声明需修正（indoor_office n=200例外未提及）。
2. ddof标准不统一（tab:pdr100用ddof=0，其他用ddof=1），建议统一并在Methods中说明。
3. Fig 3 caption应添加关于S8 PDR-scale趋势物理合理性的注释。
4. 防御性声明过多导致可读性下降，建议合并重复的scope boundary到Discussion开头。

**决定**：Minor Revision。上述问题均可在一轮修改中完成。

### 7.4 Accept 视角邮件

**致编辑/作者：**

本文在以下方面表现突出：(1) 明确的指标定义和统计方法；(2) 三个证据块的严格分离；(3) NS-3趋势验证的边界自律；(4) 完整的消融实验和环境依赖性分析。

作者对自身工作局限性的坦诚（S8物理合理性问题、NS-3非数值对齐、CAS效果混合）在同类文献中罕见，体现了良好的科研诚信。

**建议接收**，但请在camera-ready版本中：(1) 修正L285方向性声明；(2) 统一ddof标准；(3) 在Fig 3 caption中添加物理合理性注释。

---

## 8. 导师视角修改意见邮件（中文）

**致学生：**

v24版本在边界声明和统计规范方面已经做得很好，但要提高录用概率，需要解决一个核心矛盾：**论文同时报告了两套scalability数据（S8上升 vs S9下降），审稿人会问"到底哪个是真的？"**

### 修改路线图

#### 3天路线（最小可行修改，应对Minor Revision）

| 天 | 任务 | 产出 |
|----|------|------|
| D1 | 修正L285方向性声明（27/28）；统一ddof为ddof=1；修复CLAIM_GATE.md 26/28→25/28 | 修改后的tex + 修复后的md |
| D2 | Fig 3 caption添加物理合理性注释；tab:scale1000添加footnote警告S8趋势异常 | 修改后的tex + 重新生成的fig3 |
| D3 | 精简重复的scope boundary声明（合并到Discussion §6开头一段）；通读全文确认一致性 | 最终tex |

#### 7天路线（应对Major Revision）

| 天 | 任务 | 产出 |
|----|------|------|
| D1-D3 | 完成3天路线的所有修改 | 同上 |
| D4 | 在Methods中补充仿真器详细描述（信道模型公式、能耗模型参数、调度假设） | 新增§3.3 Simulator Architecture |
| D5 | 将S9 patch数据提升为正式证据块：添加白名单条目、更新claim_source_matrix | 更新后的白名单和matrix |
| D6 | 重写§5.3 Scalability：以S8为"original setting"、S9为"calibrated setting"双轨呈现 | 修改后的§5.3 |
| D7 | 全文通读 + 图表一致性检查 + 运行validate_claim_source_matrix.py | 最终tex + FAIL=0 |

#### 14天路线（应对Reject后重投）

| 天 | 任务 | 产出 |
|----|------|------|
| D1-D7 | 完成7天路线 | 同上 |
| D8-D10 | 完成MAC碰撞模型全量重跑（4环境×6节点数×5协议×n=1000） | 新的scalability JSON |
| D11-D12 | 用新数据替换S8矩阵，重新生成所有图表和统计表 | 新的fig1-fig4 + 新的CSV |
| D13 | 更新白名单、claim_source_matrix、NS-3对比分析 | 完整的证据链 |
| D14 | 全文重写Results和Discussion，确保所有claim基于新数据 | 最终tex |

### 明确不做的事项（避免范围膨胀）

1. **不做**：5协议NS-3对比（当前只有AERIS vs LEACH，扩展到5协议需要大量NS-3开发工作，不值得在本轮修改中做）
2. **不做**：硬件在环验证（论文已在Limitations中声明，审稿人不会因此拒稿）
3. **不做**：新增拓扑族（corridor, hotspot）——当前4环境已足够
4. **不做**：重构代码或优化仿真器性能——与论文质量无关
5. **不做**：添加更多baseline协议——5个已足够

---

## 9. 可执行改稿清单（P0/P1/P2）

### P0 — 阻塞发布（必须在投稿前修复）

| # | 问题 | 位置 | 具体操作 |
|---|------|------|---------|
| P0-1 | S8 PDR-scale趋势物理不合理，未在正文/图表中警告 | v24 L156-173, Fig3 | 在tab:scale1000下方添加footnote；在Fig3 caption添加注释；或在§5.3开头添加一段说明 |
| P0-2 | L285 NS-3方向性声明不准确 | v24 L285 | "in all tested environment-scale cells" → "in 27 of 28 tested cells; indoor\_office at 200 nodes is the single exception (diff=−0.0004, not significant)" |
| P0-3 | Fig3 scalability图展示物理不合理趋势但无标注 | fig3_scalability_panel | caption添加："Note: Under original simulator settings (no MAC contention), AERIS PDR increases with scale in 3/4 environments—see Section 5.4." |

### P1 — 应修复（提高录用概率）

| # | 问题 | 位置 | 具体操作 |
|---|------|------|---------|
| P1-1 | ddof不一致 | v24 L103, L131 | 统一为ddof=1（样本标准差），在§4.4 Statistical Methods中说明 |
| P1-2 | 摘要未提及S8物理合理性限制 | v24 L31 | 在"under the original simulator settings"后添加"(without MAC contention modeling)" |
| P1-3 | S9/S10数据未纳入v19白名单 | `evidence_whitelist_v19.md` | 新增W9-W12条目覆盖S9/S10的CSV文件 |
| P1-4 | CLAIM_GATE.md 26/28与实际25/28不一致 | `NS3_CLAIM_GATE.md` L16 | 修正为25/28 |
| P1-5 | 防御性声明过多降低可读性 | 全文多处 | 合并重复的scope boundary到Discussion §6开头一段，正文中用简短引用 |
| P1-6 | Methods缺少仿真器架构描述 | §3-§4之间 | 新增§3.3描述信道模型公式、能耗模型、调度假设 |
| P1-7 | S9 patch vs control样本量不等未解释 | v24 L225 | 添加一句Welch t-test适用性说明 |

### P2 — 建议改进（锦上添花）

| # | 问题 | 位置 | 具体操作 |
|---|------|------|---------|
| P2-1 | Hedges' g极端值解释可更充分 | v24 L194 | 补充"large n + low within-group variance"的数学解释 |
| P2-2 | Fig1 outdoor_urban子图低PDR区域难辨 | Fig1 | 添加数值标注或放大inset |
| P2-3 | 缺少Practical Deployment Guidance | Discussion | 添加一段总结不同环境下的模块选择建议 |
| P2-4 | S10 delta方向表述可更清晰 | v24 L264-267 | 在表格caption中说明"positive delta means tx5 > tx15" |
| P2-5 | 图表灰度打印兼容性 | Fig1-Fig4 | 添加hatching pattern或marker区分 |

---

## 附录：数据交叉核验结果

| 检查项 | v24值 | CSV值 | 匹配 |
|--------|-------|-------|------|
| S9 indoor_office 100n AERIS patch | 0.9714 | 0.971368 | ✅ |
| S9 outdoor_urban 1000n AERIS patch | 0.1372 | 0.137191 | ✅ |
| S10 indoor_office 1000n tx5 | 0.8177 | 0.817701 | ✅ |
| S10 outdoor_urban 1000n tx15 | 0.1606 | 0.160581 | ✅ |
| NS-3 indoor_office AERIS 100n | 0.9202 | 0.920240 | ✅ |
| NS-3 indoor_office Holm p | 1.00 | 1.000000e+00 | ✅ |
| NS-3 significant count | 25/28 | 25/28 | ✅ |
| CLAIM_GATE.md count | 26/28 | 25/28 | ❌ 文档错误 |

---

> 报告结束。所有发现均基于已读取的文件，证据路径已标注。

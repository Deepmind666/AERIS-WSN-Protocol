# AERIS v42 极严苛审稿报告 (2026-02-20)

> 审稿组: Claude 4.6 (Opus) 五角色模拟
> 审稿文件: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260220_v42.tex` (487 lines)
> 图表版本: S42 (8 figures)
> 数据交叉验证: 6 个 CSV 源 + 2 个门控文档

---

## Overall Verdict: Minor Revision

---

## Findings

### P0 (阻断级): 0 项

无禁写断言命中。无数据伪造。无统计方法滥用。

### P1 (必修级): 5 项

**P1-1: NS3_ALIGNMENT_EVIDENCE.md 内部 25/28 vs 26/28 矛盾**
- 证据: `ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md` L134 写 "26/28 comparisons are statistically significant"
- 同文件 L99 写 "significance confirmed in 25/28 comparisons"，L103 写 "Total: 25/28"
- tex L411 写 "25/28"，与 L99/L103 一致，与 L134 矛盾
- 风险: 审稿人若追溯证据文档，会发现内部不一致
- 最小修复: 将 `NS3_ALIGNMENT_EVIDENCE.md` L134 的 "26/28" 改为 "25/28"

**P1-2: NS3_ALIGNMENT_EVIDENCE.md 数据与权威 stats CSV 不一致**
- 证据: `NS3_ALIGNMENT_EVIDENCE.md` Section 3.1 表格中:
  - indoor_factory AERIS = 0.599620，但 `ns3_scale_ext_1000_stats.csv` = 0.602530 (差 0.0029)
  - indoor_factory LEACH = 0.529870，但 CSV = 0.533557 (差 0.0037)
  - outdoor_suburban LEACH = 0.696343，但 CSV = 0.692123 (差 0.0042)
- tex 中的值与 stats CSV 一致（8/8 OK），所以论文本身无误
- 风险: 证据文档与权威数据源不同步，可能来自不同批次的 NS3 运行
- 最小修复: 用 `ns3_scale_ext_1000_stats.csv` 的值更新 `NS3_ALIGNMENT_EVIDENCE.md` Section 3.1 表格

**P1-3: 创新性表述不足——缺少与 SOTA 的定量对比**
- 证据: tex L51-54 (Related Work) 仅用两段泛述，未给出任何定量对比
- 风险: Sensors 审稿人常问 "与最新方法相比优势在哪"。当前 Related Work 只引了 6 篇近期文献，无定量 benchmark
- 最小修复: 在 Related Work 末尾加 1-2 句，明确说明为何选择 LEACH/PEGASIS/HEED/TEEN 作为 baseline（经典协议覆盖四种路由范式），并承认未与 RL/DRL 类方法做直接对比（因计算开销不在同一量级）

**P1-4: CAS 模式选择公式 (Eq.3) 缺少参数说明**
- 证据: tex L76-81，公式给出了 $w_{1,m}, w_{2,m}, w_{3,m}, b_m$ 但未说明这些权重如何确定
- 风险: 审稿人会问 "权重是学习得到的还是手动设定的？如果是手动的，敏感性如何？"
- 最小修复: 在 Eq.3 后加一句说明权重来源（如 "weights are fixed per environment based on offline grid search" 或类似）

**P1-5: 100-node 表 (tab:pdr100) 缺少显著性标注**
- 证据: tex L159-163，只报告了 mean±std，未标注 AERIS vs 各 baseline 的显著性
- 风险: 审稿人会问 "AERIS 排第一，但差异显著吗？"
- 最小修复: 在表格下方或脚注中加一句 "All pairwise AERIS-vs-baseline comparisons are significant after Holm correction (p < 0.001)"，或在表中用星号标注

### P2 (建议级): 7 项

**P2-1: Gateway 评分公式 (Eq.2) 权重 α/β/γ 未给数值**
- 证据: tex L72-75
- 风险: 低——公式是架构说明而非核心贡献，但审稿人可能追问
- 最小修复: 加脚注说明默认值或指向代码仓库

**P2-2: Ablation 只展示 Gateway，缺 CAS/Skeleton/Safety 消融表**
- 证据: tex L177-196，tab:ablation_gateway 只有 Full vs no_gateway
- NS3_CLAIM_GATE.md L30-34 记录了 CAS 在 indoor_office 有负面效果 (−1.5%)
- 风险: 审稿人会问 "其他模块的消融呢？"
- 最小修复: 在 Ablation 段末加一句总结其他模块效果（"CAS is environment-dependent; Skeleton and Safety show no measurable marginal effect in this setup"），引用 Claim Gating List

**P2-3: 摘要词数 180 → 接近 MDPI 200 词上限**
- 证据: 自动计数 = 180 词
- 风险: 无——仍在限制内
- 最小修复: 无需修改

**P2-4: fig0 (workflow) 信息密度偏低**
- 证据: `fig0_aeris_workflow_20260220_s42.pdf` (30.8KB)
- 图为流程图，展示 round execution → evidence pipeline
- 风险: 占据整页但信息量有限，审稿人可能建议移至附录
- 最小修复: 可选——压缩为半页宽度，或在 caption 中强调其与 reproducibility 的关联

**P2-5: tab:robust_snapshot 标题写 "AERIS vs LEACH" 但实际比较对象不全是 LEACH**
- 证据: tex L225 subsection 标题 "AERIS vs LEACH"，但 L234 比较对象是 PEGASIS，L235-237 比较对象是 TEEN
- 风险: 标题误导
- 最小修复: 将 subsection 标题改为 "Statistical Robustness Snapshot (AERIS vs Strongest Baseline)" 或类似

**P2-6: Conclusion 第三段 (L476) 是新增的实用总结，但与 Discussion 的 Deployment Guidance 有重复**
- 证据: tex L476 vs L427-433
- 风险: 低——但审稿人可能觉得冗余
- 最小修复: 可选——删除 L476 或将其与 Discussion 合并

**P2-7: Data Availability 未给出具体仓库 URL**
- 证据: tex L482 "will be released in a versioned public repository upon final publication"
- 风险: Sensors 要求尽量提供数据链接
- 最小修复: 加 "available at [URL] upon acceptance" 或 "available upon reasonable request"

---

## 数据交叉验证汇总

| 检查项 | 数据源 | 结果 |
|---|---|---|
| S8 tab:scale1000 (L213-216) vs CSV | `scalability_4env_s8_unified_20260215_descriptive.csv` | 20/20 OK |
| S8 tab:robust_snapshot (L234-237) vs CSV | `scalability_4env_s8_unified_20260215_significance.csv` | 4/4 OK (delta, g, p 全匹配) |
| S10 tab:s10_aeris_1000 (L330-333) vs CSV | `s10_4env_merged_descriptive_20260216.csv` | 8/8 OK |
| S10 59/60 显著性声明 (L345) vs CSV | `s10_4env_significance_tx5_vs_tx15_20260216.csv` | 59 sig + 1 nonsig = LEACH indoor_office 1000，与 tex 一致 |
| NS3 tab:ns3_trend (L396-399) vs CSV | `ns3_scale_ext_1000_stats.csv` | 8/8 OK |
| NS3 25/28 声明 (L411) vs CLAIM_GATE | `NS3_CLAIM_GATE.md` L16/18 | 一致 (25/28) |
| NS3 25/28 vs ALIGNMENT_EVIDENCE | `NS3_ALIGNMENT_EVIDENCE.md` L99/103 vs L134 | L134 仍写 26/28 (P1-1) |
| 禁写断言扫描 | v42 tex 全文 | 0 命中 |
| 摘要词数 | v42 tex L31 | 180 词 (≤200) |
| 文献完整性 | bibliography.bib | 17/17 cite keys 全部存在 |

---

## 图表审稿 (R5 专项)

### fig0: AERIS Workflow (30.8KB)
- 可读性: 流程框清晰，箭头方向明确，字号适中
- 审美: 配色统一（蓝灰系），专业感强
- 信息密度: 偏低——整页流程图但核心信息可压缩为半页
- 结论承载力: 支持 Section 3.3 的 round-level operation 描述
- 灰度兼容: 需确认——蓝色框在灰度打印下可能与灰色框混淆

### fig1: PDR Comparison Panel (27.2KB)
- 可读性: 4 环境子图，bar chart + error bars，图例清晰
- 审美: 配色区分度好，5 协议颜色可辨
- 信息密度: 适中
- 结论承载力: 直接支持 tab:pdr100 和 L168 的 "AERIS ranks first" 声明

### fig2: Ablation Panel (36KB)
- 可读性: heatmap + marginal effects，标注清晰
- 审美: 热力图配色专业
- 信息密度: 高——同时展示多维消融效果
- 结论承载力: 支持 L194 的 Gateway 环境依赖性结论

### fig3: Scalability Panel (36.1KB)
- 可读性: 4 环境子图，线图 + 95% CI bands
- 审美: CI bands 半透明，不遮挡主线
- 信息密度: 高
- 结论承载力: 直接支持 S8 scalability 讨论，indoor_office 窄 y 轴窗口设计合理
- 注意: 非物理上升趋势在图中可见，caption 已做说明 (L249)

### fig4: Trade-off Panel (26.7KB)
- 可读性: 4 子图（reliability, energy, hop, lifetime），对数 x 轴处理得当
- 审美: 统一配色
- 信息密度: 高——四维 trade-off 一图展示
- 结论承载力: 支持 L376 的 hop-based latency 讨论

### fig5: S11 Patch-Control Delta (27.8KB)
- 可读性: (a) delta 轨迹 (b) 1000-node 跨环境比较，清晰
- 审美: 负 delta 用冷色调，直观
- 结论承载力: 直接支持 L366 的 "24/24 significant" 声明

### fig6: S10 Power Sensitivity (44.2KB)
- 可读性: 5 子图 (a-d 环境 + e 汇总)，hollow markers 标注非显著 cell
- 审美: 最复杂的图，但布局合理
- 信息密度: 最高——60 cell 全矩阵可视化
- 结论承载力: 支持 L345 的 59/60 显著性声明和非单调性结论

### fig7: NS3 Trend Panel (32.5KB)
- 可读性: delta (percentage points) 随 node count 变化，hollow markers 标注非显著
- 审美: 简洁专业
- 结论承载力: 直接支持 L411 的 25/28 声明

**图表总评: 8 张图整体质量达到 Sensors 投稿标准。fig6 信息密度最高且设计最精细。fig0 可考虑压缩。所有图均使用 PDF 矢量格式，打印质量有保障。**

---

## Reviewer Decisions (R1-R5)

### R1 方法学保守派

**结论: Minor Revision**

Top 3 问题:

1. CAS 公式权重来源不明 (P1-4)
   - 证据: tex L76-81，Eq.3 的 $w_{1,m}$, $b_m$ 等参数无来源说明
   - 最小修复: 加一句权重确定方式

2. 证据分层设计合理但 S8 非物理趋势未充分解释机理
   - 证据: tex L222，仅说 "plausible cause is MAC-layer contention penalties omission"
   - 最小修复: 在 Discussion 中加 2-3 句解释为何 S8 仍有报告价值（历史基线、大样本统计功效）

3. Related Work 缺少 baseline 选择理由
   - 证据: tex L51-54，未说明为何选这四个经典协议而非近期 RL 方法
   - 最小修复: 加 1 句 baseline 选择依据

### R2 统计学严格派

**结论: Minor Revision**

Top 3 问题:

1. 100-node 核心表缺显著性标注 (P1-5)
   - 证据: tex L159-163，tab:pdr100 只有 mean±std
   - 最小修复: 加脚注或星号标注 Holm-corrected p

2. Hedges' g 值极端 (g=180) 的解读虽有脚注但仍不够充分
   - 证据: tex L244 脚注，g=180.0377 (outdoor_urban, L236)
   - 当 n=1000 且组内方差极小时，g 失去跨研究可比性。脚注已提及但未引用 Cohen/Sawilowsky 的效应量解释框架
   - 最小修复: 在脚注中加一句引用 (如 Sawilowsky 2009 的 "huge" 阈值 = 2.0)

3. S9 patch n=1000 vs control n=600 的不等样本量设计缺乏 power analysis 说明
   - 证据: tex L275，"Patch cells use n=1000 and control cells use n=600"
   - tex L462 解释为 "phased execution"，但未说明 power 是否足够
   - 最小修复: 加一句 "At n=600, Welch's test achieves >99% power for detecting ΔPDR≥0.01 at α=0.05"

### R3 工程复现派

**结论: Minor Revision**

Top 3 问题:

1. 仿真器代码未开源，Data Availability 仅承诺 "upon final publication"
   - 证据: tex L482
   - 最小修复: 提供匿名 GitHub 链接或 Zenodo DOI

2. S8 frozen matrix 的 "frozen" 含义不够精确
   - 证据: tex L108，"frozen S8 regime is the original simulator path"
   - 审稿人会问: frozen 是指代码版本锁定？seed 锁定？参数锁定？
   - 最小修复: 加一句 "frozen at commit [hash], with deterministic seeds 42001-43000"

3. NS3 实现与 Python 实现的差异未量化
   - 证据: tex L385-386 说 "trend check rather than numerical-equivalence proof"，但未说明两个平台的具体实现差异
   - `NS3_CLAIM_GATE.md` L60-68 有详细说明但未反映在正文中
   - 最小修复: 在 Limitations 中加一句 "NS-3 uses simplified protocol models without full IEEE 802.15.4 MAC emulation"

### R4 应用价值导向派

**结论: Accept (with minor suggestions)**

Top 3 问题:

1. Deployment Guidance 表 (tab:deployment_summary) 缺少定量阈值
   - 证据: tex L444-448，建议均为定性描述 ("gains are small", "remains preferable")
   - 最小修复: 在 caveat 列加入关键数值（如 "ΔPDR < 0.01 at 1000 nodes"）

2. 能耗-寿命 trade-off 讨论不足
   - 证据: tex L424 仅一句 "lower total energy can co-occur with shorter lifetime"
   - fig4 展示了四维 trade-off 但正文未充分解读
   - 最小修复: 在 Discussion 中加 2 句解读 fig4 的能耗-寿命关系

3. 实际部署场景映射缺失
   - 证据: 四个环境名称 (indoor_office 等) 来自信道模型，但未说明对应哪些真实部署场景
   - 最小修复: 在 Section 4.1 加一句映射说明（如 "indoor_office corresponds to ITU-R P.1238 office propagation"）

### R5 版面与图表美学审稿人

**结论: Accept**

Top 3 问题:

1. fig0 (workflow) 占整页但信息密度偏低
   - 证据: `fig0_aeris_workflow_20260220_s42.pdf`，30.8KB 矢量图
   - 最小修复: 缩至 0.7\textwidth 或移至附录

2. 表格数量偏多 (11 张表 + 8 张图 = 19 个浮动体)
   - 证据: v42 tex 共 11 个 table 环境
   - Sensors 对此无硬性限制，但密度较高
   - 最小修复: 可选——将 tab:rigor_patch (L256) 和 tab:s9_pegasis_1000 (L303) 合并或移至附录

3. tab:deployment_summary (L435) 的列宽分配可优化
   - 证据: tex L440，三列宽度 0.24/0.42/0.26，中间列偏宽
   - 最小修复: 微调为 0.22/0.40/0.30

---

## 给作者的中文邮件 (四封)

### 邮件 1: Reject 模板

主题: 关于稿件 AERIS 的审稿意见

作者您好，

感谢您向 Sensors 投稿。经审稿组评议，本稿存在以下核心问题导致无法接收：

1. S8 scalability matrix 存在非物理上升趋势（3/4 环境），且作者承认 MAC 碰撞建模缺失。在此前提下，大规模可扩展性结论的可信度不足。
2. S11 matched matrix 显示 AERIS 在所有 24 个 cell 中 patch-control delta 均为负且显著，说明在更严格物理假设下协议性能反而下降，这与"可靠性导向"的定位存在张力。
3. NS-3 验证仅限 AERIS vs LEACH 的趋势级比较，未覆盖全部五协议，验证深度不足。

建议作者在完成 MAC 碰撞模型校准后重新提交。

此致

### 邮件 2: Major Revision 模板

主题: 关于稿件 AERIS 的审稿意见——大修

作者您好，

本稿在实验设计的透明度和证据分层方面有明显优势，但以下问题需要实质性修改：

1. CAS 模式选择公式的权重来源未说明（Eq.3），需补充参数确定方法和敏感性分析。
2. 100-node 核心比较表缺少显著性标注，需补充 Holm-corrected p 值。
3. Related Work 需加强，说明 baseline 选择理由并讨论与近期 RL/DRL 方法的关系。
4. NS-3 与 Python 平台的实现差异需在正文中量化说明。

请在修改稿中逐条回复上述意见。

此致

### 邮件 3: Minor Revision 模板

主题: 关于稿件 AERIS 的审稿意见——小修

作者您好，

本稿在实验设计、证据分层和结论边界控制方面表现出色，数据交叉验证全部通过。以下小修意见请在修改稿中处理：

1. 请在 100-node 比较表下方补充显著性标注（Holm-corrected p 值或星号）。
2. 请在 Eq.3 后补充 CAS 权重的确定方式（一句即可）。
3. 请在 Related Work 末尾加一句 baseline 选择理由。
4. 建议在 Limitations 中补充 NS-3 实现简化说明。
5. 证据文档 NS3_ALIGNMENT_EVIDENCE.md 第 134 行的 "26/28" 需更正为 "25/28"。

以上均为文字层面修改，不涉及补实验。期待修改稿。

此致敬礼

### 邮件 4: Accept 模板

主题: 关于稿件 AERIS 的审稿意见——接收

作者您好，

本稿在以下方面达到了 Sensors 的发表标准：

1. 实验设计透明，证据分层清晰（S8/S9/S10/S11/NS3 五个 regime 各有明确角色）。
2. 统计方法规范（Welch t-test + Holm correction + Hedges' g），且对大样本效应量有专门脚注说明。
3. 结论边界控制严格，未发现过度声称。
4. 数据交叉验证全部通过，图表质量达标。

建议在 camera-ready 阶段补充 CAS 权重说明和 100-node 表显著性标注。

恭喜，建议接收。

此致敬礼

---

## 导师视角修改建议

### 24小时内必须完成
1. 修复 `NS3_ALIGNMENT_EVIDENCE.md` L134 的 "26/28" → "25/28" (P1-1)
2. 同步 `NS3_ALIGNMENT_EVIDENCE.md` Section 3.1 表格数据与 `ns3_scale_ext_1000_stats.csv` (P1-2)
3. 在 tab:pdr100 下方加显著性脚注 (P1-5)

### 72小时内应完成
4. Eq.3 后补 CAS 权重来源说明 (P1-4)
5. Related Work 末尾加 baseline 选择理由 (P1-3)
6. Limitations 中加 NS-3 实现简化说明 (R3-3)
7. tab:robust_snapshot subsection 标题修正 (P2-5)

### 1周内可增强
8. Deployment Guidance 表加定量阈值 (R4-1)
9. Discussion 中补充 fig4 能耗-寿命解读 (R4-2)
10. fig0 压缩或移至附录 (P2-4)
11. Data Availability 加仓库 URL (P2-7)

---

## Gate 判定

### 是否允许继续投稿？

**允许。** v42 无 P0 阻断项，数据交叉验证全部通过，禁写断言扫描 0 命中。5 项 P1 均为文字层面修改，不涉及补实验或重跑数据。

### 是否需要补实验？

**不需要。** 当前证据链完整：
- 100-node 核心比较 (n=30, 4 env × 5 protocols)
- S8 frozen scalability (n=1000, 4 env × 6 scales × 5 protocols)
- S9 patch-control stress (n=1000/600)
- S10 tx-power sensitivity (n=600, 60 cells)
- S11 matched patch-control (n=1000/1000, 24 cells)
- NS3 trend validation (n=30, 28 cells, 25/28 significant)

所有实验 regime 均已完成且数据一致。无需补跑。

### 最小修改清单 (按优先级，共 11 条)

| # | 优先级 | 内容 | 涉及文件 | 类型 |
|---|---|---|---|---|
| 1 | 24h | NS3_ALIGNMENT_EVIDENCE L134: 26/28→25/28 | ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md | 文档修复 |
| 2 | 24h | NS3_ALIGNMENT_EVIDENCE Section 3.1 表格数据同步 | ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md | 文档修复 |
| 3 | 24h | tab:pdr100 下方加显著性脚注 | v42.tex L168 | tex 修改 |
| 4 | 72h | Eq.3 后加 CAS 权重来源说明 | v42.tex L81 | tex 修改 |
| 5 | 72h | Related Work 加 baseline 选择理由 | v42.tex L54 | tex 修改 |
| 6 | 72h | Limitations 加 NS-3 实现简化说明 | v42.tex L460 | tex 修改 |
| 7 | 72h | tab:robust_snapshot subsection 标题修正 | v42.tex L225 | tex 修改 |
| 8 | 1w | Deployment Guidance 表加定量阈值 | v42.tex L444 | tex 修改 |
| 9 | 1w | Discussion 补 fig4 能耗-寿命解读 | v42.tex L424 | tex 修改 |
| 10 | 1w | fig0 压缩或移至附录 | v42.tex L88 | 版面优化 |
| 11 | 1w | Data Availability 加仓库 URL | v42.tex L482 | tex 修改 |

---

*报告生成: Claude 4.6 (Opus), 2026-02-20*
*审稿文件: AERIS_Sensors_MDPI_Submission_Draft_20260220_v42.tex (487 lines, 8 figures)*
*数据验证: S8 20/20, S10 8/8, NS3 8/8, S8-sig 4/4, S10-sig 59/60, bib 17/17*

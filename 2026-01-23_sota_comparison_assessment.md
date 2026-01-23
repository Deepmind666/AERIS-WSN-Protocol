# 2026-01-23 SOTA 对比图修订与审查记录

## 目的
对 `sota_comparison_6panel` 图进行严格修正（添加 TEEN、拆分 AERIS-E/AERIS-R），并以审稿人视角评估结果可信度与表达清晰性。

## 数据与脚本
- 运行脚本：`scripts/run_sota_comparison.py`
- 作图脚本：`scripts/generate_sota_figures.py`
- 结果数据：`results/sota_comparison.json`
- 输出图：`for_submission/figures/sota_comparison_6panel.pdf`（同步生成 PNG/SVG）

## 关键变更（已实施）
1. **协议集合扩展至 7 个**  
   `LEACH / HEED / PEGASIS / SEP / TEEN / AERIS-E / AERIS-R`
2. **统一能耗模型与包长**  
   CC2420 统一能耗模型，`PACKET_SIZE_BITS = 4000`（500 bytes）。
3. **AERIS 拆分为两种配置**  
   - AERIS-E（energy profile）  
   - AERIS-R（robust profile）
4. **图 (c) 改为 ΔPDR 的可解释比较**  
   使用 `ΔPDR (AERIS profile − protocol, pp)` + 95% CI。
5. **图 (f) 表格放大并精简字段**  
   表头改为：`Protocol / ΔPDR-E / p_E / ΔPDR-R / p_R`，去除“vs”等冗余前缀。

## 当前主要结果（n=30 runs, 200 rounds, 54 nodes）
- **AERIS-R**：PDR ≈ 99.93%，能耗最高（≈ 21.31 J）
- **PEGASIS**：PDR ≈ 96.38%，能耗 ≈ 18.80 J
- **LEACH/HEED/SEP**：PDR ≈ 87–89%，能耗 ≈ 18.2–18.5 J
- **AERIS-E**：PDR ≈ 81.49%，能耗 ≈ 19.36 J
- **TEEN**：PDR ≈ 57.53%，能耗最低（≈ 15.24 J）

## 严格审稿视角的问题与风险
1. **TEEN 的 PDR 过低**  
   TEEN 为事件驱动协议，低 PDR 可能反映“触发条件过严/报文更少”，与持续上报型协议不完全同类。  
   - 风险：审稿人质疑公平性或语义一致性。  
   - 建议：在正文明确 TEEN 的事件驱动机制，并说明该比较的边界与局限。

2. **AERIS-E 在 PDR 上明显落后**  
   AERIS-E 目前在 PDR 上输给 LEACH/HEED/PEGASIS/SEP。  
   - 风险：若论文核心卖点是“可靠性提升”，必须区分 AERIS-E 的定位（节能优先）。  
   - 建议：主结论聚焦 AERIS-R 的可靠性提升，AERIS-E 作为能耗优先配置补充说明。

3. **图 (d) 信息密度仍偏高**  
   已改为误差棒 + 图例下置，但仍需检查文字与点是否遮挡。  
   - 建议：若遮挡仍明显，可进一步扩大画布或缩小图例字号。

4. **显著性数值极小（p < 1e-30）**  
   统计显著度极高，容易被认为“过拟合或条件过理想”。  
   - 建议：正文仅保留“显著性成立”的结论，避免过度渲染数值。

## 当前结论（是否可用于论文）
可以作为“统一模型下的 SOTA 基线对比图”。但需在正文清楚区分：
- AERIS-R：可靠性提升为主
- AERIS-E：能耗优先但牺牲可靠性
- TEEN：事件驱动协议，语义不同，不能与持续上报策略直接等价比较

## 后续建议
- 若希望 TEEN 更可比：调整阈值或定义“有效上报”标准，并在附录解释。
- 如果强调可靠性，需要补充 AERIS-R 在更多场景下的复现实验与压力测试。

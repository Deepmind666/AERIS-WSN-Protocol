# Claude服务器任务提示词（S9-Rigor-v2）

你现在只负责服务器实验，不改论文正文。

## 任务边界
- 允许：运行实验、导出统计、生成provenance、更新NS-3证据文档。
- 禁止：修改 `for_submission/*.tex` 与本地绘图主线脚本。

## 任务1（P0）：NS-3补强矩阵

矩阵：4环境 x 节点{100,500,1000} x 协议{AERIS,LEACH} x n=30

输出文件：
- `ns3_validation/results/ns3_rigor_v2_raw_20260215.json`
- `ns3_validation/results/ns3_rigor_v2_stats_20260215.csv`
- `ns3_validation/results/ns3_rigor_v2_significance_20260215.csv`
- `ns3_validation/results/ns3_rigor_v2_raw_20260215.provenance.json`

统计方法：Welch t-test + Hedges' g + Holm-Bonferroni。
主指标：`pdr_expected`。

## 任务2（P1）：可选第二baseline

若资源允许，追加 HEED：
4环境 x 节点{100,500,1000} x AERIS vs HEED x n=30。
输出同任务1命名规则追加 `*_heed_*`。

## 任务3（P0）：门控文档更新

更新：
- `ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md`
- `ns3_validation/results/NS3_CLAIM_GATE.md`

要求：
1) 明确 `trend-level only`；
2) 明确不可写 `numerical equivalence`；
3) 写清显著/不显著的环境-节点单元。

## 资源策略（服务器）
- CPU目标：80%-90%
- 内存上限：85%（持续超过90%必须降worker）
- 日志汇报间隔：20-30分钟

## 回报格式（必须）
1. 文件清单（完整路径）
2. 本次完成
3. 仍需核对
4. 当前ETA与依据（速率/完成比例）

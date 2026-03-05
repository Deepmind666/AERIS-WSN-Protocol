# AERIS阶段性进展汇报（给导师）

Date: 2026-02-15

## 1. 当前阶段结论

- 已完成 Sensors 模板主稿阶段版：
  - `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260215_v18.tex`
  - `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260215_v18.pdf`
- 已完成统一大规模可扩展性矩阵（S8）：
  - 4 环境 x 6 节点规模 x 5 协议 x n=1000/cell
- NS-3 已有趋势级交叉验证（AERIS vs LEACH），但尚未达到“数值等价验证”层级。

## 2. 目前可支撑的核心发现

1. 在 100 节点多环境 n=30 评测中，AERIS 在四环境 PDR 均值最高（限定于测试基线集合）。
2. 在 S8 统一可扩展性矩阵中（100-1000 节点，n=1000/cell），AERIS 在所有环境-规模 cell 的均值均为第一。
3. 消融显示 Gateway 在恶劣环境贡献更显著，CAS边际效应具有环境依赖性，不可写成统一正增益。
4. NS-3 当前结论仅限 trend-level，不声称数值级对齐完成。

## 3. 当前不足与风险

1. 仿真严谨性补丁（MAC竞争建模 + baseline公平增强）尚在执行计划中，未全部落地。
2. NS-3 仍以 AERIS vs LEACH 为主，五协议跨平台闭环不完整。
3. 需继续压实“可写/禁写”门控，避免超范围结论。

## 4. 下一阶段计划（并行）

- 本地（Codex）：
  1) 持续精修论文和图表，保持 claim 与证据一致；
  2) 小矩阵 smoke/pilot 验证仿真严谨性补丁。
- 服务器（Claude）：
  1) 运行长时矩阵与NS-3扩展实验；
  2) 产出 provenance + significance 证据包。

## 5. 对投稿状态的实事求是判断

- 现在可以用于“阶段汇报”和“导师讨论稿”。
- 距离正式投稿终版还差一步：
  1) 完成仿真严谨性补丁并给出门控通过证据；
  2) 补强 NS-3 交叉验证覆盖面。

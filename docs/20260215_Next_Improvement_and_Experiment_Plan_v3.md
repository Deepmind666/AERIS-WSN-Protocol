# 下一阶段改进与实验计划（v3）

Date: 2026-02-15
Owner: Codex（本地） + Claude（服务器）

## 目标

在保持当前可汇报稿可用的前提下，完成“仿真严谨性补强 + NS-3证据补强”，把稿件从阶段稿推进到可投稿终稿。

## A. 本地任务（Codex）

## A1 论文主线精修（优先）
- 基线文件：`for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260215_v19.tex`
- 动作：
  1) 逐段执行 claim gate 对齐；
  2) 图注、表注、统计口径统一；
  3) 限制性描述集中到 Discussion + Limitations。
- 输出：
  - `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260215_v20.tex`
  - 对应 PDF 预览。

## A2 仿真严谨性补丁 smoke/pilot（本地小规模）
- 只跑小矩阵验证逻辑，不做全量：
  - 4环境 x 节点{100,500,1000} x 5协议 x n=20
- 目标：
  1) 检查 PDR-Scale 单调性是否改善；
  2) 检查 baseline 公平性增强后结论是否稳定；
  3) 评估是否值得进入 full rerun。

## A3 本地资源策略
- CPU 目标 65%，硬上限 80%
- MEM 目标 70%，硬上限 80%
- 超阈值降 worker，禁止满载抢占。

## B. 服务器任务（Claude）

## B1 NS-3补强（优先）
- 矩阵：4环境 x 节点{100,500,1000} x AERIS vs LEACH x n=30
- 可选扩展：AERIS vs HEED（同矩阵）
- 输出：raw/stats/significance/provenance 四件套。

## B2 生产级长跑（待A2通过后）
- 4环境 x 6节点 x 5协议 x n=1000 全量重跑
- 只在 A2 门槛通过后启动，避免无效大算力开销。

## B3 服务器资源策略
- CPU 目标 80%-90%
- MEM 上限 85%，持续超过 90% 必须降 worker
- 每 20-30 分钟汇报一次 ETA 与完成比例。

## C. Go/No-Go 门槛

进入 full rerun 之前，必须同时满足：
1. smoke/pilot 中不存在明显物理反常（规模增大导致普遍异常升PDR）；
2. 关键结论方向稳定（AERIS 在恶劣环境仍显著领先）；
3. NS-3 trend-level 与 Python 方向不冲突；
4. claim gate 全绿（禁写语句 0 命中）。

## D. 里程碑

- M1（今天）: v19阶段稿 + s23图组完成（已达成）
- M2（1-2天）: NS-3补强首轮统计完成
- M3（2-4天）: smoke/pilot 判定与 full rerun 决策
- M4（后续）: full rerun + 终稿收敛

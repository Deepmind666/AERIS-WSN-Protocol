# 20260226 v70 Parallel Handoff (Codex + Claude)

## 本地（Codex）
- 已启动脚本：`scripts/run_local_s10r_queue_20260226.ps1`
- PID：`105852`（父进程）
- 当前执行：`indoor_office tx5`（S10R 任务 1/6）
- 日志：`logs/s10r_local_queue_20260226_115629.log`
- 最新吞吐（日志）：`8780/30000, rate≈13.0/s, ETA≈27min`（仅当前任务）

### 本地执行参数
- env: indoor_office + indoor_factory
- tx: 5 / 10 / 15
- nodes: 100,200,300,500,800,1000
- protocols: AERIS, LEACH, PEGASIS, HEED, TEEN
- replicates: 1000
- run-tier: publication
- flags: --mac-collision --multihop-relay

## 服务器（Claude）
- 请直接使用：`docs/20260226_Claude_Server_S10R_Prompt.md`（v2）
- 任务范围：outdoor_urban + outdoor_suburban（tx 5/10/15）
- 每任务验收：raw_results=30000, error_runs=0, run_tier=publication, primary_metric=pdr_expected

## 后处理入口（已准备）
- 四环境 S10R 合并：`scripts/postprocess_s10r_4env.py`
  - 输出：
    - `results/mega_experiments/s10r_4env_merged_descriptive_20260226.csv`
    - `results/mega_experiments/s10r_4env_significance_tx5_vs_tx10_vs_tx15_20260226.csv`
    - `results/mega_experiments/s10r_4env_reconciliation_20260226.md`

- NS-3 五协议全节点重算（已执行完成）：`scripts/recompute_ns3_5proto_significance_fullnodes.py`
  - 已产出：
    - `ns3_validation/results/ns3_5proto_fullnodes_descriptive_20260226.csv`
    - `ns3_validation/results/ns3_5proto_fullnodes_significance_20260226.csv`
    - `ns3_validation/results/ns3_5proto_fullnodes_recalc_report_20260226.md`

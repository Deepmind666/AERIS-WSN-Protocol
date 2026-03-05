# 2026-02-11 本地 Fix2 实验收口报告（Codex）

## 1. 运行信息
- 脚本：`scripts/run_scalability_fix2_local_2env.ps1`
- 参数：`Replicates=60, Workers=14, Nodes=100,200,300,500,800,1000, Rounds=300`
- 资源上限：`CPU<=80%, MEM<=80%`
- 环境：`indoor_factory`、`outdoor_urban`
- Manifest：`results/mega_experiments/scalability_fix2_local_2env_20260211_133833_manifest.json`

## 2. 完成情况
- `indoor_factory`：`1800/1800`，`failed=0`，`exit_code=0`，耗时 `1496s`
- `outdoor_urban`：`1800/1800`，`failed=0`，`exit_code=0`，耗时 `1079s`
- 总耗时：`2575s`（约 `42m55s`）

## 3. 输出文件
- `results/mega_experiments/scalability_fix2_indoor_factory_20260211_133833.json`
- `results/mega_experiments/scalability_fix2_outdoor_urban_20260211_133833.json`
- `results/mega_experiments/scalability_fix2_local_2env_20260211_133833_manifest.json`

## 4. 本地统计结果（两环境）
由 `scripts/aggregate_scalability_fix2_local2env.py` 生成：
- `results/mega_experiments/scalability_fix2_local2env_20260211_descriptive.csv`
- `results/mega_experiments/scalability_fix2_local2env_20260211_significance.csv`
- `results/mega_experiments/scalability_fix2_local2env_20260211_summary.md`

关键观察（AERIS vs LEACH）：
- `indoor_factory`: 各节点规模差值约 `+0.77 ~ +0.79`，Holm 显著。
- `outdoor_urban`: 各节点规模差值约 `+0.71 ~ +0.82`，Holm 显著。

## 5. 结论与门控
- 本地两环境 fix2 数据完整，资源控制满足约束。
- 该结果仅覆盖 2 环境，不能单独支撑“四环境总声明”。
- 主稿结论仍需等待服务器补齐的 fix550 两环境并完成四环境统一统计。

## 6. 下一步
1) 等待服务器补齐 `indoor_factory/outdoor_urban` fix550 结果与 sidecar。  
2) 重建四环境统一统计（descriptive + significance + claim gate）。  
3) 以统一四环境源更新 NS3/Python 对照表与 Section 6/8 文稿引用。  


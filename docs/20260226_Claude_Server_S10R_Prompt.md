# 20260226 Claude Server Prompt (S10R + NS3 sanity, v2)

你现在是 AERIS 项目的服务器执行同事。请严格按下面规则执行，不越权。

## 0) 首条回执（必须三句）
1. 已恢复并将全程中文输出。
2. 已锁定本任务范围（只做服务器实验与结果核验，不改 src 核心算法）。
3. 将先检查服务器占用，再按队列顺序执行。

## 1) 强制规则
- 仅使用 `ssh FatMachine`。
- 禁止在本地执行服务器任务。
- 禁止改动 `src/`、`for_submission/*.tex`。
- 只允许新增 `results/mega_experiments/` 下实验产物与 `docs/` 下交付报告。
- 若服务器忙，排队执行，不抢占其他任务。
- 每次汇报必须包含 ETA 区间和依据（最近一段吞吐/已完成任务数）。

## 2) 本轮目标
为 Figure 6 补齐更密集矩阵（2 环境由你负责）：
- 环境：`outdoor_urban`, `outdoor_suburban`
- 发射功率：`tx=5,10,15`
- 节点：`100,200,300,500,800,1000`
- 协议：`AERIS,LEACH,PEGASIS,HEED,TEEN`
- 重复：`replicates=1000`
- 物理开关：`--mac-collision --multihop-relay`
- 其他：`--run-tier publication --seed 42001 --rounds 300`

每个 JSON 目标行数：`30000`。

## 3) 执行顺序（串行）
1. outdoor_urban tx5
2. outdoor_urban tx10
3. outdoor_urban tx15
4. outdoor_suburban tx5
5. outdoor_suburban tx10
6. outdoor_suburban tx15

## 4) 标准命令模板（Windows 服务器，绝对 Python 路径）
```powershell
"C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe" scripts/run_scalability_experiment.py \
  --env <ENV> --tx-power <TX> \
  --replicates 1000 --nodes 100,200,300,500,800,1000 \
  --rounds 300 --workers 20 --run-tier publication \
  --mac-collision --multihop-relay \
  --max-cpu-percent 90 --max-mem-percent 96 --allow-partial \
  --output results/mega_experiments/scalability_<ENV>_server_s10r_tx<TX>_20260226.json
```

## 5) 每个任务完成后必须产出（逐任务回报）
- 对应 JSON
- 对应 provenance sidecar（含 `data_sha256`, `git_commit`, `script_sha256`, `config_hash`）
- 一行完整性检查结果：
  - raw_results==30000
  - error_runs==0
  - run_tier==publication
  - primary_metric==pdr_expected

## 6) 批次完成后必须产出
1. `results/mega_experiments/s10r_server_2env_merged_descriptive_20260226.csv`
2. `results/mega_experiments/s10r_server_2env_significance_tx5_vs_tx10_vs_tx15_20260226.csv`
3. `results/mega_experiments/s10r_server_2env_reconciliation_20260226.md`

## 7) NS-3 sanity（只核验，不重跑）
读取：
- `ns3_validation/results/ns3_5proto_summary.json`
- `ns3_validation/results/ns3_5proto_significance.json`
输出：
- `docs/20260226_ns3_5proto_sanity_check.md`
内容必须说明：
- 是否覆盖 5 协议（AERIS/LEACH/PEGASIS/HEED/TEEN）
- 各环境与节点覆盖范围
- 为什么论文中的 NS-3 主图只画 AERIS vs LEACH（趋势验证边界）

## 8) 回报模板（每轮固定）
1. 文件清单
2. 本次完成
3. 仍需核对

并附元数据表：
`environment, tx_power, raw_results, error_runs, run_tier, primary_metric, git_commit, data_sha256`

## 9) 失败处理
- 若某任务失败：只重跑失败任务，不重跑已通过任务。
- 若服务器忙：报告“已排队”，并给下一检查时间（绝对时间，如 `2026-02-26 14:30`）。
- 不允许 silent failure。

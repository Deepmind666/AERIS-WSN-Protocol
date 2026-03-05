请严格审查以下内容：

文件清单：
- results/mega_experiments/scalability_outdoor_urban_server_s10r_tx*_20260226.json
- results/mega_experiments/scalability_outdoor_suburban_server_s10r_tx*_20260226.json
- 对应全部 .provenance.json
- results/mega_experiments/s10r_server_2env_reconciliation_20260226.csv
- results/mega_experiments/s10r_server_2env_reconciliation_20260226.md

本次完成：
1) 验收每个 JSON 是否满足：
   - raw_results = 30000
   - error_runs = 0
   - run_tier = publication
   - primary_metric = pdr_expected
2) 校验 sidecar 的 data_sha256 / script_sha256 / config_hash / git_commit 完整性
3) 输出两环境三功率覆盖完整性矩阵（缺口=0）

仍需核对：
1) 与本地 indoor_office / indoor_factory 合并后的 4 环境一致性（节点与协议维度）
2) 合并后显著性表中是否存在异常空 cell 或 n 不一致

固定回报格式：
1. 文件清单
2. 本次完成
3. 仍需核对
并附元数据表：
environment, tx_power, raw_results, error_runs, run_tier, primary_metric, git_commit, data_sha256

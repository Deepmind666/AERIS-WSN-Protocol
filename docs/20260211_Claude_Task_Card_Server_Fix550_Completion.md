# 给 Claude4.6 的任务卡（服务器侧，补齐 fix550 四环境）

## 执行边界（必须）
1. 仅执行本任务卡，不新增实验。
2. 不修改 src 核心算法代码。
3. 全程中文输出；每次进度回报必须包含 ETA。
4. 资源上限：CPU <= 80%，MEM <= 80%。
5. 结果必须可追溯：git_commit、git_dirty、git_diff_stat、script_sha256、config_hash。

## 目标
补齐 fix550 四环境证据链中缺失的两个环境（indoor_factory, outdoor_urban），与已完成的 indoor_office/outdoor_suburban 合并成统一可投稿数据集。

---

## 任务 S1（P0）：运行 2 环境 fix550

### 命令（服务器）
对 indoor_factory 执行：
python scripts/run_scalability_experiment.py --replicates 550 --workers 12 --seed 42001 --nodes 100,200,300,500,800,1000 --rounds 300 --env indoor_factory --tx-power 10.0 --run-tier publication --max-cpu-percent 80 --max-mem-percent 80 --resource-check-sec 2.0 --output results/mega_experiments/scalability_indoor_factory_server_fix550_20260211.json

对 outdoor_urban 执行：
python scripts/run_scalability_experiment.py --replicates 550 --workers 12 --seed 42001 --nodes 100,200,300,500,800,1000 --rounds 300 --env outdoor_urban --tx-power 10.0 --run-tier publication --max-cpu-percent 80 --max-mem-percent 80 --resource-check-sec 2.0 --output results/mega_experiments/scalability_outdoor_urban_server_fix550_20260211.json

### 验收
每个环境必须满足：
- raw_results = 16500（550 * 6 * 5）
- error_runs = 0
- primary_metric = pdr_expected
- run_tier = publication

---

## 任务 S2（P0）：生成 provenance sidecar

### 命令（服务器）
python scripts/generate_scalability_provenance.py --results results/mega_experiments/scalability_indoor_factory_server_fix550_20260211.json
python scripts/generate_scalability_provenance.py --results results/mega_experiments/scalability_outdoor_urban_server_fix550_20260211.json

### 验收
每个 JSON 均有对应 _provenance.json，且包含：
- git_commit
- git_dirty
- git_diff_stat
- script_sha256（64 hex）
- config_hash
- run_tier
- primary_metric

---

## 任务 S3（P1）：生成四环境合并统计（仅基于 fix550）

### 输入文件（四环境）
1) results/mega_experiments/scalability_indoor_office_server_fix550_20260210.json  
2) results/mega_experiments/scalability_outdoor_suburban_server_fix550_20260210.json  
3) results/mega_experiments/scalability_indoor_factory_server_fix550_20260211.json  
4) results/mega_experiments/scalability_outdoor_urban_server_fix550_20260211.json

### 输出文件
1) results/mega_experiments/scalability_4env_fix550_20260211_descriptive.csv  
2) results/mega_experiments/scalability_4env_fix550_20260211_significance.csv  
3) results/mega_experiments/scalability_4env_fix550_20260211_manifest.json

### 统计要求
- 比较：AERIS vs LEACH/PEGASIS/HEED/TEEN
- 指标：pdr_expected
- 检验：Welch t-test + Hedges g + Holm

---

## 回报模板（必须）
1. 文件清单（完整路径）  
2. 本次完成（含每一步耗时）  
3. 仍需核对  
4. 当前累计耗时 + ETA  


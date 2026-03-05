# Claude 任务卡（Scalability Fix2，服务器侧）

## 目标
在服务器侧运行修复后可扩展性实验（与本地同口径、同脚本），补齐本地未跑的 2 个环境，并生成 provenance。

## 输入与版本
- 代码基线：请先 `git pull` 到与本地一致的最新分支状态。
- 脚本：
  - `C:\AERIS-WSN-Protocol\scripts\run_scalability_experiment.py`
  - `C:\AERIS-WSN-Protocol\scripts\generate_scalability_provenance.py`
- 运行口径：
  - `primary_metric = pdr_expected`
  - `run_tier = publication`
  - 资源上限：`CPU <= 80%`，`MEM <= 80%`

## 任务 A（必做）
仅运行以下两个环境（本地已在跑 indoor_factory/outdoor_urban）：
- `indoor_office`
- `outdoor_suburban`

固定参数：
- `replicates=60`
- `workers=12`（服务器可根据稳定性微调到 14，但必须汇报）
- `nodes=100,200,300,500,800,1000`
- `rounds=300`
- `seed=42001`
- `tx_power=10`

建议命令（逐环境）：
```bash
python scripts/run_scalability_experiment.py \
  --replicates 60 --workers 12 --seed 42001 \
  --nodes 100,200,300,500,800,1000 --rounds 300 \
  --env indoor_office --tx-power 10 --run-tier publication \
  --max-cpu-percent 80 --max-mem-percent 80 --resource-check-sec 2
```

```bash
python scripts/run_scalability_experiment.py \
  --replicates 60 --workers 12 --seed 42001 \
  --nodes 100,200,300,500,800,1000 --rounds 300 \
  --env outdoor_suburban --tx-power 10 --run-tier publication \
  --max-cpu-percent 80 --max-mem-percent 80 --resource-check-sec 2
```

## 任务 B（必做）
对任务 A 生成的两个 JSON 各自执行 provenance 生成：

```bash
python scripts/generate_scalability_provenance.py --input <env_json_path>
```

## 验收标准
1. 每个环境输出 `raw_results = 1800`（= `60 * 6 * 5`）。
2. `error_runs = 0`。
3. `run_tier = publication`，`primary_metric = pdr_expected`。
4. 每个 JSON 都有对应 `*.provenance.json`。
5. 回报必须包含：
   - 输出文件完整路径
   - 各环境耗时与总耗时
   - 关键元数据：`git_commit`, `git_dirty`, `git_diff_stat`, `script_sha256`, `config_hash`, `run_tier`, `primary_metric`
   - 各环境 `1000` 节点下 5 协议 PDR 排名

## 禁止事项
- 不修改 `src/` 协议算法代码。
- 不新增未指派实验。
- 不改论文文稿。

## 回报模板
- 文件清单
- 本次完成
- 仍需核对

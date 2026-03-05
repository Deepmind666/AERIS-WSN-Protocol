# Claude4.6 新会话重启提示词（AERIS：outdoor_urban + outdoor_suburban）

你现在是 AERIS-WSN-Protocol 项目执行同事。你只做本任务，不扩展。
目标：在不重复浪费算力的前提下，完成 outdoor_urban 与 outdoor_suburban 两个环境的 v50-rigor 证据收口，输出可锁版的服务器对账与冻结清单。

---

## 0) 强制规则（先执行）
1. 全程中文输出（compact/恢复后也必须中文）。
2. 修改任何代码/文档前，先给出：`路径 + 计划 + 影响范围`，等待批准。
3. 每次实验启动或状态变更必须给 ETA（剩余时间区间）。
4. 禁止高频轮询；默认 30~60 分钟里程碑汇报一次。
5. 只做本任务，不新增实验、不改论文正文。

---

## 1) 必读文件（按顺序）
1. `.claude/RULES.md`
2. `.codex/RULES.md`
3. `docs/20260223_v57_Refinement_Checklist.md`
4. `results/mega_experiments/server_reconciliation_v50rigor_20260222.md`
5. `results/mega_experiments/server_reconciliation_v50rigor_20260222.csv`
6. `results/mega_experiments/server_freeze_v50rigor_manifest_20260222.md`
7. `results/mega_experiments/scalability_outdoor_urban_v50rigor_20260222_server.json`
8. `results/mega_experiments/scalability_outdoor_urban_v50rigor_20260222_server.provenance.json`
9. `results/mega_experiments/scalability_outdoor_suburban_v50rigor_20260222_103921.json`
10. `results/mega_experiments/scalability_outdoor_suburban_v50rigor_20260222_103921.provenance.json`
11. `results/mega_experiments/scalability_4env_v50rigor_20260222_descriptive.csv`
12. `results/mega_experiments/scalability_4env_v50rigor_20260222_significance.csv`

---

## 2) 服务器连接方式（必须）
- 只使用：`ssh FatMachine`
- 禁止：裸 IP 登录（例如 `ssh admin@100.104.82.45`）
- Python 解释器：`C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe`
- 远程工作目录：`C:\Users\sshuser\AERIS-WSN\`

---

## 3) 任务A：先复核，不先重跑（优先级最高）
对以下两个文件做独立完整性复核：
- `scalability_outdoor_urban_v50rigor_20260222_server.json`
- `scalability_outdoor_suburban_v50rigor_20260222_103921.json`

复核项（6项必须全 PASS）：
1. `raw_results == 96000`
2. `error_runs == 0`
3. `run_tier == publication`
4. `primary_metric == pdr_expected`
5. 30 cells（6 nodes × 5 protocols）每个 cell `n == 3200`
6. JSON 的 SHA256 与 sidecar 的 `data_sha256` 一致

若 6/6 PASS：标记 PASS，不重跑。

---

## 4) 任务B：仅在失败时补跑（条件触发）
仅当任务A任意一项 FAIL，才补跑对应环境。命令固定：

```powershell
python scripts/run_scalability_experiment.py --env <outdoor_urban|outdoor_suburban> --replicates 3200 --seed 42001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 20 --run-tier publication --tx-power 10.0 --mac-collision --multihop-relay --allow-partial --output results/mega_experiments/scalability_<env>_v50rigor_<timestamp>_server.json
```

补跑后必须立刻生成 sidecar，并在回报中给：
- 数据 SHA256
- `git_commit`
- `script_sha256`
- `config_hash`

---

## 5) 任务C：统一对账与冻结清单更新
无论是否补跑，都更新并输出新时间戳版本：
1. `results/mega_experiments/server_reconciliation_v50rigor_<timestamp>.csv`
2. `results/mega_experiments/server_reconciliation_v50rigor_<timestamp>.md`
3. `results/mega_experiments/server_freeze_v50rigor_manifest_<timestamp>.md`

要求：
- 每个文件记录 `SHA256 + provenance 路径 + PASS/FAIL`
- 若出现 `git_commit=unknown`，必须写原因与后续处理建议

---

## 6) 最终回报格式（严格）
只允许三段：
1. `文件清单`
2. `本次完成`
3. `仍需核对`

并附：
- 关键元数据表：`env, raw_results, error_runs, run_tier, primary_metric, data_sha256`
- 若发生补跑：`总耗时 + ETA 复盘（预估 vs 实际）`

---

## 7) 禁止事项
- 禁止修改 `src/` 核心算法
- 禁止修改论文 tex/pdf
- 禁止启动未指派实验
- 禁止长篇解释和泛化结论

---

## 8) 任务完成判定
同时满足以下条件才算完成：
- outdoor_urban 与 outdoor_suburban 两环境 authoritative 文件 PASS
- reconciliation 与 freeze manifest 已更新为新时间戳
- 回报格式符合“三段式”，且元数据齐全

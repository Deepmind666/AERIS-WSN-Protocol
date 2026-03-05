# 给 Claude4.6 的任务卡（fix550 证据链统一，立即执行）

## 执行原则
1. 全程中文输出。
2. 不新增未指派实验。
3. 不修改 `src/` 核心算法。
4. 每次回报都给 ETA。

## 当前背景（已确认）
- `bf59e4a8..b6b2e5e` 仅差异文件：`docs/20260210_Codex_Claude_Worksplit.md`（docs-only）。
- 因此可将 `bf59e4a8` 与 `b6b2e5e` 视为“代码等价 commit”。

---

## 任务 A（P0）：出具 commit 等价证明

### 输入
- `git diff --name-only bf59e4a8..b6b2e5e`
- `git show --no-patch --oneline bf59e4a8`
- `git show --no-patch --oneline b6b2e5e`

### 输出
- `C:\AERIS-WSN-Protocol\results\mega_experiments\fix550_commit_equivalence_20260211.md`

### 必须包含
1. diff 文件列表  
2. 结论：代码层（src/scripts）是否有差异  
3. 是否可合并同一证据链（YES/NO）及理由

---

## 任务 B（P0）：补齐现有 fix550 的 provenance 缺口

### 输入文件
1) `results/mega_experiments/scalability_indoor_office_server_fix550_20260210.json`  
2) `results/mega_experiments/scalability_outdoor_suburban_server_fix550_20260210.json`  
3) `results/mega_experiments/scalability_outdoor_urban_fix550_20260210_102734.json`

### 操作
- 对缺失 sidecar 的文件执行：
  `python scripts/generate_scalability_provenance.py --results <json_path>`

### 输出
- 对应 `*_provenance.json` 文件
- `C:\AERIS-WSN-Protocol\results\mega_experiments\fix550_provenance_audit_20260211.csv`

### 审计字段
- environment
- json_file
- provenance_file
- git_commit
- git_dirty
- script_sha256_len
- config_hash_exists
- run_tier
- primary_metric

---

## 任务 C（P1）：准备四环境合并脚本（不跑最终合并）

### 目标
在 `indoor_factory_local_fix550` 到位后，可一键生成四环境统一统计。

### 输出
- `C:\AERIS-WSN-Protocol\scripts\merge_fix550_four_env.py`

### 脚本要求
1. 输入 4 个 JSON 路径（命令行参数）。  
2. 检查每个文件：`raw_results==16500`、`error_runs==0`、`primary_metric==pdr_expected`。  
3. 输出：
   - `scalability_4env_fix550_20260211_descriptive.csv`
   - `scalability_4env_fix550_20260211_significance.csv`
   - `scalability_4env_fix550_20260211_manifest.json`
4. 统计方法：Welch t-test + Hedges g + Holm。  
5. 本任务仅做脚本和 dry-run（可缺 1 个环境时返回“等待文件”）。

---

## 回报模板（必须）
1. 文件清单（完整路径）  
2. 本次完成  
3. 仍需核对  
4. 当前耗时 + ETA  


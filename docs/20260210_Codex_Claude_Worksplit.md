# 2026-02-10 本地-服务器分工任务单（Codex/Claude）

项目：AERIS-WSN-Protocol  
目标：在不超过资源阈值的前提下，补齐可扩展性与NS-3投稿门槛证据链。

## A. 资源与流程硬约束（本轮强制）

1. CPU <= 70%，内存 <= 70%。
2. 每次实验状态更新必须包含 ETA。
3. 任何实验回报必须含：git_commit, git_dirty, git_diff_stat, run_tier, primary_metric。
4. 服务器实验仅基于已 push commit 执行。

## B. 当前分工

### B1. Codex（本地）

任务：补跑缺失分片 outdoor_urban（n=550）。

命令（已启动）：
python scripts/run_scalability_experiment.py --nodes 100,200,300,500,800,1000 --replicates 550 --workers 10 --rounds 300 --env outdoor_urban --max-cpu-percent 65 --max-mem-percent 65 --run-tier publication --output results/mega_experiments/scalability_outdoor_urban_fix550_20260210_102734.json

验收：
- raw_results = 16500
- error_runs = 0
- incomplete_runs = 0

### B2. Claude（服务器）

任务 S1（立即执行）：同步代码到 commit `bf59e4a` 并回传环境确认。

步骤：
1) git fetch --all
2) git checkout bf59e4a
3) git rev-parse --short=8 HEAD

回报必须包含：
- 当前 commit
- conda 环境名
- python --version

任务 S2（本轮执行）：重跑两个服务器分片（资源受控版，替代旧的 100/90 配置）。

环境：indoor_office, outdoor_suburban

命令模板（每个环境一条）：
conda run -n aether-wsn python scripts/run_scalability_experiment.py --nodes 100,200,300,500,800,1000 --replicates 550 --workers 12 --rounds 300 --env <ENV> --max-cpu-percent 65 --max-mem-percent 65 --run-tier publication --output results/mega_experiments/scalability_<ENV>_server_fix550_20260210.json

任务 S3（完成 S2 后）：生成 provenance。

命令：
python scripts/generate_scalability_provenance.py --overnight-dir results/mega_experiments/<SERVER_OUTPUT_DIR>

任务 S4（并行文档任务，不跑新实验）：提交 NS-3 对齐执行清单。

输出文件：
docs/20260210_NS3_Publication_Gate_Checklist.md

内容最少包括：
- Python vs NS-3 参数逐项映射
- n>=30 seeds 的执行方案
- 允许写入论文的句式与禁写句式

## C. 统一回报模板（Codex/Claude都用）

文件清单
- <full_path_1>
- <full_path_2>

本次完成
1) ...
2) ...

当前阶段
- 阶段名：...
- 已用时：...
- 进度依据：<done>/<total>（来源：日志路径 + 时间戳）
- 预估剩余：...
- 预计完成：...

仍需核对
1) ...
2) ...

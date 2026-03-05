---
name: experiment
description: 实验启动前的标准化检查与执行协议。确认参数、资源、验收标准，防止遗漏元数据或误用环境。
---

# 实验启动协议

## 触发条件
用户输入 `/experiment` 或要求"启动实验"、"跑实验"时执行。

## 执行步骤

### Step 1: 参数确认（必做）
向用户确认以下参数，缺一不可：
- 目标机器：本地 / FatMachine
- 环境列表：indoor_office / indoor_factory / outdoor_urban / outdoor_suburban
- 功率列表：tx5 / tx10 / tx15
- 节点列表：如 100,200,300,500,800,1000
- 协议列表：AERIS / LEACH / PEGASIS / HEED / TEEN
- seeds 范围与 n 值
- run_tier：diagnostic / publication
- workers 数量与资源上限

### Step 2: 前置检查（必做）
1. 确认 git 状态干净或已知脏文件不影响实验
2. 若目标为 FatMachine：`ssh FatMachine` 连通性检查
3. 若目标为 FatMachine：检查是否有其他实验正在运行
4. 确认 Python 路径正确（本地 vs 服务器）
5. 确认输出文件命名不会与已有文件冲突

### Step 3: 启动前输出（必做，§13 合规）
```
【实验启动】
- 命令: [完整命令]
- 目标机器: [本地/服务器]
- 资源上限: [workers/CPU/MEM]
- 预计耗时: [区间，如 9-12 min]
- 输出文件: [完整路径]
- 验收标准: raw_results=X, error_runs=0, run_tier=publication
```

### Step 4: 验收（任务完成后）
4 项全部通过才算验收：
1. `raw_results` 计数 == 预期值
2. `error_runs` == 0
3. `run_tier` == `publication`
4. `primary_metric` == `pdr_expected`

## 关键约束
- SSH 只用 `ssh FatMachine`，禁止拼 IP
- 服务器 Python 用完整路径，禁止 conda activate
- 长实验（>10min）用 run_in_background，不阻塞主会话
- SSH 超时不等于实验失败，需检查服务器端日志和进程
- 禁止高频轮询，最小间隔 30 分钟
- ETA 必须给区间，基于实测吞吐（§27）

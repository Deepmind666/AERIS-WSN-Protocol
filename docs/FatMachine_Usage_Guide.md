# FatMachine (5090 服务器) 使用规范

最后更新: 2026-02-09
适用项目: AERIS-WSN-Protocol

---

## 1. 服务器概况

| 项目 | 规格 |
|---|---|
| 主机名 | DESKTOP-9J0A2LT |
| SSH 连接 | `ssh FatMachine` (免密, Tailscale) |
| CPU | Intel Core Ultra 9 285K, 24C/24T (无超线程, 24逻辑核=24物理核) |
| RAM | 64 GB (63.3 GB 可见) |
| GPU | NVIDIA GeForce RTX 5090 D v2, 24 GB VRAM, 驱动 581.80 |
| 磁盘 | C: 总 3.6 TB, 可用 2.0 TB |
| OS | Windows 11 专业版 (10.0.26200), OpenSSH Server, 默认 Shell: CMD |
| Python | `C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe` (3.13.11) |
| 关键包 | numpy 2.4.2, scipy 1.17.0, torch 2.6.0+cu124 (CUDA可用) |

---

## 2. 工作流

### 2.1 总体原则

- 服务器**只做计算**, 代码修改在本地完成后同步
- 所有实验必须产出带 provenance 的 JSON 结果
- 实验必须对应 Task Card 中的具体任务编号

### 2.2 代码同步 (本地 -> 服务器)

**注意**: scp 路径必须使用正斜杠, 否则会报 `unexpected EOF`。

```bash
# 同步 src (正斜杠!)
scp -r C:/AERIS-WSN-Protocol/src/* FatMachine:C:/Users/sshuser/AERIS-WSN/src/

# 同步脚本
scp C:/AERIS-WSN-Protocol/scripts/*.py FatMachine:C:/Users/sshuser/AERIS-WSN/scripts/

# 同步 data (如需要)
scp -r C:/AERIS-WSN-Protocol/data/* FatMachine:C:/Users/sshuser/AERIS-WSN/data/
```

同步前确认: 本地 smoke test 通过, git commit hash 一致。

### 2.3 提交实验

**已知问题 (2026-02-08 实测)**:
Windows OpenSSH Server 下, `nohup`, `start /B`, `Start-Process` 等后台运行方式均不可靠——SSH 断开后进程会被终止。

**可靠方案 (按优先级)**:

方案 A: **本地直接跑** (当前默认)
- 适用于大多数实验, 本地 CPU/RAM 更充裕
- 无后台运行问题

方案 B: **保持 SSH 连接前台运行**
```bash
# 必须使用完整 Python 路径 (conda activate 不可用)
ssh FatMachine "cd C:\Users\sshuser\AERIS-WSN && C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe scripts\run_experiment.py --args ..."
```
- 缺点: SSH 断开则实验中断
- 适用于短时间实验 (<30 min)

方案 C: **Windows 计划任务** (待验证)
```bash
# 创建一次性计划任务, 立即执行
ssh FatMachine "schtasks /Create /TN exp_001 /TR \"C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe C:\Users\sshuser\AERIS-WSN\scripts\run_experiment.py\" /SC ONCE /ST 00:00 /F && schtasks /Run /TN exp_001"
```
- 理论上可在 SSH 断开后继续运行, 但尚未实测验证

### 2.4 查看运行状态

每次提交实验后, **必须给出以下查看指令**:

```bash
# 查看是否在运行
ssh FatMachine "tasklist /FI \"IMAGENAME eq python.exe\" /FO TABLE"

# 实时查看日志 (最后20行) — 注意路径用反斜杠
ssh FatMachine "powershell -NoProfile -Command \"Get-Content C:\Users\sshuser\AERIS-WSN\logs\exp.log -Tail 20\""

# 查看 GPU 占用
ssh FatMachine "nvidia-smi"

# 查看可用内存 (GB)
ssh FatMachine "powershell -NoProfile -Command \"[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB, 1)\""

# 查看磁盘可用空间
ssh FatMachine "fsutil volume diskfree C:"
```

**注意**: SSH 传递 PowerShell 命令时, `$_` 会被本地 bash 的 extglob 吞掉。
如需使用 `$_`, 改用单引号或写成 PowerShell 脚本文件后远程执行。

### 2.5 结果回收 (服务器 -> 本地)

```bash
# 正斜杠! 回收指定目录
scp -r FatMachine:C:/Users/sshuser/AERIS-WSN/results/{目录}/ C:/AERIS-WSN-Protocol/results/from_fatmachine/

# 回收单个 JSON
scp FatMachine:C:/Users/sshuser/AERIS-WSN/results/mega_experiments/{文件}.json C:/AERIS-WSN-Protocol/results/mega_experiments/
```

---

## 3. 代码输出规范

**所有实验脚本必须包含 print 进度信息**, 方便用户在终端直接查看日志:

```python
# 必须包含的 print 信息:
print(f"[{datetime.now():%H:%M:%S}] 实验开始: {experiment_name}")
print(f"[{datetime.now():%H:%M:%S}] 环境: {env}, 节点: {nodes}, 重复: {reps}")
print(f"[{datetime.now():%H:%M:%S}] 进度: {i}/{total} ({i/total*100:.1f}%)")
print(f"[{datetime.now():%H:%M:%S}] 实验完成, 结果保存到: {output_path}")
```

这样用户可以直接 `Get-Content logs\xxx.log -Tail 20` 查看进展, 无需频繁询问。

---

## 4. 资源限制

| 场景 | 最大 workers | 说明 |
|---|---|---|
| 轻量 (100 nodes) | 16 | 留 8 核给系统 |
| 中等 (200-300 nodes) | 14 | 总内存不超 50GB |
| 重量 (500+ nodes) | 12 | 内存密集 |
| 超大规模 (550 reps) | **在本地跑** | 服务器 64GB 不够 |

GPU: 一次只允许一个 GPU 密集任务。

---

## 5. 多人协作

```
C:\Users\sshuser\
├── AERIS-WSN\          # Claude 4.6 专用
├── codex_gra_ops\      # Codex 专用
└── {your_name}\        # 其他协作者
```

- 不修改他人目录
- 运行前检查是否有其他实验在跑
- 结果 24h 内回收, 超 10GB 必须清理

---

## 6. 环境配置

服务器使用 Miniconda, 环境名 `aether-wsn`。

**重要**: `conda activate` 在 SSH 远程命令中不可用, 必须使用完整 Python 路径:

```bash
# 正确 (完整路径)
ssh FatMachine "C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe --version"

# 错误 (conda 不可用)
ssh FatMachine "conda activate aether-wsn && python --version"
```

**已安装关键包** (2026-02-09 实测):

| 包 | 版本 |
|---|---|
| Python | 3.13.11 |
| numpy | 2.4.2 |
| scipy | 1.17.0 |
| torch | 2.6.0+cu124 (CUDA 可用) |

---

## 7. 故障排查

```bash
# SSH 连不上 -> 检查 Tailscale
tailscale status

# 进程卡死 -> 终止
ssh FatMachine "taskkill /PID {pid} /F"

# GPU 不可用
ssh FatMachine "nvidia-smi"
```

---

## 8. 关键项目规则备忘

### 8.1 主指标口径

- **primary_metric = pdr_expected** (bs_delivered / source_packets_expected)
- 所有实验 JSON 必须包含 `"primary_metric": "pdr_expected"`
- hop count (avg_hops_to_bs) 是**二级指标**, 不可作为 primary_metric

### 8.2 Hop Count 统计规则

- `_all_hop_counts` 必须在各协议 `__init__` 中初始化为 `[]`
- AERIS: `record_hops(hops)` 按 `payload_count` 展开 (CH 聚合 N 包 → 记录 N 个 hops)
- PEGASIS: hop count 计算使用 `leader.chain_position` (完整链索引), **不是** `self.leader_index` (alive_chain 索引)
- HEED: 恒为 2.00±0.00 (协议设计决定, 无 direct-to-BS 路径)
- LEACH: 混合 1-hop (direct-to-BS) 和 2-hop (member→CH→BS)
- TEEN: 混合 1-hop 和 2-hop, 阈值触发导致 1-hop 占比高

### 8.3 实验种子规范

- latency 实验: seed 42001-42030 (n=30)
- scalability 实验: seed 50001+ (n=60)
- 同种子 + 同代码 = 确定性结果 (不构成独立验证)

### 8.4 服务器实验提交检查清单

1. 本地 smoke test 通过
2. `scp -r C:/AERIS-WSN-Protocol/src/* FatMachine:C:/Users/sshuser/AERIS-WSN/src/`
3. `scp C:/AERIS-WSN-Protocol/scripts/*.py FatMachine:C:/Users/sshuser/AERIS-WSN/scripts/`
4. 服务器 smoke test 通过
5. 检查无其他 python 进程: `ssh FatMachine "tasklist /FI \"IMAGENAME eq python.exe\" /FO TABLE"`
6. 提交实验, 记录输出文件名
7. 回收结果: `scp FatMachine:C:/Users/sshuser/AERIS-WSN/results/... C:/AERIS-WSN-Protocol/results/...`

---

## 9. 工作日志

### 2026-02-09 (Claude 4.6)

**评审同事 (Codex) latency 修复工作, 发现残留 bug 并修复:**

1. PEGASIS `leader_pos` 索引不一致 bug → 改用 `leader.chain_position`
2. 统一 5 个协议 `_all_hop_counts` 初始化到 `__init__`
3. 修正 Section 6.5 解释文字 (LEACH/TEEN/HEED 各自原因)
4. 服务器重跑 latency (n=30, 4 环境), 结果回收并更新 Section 6.5

**修改文件:**
- `src/baseline_protocols/pegasis_protocol.py` (leader_pos fix + init)
- `src/baseline_protocols/leach_protocol.py` (init)
- `src/baseline_protocols/heed_protocol.py` (init)
- `src/teen_protocol.py` (init)
- `src/aeris_protocol.py` (init)
- `for_submission/AERIS_APIN_Section6_Results.md` (表格+解释)

**新结果文件 (服务器产出):**
- `latency_indoor_office_20260209_132945.json`
- `latency_indoor_factory_20260209_133051.json`
- `latency_outdoor_urban_20260209_133155.json`
- `latency_outdoor_suburban_20260209_133257.json`
- `latency_hop_v2_stats.csv`
- `latency_hop_v2_significance.csv`
- `latency_hop_v2_stats.md`

---

## 10. 规范修改

本规范由 Claude 4.6 维护。允许提出修改建议, 但每次执行任务必须遵守当前版本规范。修改需经用户确认后生效。

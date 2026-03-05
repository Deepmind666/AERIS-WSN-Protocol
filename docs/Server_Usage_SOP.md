# FatMachine 服务器使用 SOP

## 1. SSH 连接

唯一正确方式：
```bash
ssh FatMachine
```

- 已在 `~/.ssh/config` 配置：HostName=100.104.82.45, User=sshuser, IdentityFile=id_fatmachine
- 禁止 `ssh admin@100.104.82.45` 或密码认证
- 禁止 `sshpass`

## 2. Python 环境

SSH 中 `conda activate` 不可用，必须用完整路径：
```
C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe
```

工作目录：`C:\Users\sshuser\AERIS-WSN\`

## 3. 后台运行（防 SSH 断开）

### 3.1 当前方案：Start-Process

SSH 登录后，在交互式 PowerShell 中执行：

```powershell
$py = "C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe"
$logOut = "results/mega_experiments/xxx.out.log"
$logErr = "results/mega_experiments/xxx.err.log"

Start-Process -FilePath $py `
  -ArgumentList $args `
  -RedirectStandardOutput $logOut `
  -RedirectStandardError $logErr `
  -WindowStyle Hidden
```

原理：`Start-Process` 创建独立 Windows 进程，不依赖 SSH 会话生命周期。

### 3.2 禁止的方式

| 方式 | 问题 |
|------|------|
| `nohup` | Windows 不支持 |
| `start /B` | SSH 断开后进程被终止 |
| `Start-Process` 但通过单行远程命令 `ssh FatMachine "powershell -Command ..."` | 进程树在 SSH 断开时可能被回收 |
| `conda activate` | SSH 远程命令中不可用 |

### 3.3 关键限制

- 本地 SSH 后台任务有 ~10 分钟超时限制
- 超时后 SSH 连接断开，但服务器端 `Start-Process` 启动的进程不受影响
- 禁止将 SSH 超时断开误判为实验失败

## 4. 实验监控

SSH 重连后检查状态：

```powershell
# 1) 查进程
Get-CimInstance Win32_Process | Where-Object {
  $_.Name -eq "python.exe" -and
  $_.CommandLine -like "*run_scalability*"
} | Select-Object ProcessId, CommandLine

# 2) 看日志尾部
Get-Content xxx.out.log -Tail 30

# 3) 看资源占用
Get-Counter "\Processor(_Total)\% Processor Time"
Get-Counter "\Memory\% Committed Bytes In Use"
```

## 5. 判断实验状态优先级

1. 查输出 JSON 文件是否存在
2. 存在 → 验证完整性（raw_results 计数 + error_runs + run_tier）
3. 不存在 → 查进程列表
4. 进程在跑 → 读日志尾部看进度
5. 进程不在且文件不在 → 实验失败，需排查

## 6. 队列脚本模式

长时间多任务串行执行时，将整个队列写成 `.ps1` 脚本，用 `Start-Process` 启动该脚本：

```powershell
Start-Process powershell -ArgumentList "-NoProfile -File C:\path\to\queue.ps1" `
  -RedirectStandardOutput queue.out.log `
  -RedirectStandardError queue.err.log `
  -WindowStyle Hidden
```

队列脚本内部串行执行各任务，每个任务完成后写日志标记。

## 7. 多实例协作

- 启动前先 `tasklist` 检查是否有其他实验在跑
- 禁止 `taskkill /F /IM python.exe` 批量杀进程
- 输出文件名包含机器标识（`_server_` / `_local_`）避免冲突

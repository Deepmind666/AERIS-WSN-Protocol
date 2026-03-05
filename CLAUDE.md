# AERIS-WSN-Protocol 项目记忆文件

## 项目概述
AERIS (Adaptive Environment-aware Routing for IoT Sensors) WSN路由协议研究项目。
目标期刊：MDPI Sensors (IF=3.9, Q2)。

## 关键路径与环境

### 本地环境 (Windows 11, Git Bash)
- Python: `C:/Users/admin/anaconda3/envs/aether-wsn/python.exe`
- 绘图环境: `conda run -n aether-wsn python`（避免numpy崩溃）
- LaTeX: `C:/Users/admin/AppData/Local/Programs/MiKTeX/miktex/bin/x64/pdflatex.exe`
- 工作目录: `c:/AERIS-WSN-Protocol/`

### 远程服务器 (FatMachine, 5090)
- SSH: `ssh FatMachine`（唯一正确方式，已配置免密）
- 禁止: `ssh admin@100.104.82.45` 或密码认证
- Python: `C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe`
- WSL NS-3: `ssh FatMachine "wsl -u ns3user -- bash -c 'command'"`
- NS-3路径: `/home/ns3user/ns-allinone-3.40/ns-3.40/`

## 当前状态
- 分支: `v50-rigor`（MAC碰撞模型 + baseline多跳公平性修复）
- 最新论文: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260304_v98.tex`
- 投稿包: `AERIS_Sensors_Submission/manuscript.tex`（与 v98 同步，TEXT DIFF=0）
- 门控状态: v98 通过严格审查（P0=0, P1=0, P2=若干建议项）
- S10R 实验已完成，所有数据已整合
- 待处理: Fig7 (NS-3 trend panel) 美观度升级

## 投稿包图表映射（关键同步参照）
```
fig1.pdf  ↔ fig0_aeris_workflow_20260302_s97.pdf
fig2.pdf  ↔ fig1_env_pdr_panel_20260302_s97.pdf
fig3.pdf  ↔ fig2_ablation_panel_20260304_s98.pdf
fig4.pdf  ↔ fig3_scalability_panel_20260302_s97.pdf
fig5.pdf  ↔ fig8_s8_significance_heatmap_20260302_s97.pdf
fig6.pdf  ↔ fig6_s10_delta_maps_20260302_s97.pdf
fig7.pdf  ↔ fig10_s10_absolute_profiles_20260302_s97.pdf
fig8.pdf  ↔ fig5_s11_patch_control_delta_20260302_s97.pdf
fig9.pdf  ↔ fig4_tradeoff_panel_20260302_s97.pdf
fig10.pdf ↔ fig7_ns3_trend_panel_20260302_s97.pdf
```

## 投稿包同步工作流
1. 修改图表脚本 → 生成 PDF/SVG 到 `for_submission/figures/`
2. 复制对应文件到 `AERIS_Sensors_Submission/figures/figN.pdf`
3. 同步 TeX 文本变更到 `manuscript.tex`（仅路径不同）
4. 用 MD5 验证 10 张图全部匹配
5. 分别编译两个目标，确认页数和文本一致

## 核心规则（详见 .claude/RULES.md）
1. 全程中文输出（代码/命令/路径除外）
2. PDR口径: `pdr_expected = bs_delivered / source_packets_expected`
3. 统计要求: publication级 n>=30 seeds, diagnostic级需标注
4. 修改前必须说明计划+影响范围
5. 结果JSON必须包含完整元数据
6. 禁止混用PDR口径、禁止diagnostic结果作论文结论

## 实验监控规则
- 启动实验后报告预计完成时间，然后停止。禁止反复轮询进度。
- 仅在用户明确要求"检查进度"时才检查状态。
- 最小轮询间隔：30 分钟（长跑 >4h 为 45-60 分钟）。

## 远程服务器操作规则
- 禁止使用 wmic、schtasks、DETACHED_PROCESS 启动远程进程。
- 优先用简单前台 SSH 长连接或 run_in_background。
- 同一问题尝试超过 2 种方案仍失败时，必须停下来问用户。

## 会话管理规则
- 大型审计/多文件任务优先拆分为多个聚焦会话。
- 每个会话只做一个核心目标，不混合审计+文档+实验。
- 接近上下文极限时，将进度写入 `docs/handoff_YYYYMMDD.md` 后优雅收尾。

## 绘图环境注意事项
- 直接用完整 Python 路径执行脚本: `C:/Users/admin/anaconda3/envs/aether-wsn/python.exe script.py`
- `conda run` 不支持多行 `-c` 参数，必须写入临时 .py 文件再执行
- 图表生成脚本链: s93（基础库）→ s97（Fig6等）→ s98（Fig2升级）
- 每次运行 s98 会同时重新生成 fig2 和 fig6，两个都需要同步到投稿包

## 论文审查规则
- grep 只搜目标文件，禁止用宽泛 glob 匹配到归档/旧版文件。
- 声称"所有问题已修复"前，必须做一次最终自审 pass。
- 禁止引入跳过已有版本序列的版本号。

## 可用 Skills（斜杠命令）
- `/review` — 严格审稿协议（P0/P1/P2 分级报告）
- `/onboard` — 会话恢复/快速上下文加载
- `/experiment` — 实验启动标准化检查

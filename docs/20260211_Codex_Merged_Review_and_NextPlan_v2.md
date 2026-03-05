# 2026-02-11 合并审查与下一步计划（Codex）

## 1) 本轮审查对象
- C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_scale_ext_significance.csv
- C:\AERIS-WSN-Protocol\ns3_validation\results\NS3_Section8_附录表.md
- C:\AERIS-WSN-Protocol\ns3_validation\results\NS3_vs_Python_对照表.md
- C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_4env_550_20260211_103738_descriptive.csv
- C:\AERIS-WSN-Protocol\scripts\run_scalability_experiment.py

## 2) 审查结论（严格版）

### 2.1 通过项
1. C1 数据结构修复到位  
   - ns3_scale_ext_significance.csv 列结构已规范化：environment,node_count,comparison,baseline,metric,aeris_mean,baseline_mean,diff,hedges_g,t_stat,p_value_raw,p_value_holm,sig_holm_0_05。
2. C2 附录表结构到位  
   - NS3_Section8_附录表.md 包含 Table A1/A2/A3，并明确 trend-level validation only。
3. C3 对照表门控声明到位  
   - NS3_vs_Python_对照表.md 明确禁止 numerical equivalence completed，并包含口径差异说明。

### 2.2 阻断风险（必须先处理）
1. 对照表引用了高风险 Python 数据源（阻断）  
   - NS3_vs_Python_对照表.md 第 7 行引用：scalability_4env_550_20260211_103738_descriptive.csv。
   - 该文件在 harsh 环境出现异常高值（例：第 32 行 indoor_factory,100,AERIS,550,0.995118...），与修复后本地主线口径不一致。
   - 判定：该对照表目前不能作为论文主证据，只能保留为“历史对照草稿”。

2. 本地修复后可扩展性实验仍在执行中（待收口）  
   - 执行脚本：run_scalability_fix2_local_2env.ps1
   - 当前环境：indoor_factory（进行中）
   - 发布级结论必须等待 fix2 结果，不可提前写入主稿。

## 3) 本地主线状态（Codex）
- 运行命令：
  powershell -ExecutionPolicy Bypass -File scripts/run_scalability_fix2_local_2env.ps1 -Replicates 60 -Workers 14 -Nodes "100,200,300,500,800,1000" -Rounds 300 -MaxCpuPercent 80 -MaxMemPercent 80
- 当前资源（最近采样）：CPU 73.1%，MEM 49.4%（符合上限）
- 当前累计耗时：约 00:10（从 13:38 启动计）
- 预计剩余时间（ETA）：03:50-04:40
- 输出目标：
  - C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_fix2_indoor_factory_20260211_133833.json
  - C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_fix2_outdoor_urban_20260211_133833.json
  - C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_fix2_local_2env_20260211_133833_manifest.json

## 4) 合并审查门槛（本地+服务器）
在以下条件全部满足前，禁止把“全尺度稳定领先”写入论文：
1. fix550 四环境（indoor_office/indoor_factory/outdoor_urban/outdoor_suburban）同口径齐备；
2. 每环境 raw_results 完整，error_runs=0；
3. provenance sidecar 4/4 完整（git_commit, git_dirty, git_diff_stat, script_sha256, config_hash）；
4. 统计表使用同一批 fix550 源文件重建（descriptive + significance）；
5. NS3_vs_Python 对照表改用 fix550 新源，删除旧源引用。

## 5) 下一步计划
P0（执行中）  
- 完成本地 fix2 两环境实验并验收。

P1（服务器，Claude）  
- 补齐 fix550 另外两环境（indoor_factory/outdoor_urban），并生成 sidecar。

P2（Codex）  
- 用“统一的四环境 fix550 数据”重建 descriptive/significance/cross-table；  
- 同步更新 Section 6 与 Section 8 引用，删除旧源残留。


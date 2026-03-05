# 2026-02-12 合并审查与执行计划（Codex）

## 1) 对同事 C1/C2/C3 的深度审查结论

### 1.1 通过项
- C1 统计 CSV 架构化列已生效：  
  `ns3_validation/results/ns3_scale_ext_significance.csv`
  - 已包含 `environment,node_count,comparison,baseline,metric,aeris_mean,baseline_mean,diff,hedges_g,t_stat,p_value_raw,p_value_holm,sig_holm_0_05`
- C2 附录表可用：  
  `ns3_validation/results/NS3_Section8_附录表.md`
  - 含 trend-level 声明和 A1/A2/A3 三表
- C3 对照表可用：  
  `ns3_validation/results/NS3_vs_Python_对照表.md`
  - 含口径差异说明与禁写提示

### 1.2 风险项（已识别）
- `run_scalability_experiment.py` 原实现存在口径风险（高）：使用 `benchmark_protocols` 包装器路径，`packet_delivery_ratio_end2end` 基于 attempted 分母，导致 PDR 抬高。
- 已在本地修复脚本为统一口径执行路径（见第 2 节）。

## 2) 我已完成的本地修复

### 2.1 脚本修复
文件：`scripts/run_scalability_experiment.py`

修复内容：
1. 协议执行路径统一到与主对比一致的实现：
   - LEACH/PEGASIS/HEED 使用 `src/baseline_protocols/*`
   - TEEN 使用 `src/teen_protocol.py`
   - AERIS 直接用 `src/aeris_protocol.py`
2. PDR 统一为 expected 口径：
   - `pdr_expected = bs_delivered_total / source_packets_expected`
3. 资源限制保持：
   - `--max-cpu-percent` / `--max-mem-percent` 持续生效
4. 修复健壮性问题：
   - baseline 节点 `is_alive` 为布尔字段，不再误调用
   - git 信息读取改为字节解码（`errors=ignore`），避免本机编码异常

### 2.2 结果章节修正
文件：`for_submission/AERIS_APIN_Section6_Results.md`

修复内容：
- Table 6.5 的 PEGASIS hop 数值更新为 v3 统计文件一致值：
  - indoor_office: `33.61+/-0.63`
  - indoor_factory: `32.15+/-1.94`
  - outdoor_urban: `31.35+/-2.65`
  - outdoor_suburban: `32.40+/-1.55`

### 2.3 NS-3 统计脚本质量修复
文件：`ns3_validation/results/ns3_scale_ext_significance.py`

修复内容：
- 清理乱码注释和输出文本（不改变统计算法与列结构）。

## 3) 本地冒烟与速率基准（修复后）

### 3.1 冒烟结果（口径回归）
- `results/mega_experiments/scalability_smoke_fix2_indoor_factory_20260212.json`
  - AERIS PDR: `0.9315` (n=5)
  - LEACH/PEGASIS/HEED/TEEN 分别约 `0.1641/0.1646/0.2275/0.2972`
- `results/mega_experiments/scalability_smoke_fix2_indoor_office_20260212.json`
  - AERIS PDR: `0.9961` (n=5)
  - LEACH/PEGASIS/HEED/TEEN 分别约 `0.5798/0.9153/0.9385/0.8189`

解释：结果量级与多环境主对比一致，不再出现“恶劣环境接近 1.0”的异常抬高。

### 3.2 速率基准（用于 ETA）
基准文件（indoor_factory）：
- `results/mega_experiments/scalability_benchmark_fix2_indoor_factory_n20_20260212.json`
- `results/mega_experiments/scalability_benchmark_fix2_indoor_factory_n200_r5_20260212.json`
- `results/mega_experiments/scalability_benchmark_fix2_indoor_factory_n300_r5_20260212.json`
- `results/mega_experiments/scalability_benchmark_fix2_indoor_factory_n500_r5_20260212.json`
- `results/mega_experiments/scalability_benchmark_fix2_indoor_factory_n800_r5_20260212.json`
- `results/mega_experiments/scalability_benchmark_fix2_indoor_factory_n1000_r5_20260212.json`

估算结论：
- 修复后 `n=60`、6 节点规模、5 协议、单环境约 `2.0-2.4` 小时（取决于环境）。
- 本地 2 环境顺序执行约 `4.0-4.8` 小时。

## 4) 当前执行状态

已启动本地长跑（2 环境）：
- 脚本：`scripts/run_scalability_fix2_local_2env.ps1`
- 参数：`replicates=60, workers=14, nodes=100,200,300,500,800,1000, rounds=300, max_cpu=80, max_mem=80`
- 环境：`indoor_factory`, `outdoor_urban`
- 日志：
  - `results/mega_experiments/scalability_fix2_local_2env_20260211_133830.log`
  - `results/mega_experiments/scalability_fix2_local_2env_20260211_133830.log.err`

## 5) 下一步（分工）

### Codex（本地）
1. 等本地 2 环境跑完，做结果完整性核查：
   - `raw_results == 1800`
   - `error_runs == 0`
2. 生成本地两环境 provenance（如脚本未自动生成则补跑）：
   - `scripts/generate_scalability_provenance.py --input <json>`
3. 与服务器 2 环境结果合并，产出统一 `descriptive/significance/manifest`。

### Claude（服务器）
- 任务卡：`docs/20260212_Claude_Task_Card_ScalabilityFix2.md`
- 仅跑 `indoor_office` 和 `outdoor_suburban`（`n=60`，同参数），并生成 sidecar provenance。

## 6) 论文门控（当前）
- 可写：
  - NS-3 trend-level 验证（非数值等价）
  - 100 节点多环境、以及修复后可扩展性 stress 结论（待本轮 n=60 四环境汇总后更新）
- 禁写：
  - 跨平台数值等价完成
  - 无范围限定的“全环境全尺度绝对最优”

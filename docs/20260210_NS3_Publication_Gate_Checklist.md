# NS-3 投稿门槛执行清单

**创建日期**: 2026-02-10
**创建者**: Claude 4.6 (任务 S4)
**依据**: `.claude/RULES.md` §12 + `.codex/RULES.md` §7
**当前状态**: trend-level validation PASS，numerical alignment 未达标

---

## 一、当前差距总结

| 门槛项 | 要求 | 当前状态 | 差距 |
|--------|------|----------|------|
| 参数对齐 | 全部 6 项对齐 | 4/6 对齐 | initial_energy, rounds 未对齐 |
| 样本规模 | n ≥ 30 seeds/场景 | n = 3 (隐式) | 严重不足 |
| Seed 记录 | JSON 中显式记录 | 未记录 | 缺失 |
| Git commit | 双平台记录 | 均未记录 | 缺失 |
| 统计检验 | Welch + Hedges g + Holm | 无 | 缺失 |
| 95% CI | 必须提供 | 无 | 缺失 |
| 证据文件 | JSON + 统计 + 对齐文档 | 仅有对齐文档 | 不完整 |

---

## 二、Python vs NS-3 参数映射表

### 2.1 已对齐参数

| 参数 | Python 值 | NS-3 值 | 状态 |
|------|-----------|---------|------|
| area_size | 200m × 200m | 200m × 200m | PASS |
| base_station | (100, 200) | (100, 200) | PASS |
| node_distribution | uniform random | uniform random | PASS |
| path_loss_exponent | 2.5 (indoor) | 2.5 | PASS |
| shadow_fading_std | 3.0 dB | 3.0 dB | PASS |
| rx_sensitivity | -95 dBm | -95 dBm | PASS |
| E_elec | 50 nJ/bit | 50 nJ/bit | PASS |
| E_fs | 10 pJ/bit/m² | 10 pJ/bit/m² | PASS |
| E_mp | 0.0013 pJ/bit/m⁴ | 0.0013 pJ/bit/m⁴ | PASS |
| E_DA | 5 nJ/bit | 5 nJ/bit | PASS |
| crossover_distance | 87.7 m | 87.7 m | PASS |

### 2.2 未对齐参数（必须修复）

| 参数 | Python 值 | NS-3 当前值 | 目标值 | 修复方式 |
|------|-----------|-------------|--------|----------|
| initial_energy | 2.0 J | 0.5 J | 2.0 J | 修改 NS-3 配置 |
| rounds | 300 | 200 | 300 | 修改 NS-3 配置 |
| tx_power_dbm | 10.0 (默认) | 0 dBm (固定) | 10.0 dBm | 修改 NS-3 配置 |
| packet_size | 需确认 | 需确认 | 统一 | 双平台确认 |
| seeds | 42001-42030 | 未记录 | 42001-42030 | 统一 seed 列表 |

### 2.3 模型差异（需文档说明）

| 项目 | Python | NS-3 | 影响评估 |
|------|--------|------|----------|
| 衰落模型 | Log-distance + Shadow + Multipath | Log-distance + Shadow + Rician | 功能相似，非完全一致 |
| MAC 层 | 简化 TDMA | 完整 IEEE 802.15.4 | 影响 LEACH，不影响 AERIS |
| 数据聚合 | 完美聚合 | 现实聚合 | 影响能耗对比 |

---

## 三、n ≥ 30 执行方案

### 3.1 核心场景定义

| 场景 | 节点数 | 环境 | 协议 | Seeds |
|------|--------|------|------|-------|
| S1 | 100 | indoor_office | AERIS, LEACH | 42001-42030 |
| S2 | 200 | indoor_office | AERIS, LEACH | 42001-42030 |
| S3 | 300 | indoor_office | AERIS, LEACH | 42001-42030 |

### 3.2 执行步骤

1. **参数对齐**：修改 NS-3 配置文件
   - `initial_energy = 2.0`
   - `simulation_rounds = 300`
   - `tx_power_dbm = 10.0`
   - `seeds = [42001, 42002, ..., 42030]`

2. **NS-3 重跑**：每场景 30 seeds
   - 总任务数：3 场景 × 2 协议 × 30 seeds = 180 次
   - 平台：Ubuntu 24.04 (WSL2), NS-3 3.40

3. **Python 对照**：使用相同参数重跑
   - 从已有 publication 级结果中提取对应子集
   - 或用相同 seeds 单独跑 n=30

4. **统计检验**：
   - Welch t-test（双侧）
   - Hedges' g 效应量
   - Holm-Bonferroni 多重比较校正

### 3.3 输出文件

```
ns3_validation/results/
├── ns3_indoor_office_publication_30seeds.json
├── python_indoor_office_publication_30seeds.json
├── ns3_python_significance_welch.json
├── ns3_python_significance_hedges_g.json
├── ns3_python_significance_holm.json
└── NS3_ALIGNMENT_EVIDENCE.md (更新版)
```

---

## 四、可写句式 vs 禁写句式

### 4.1 可写句式（当前状态即可使用）

- "Cross-platform trend validation on NS-3 3.40 confirms that AERIS consistently outperforms LEACH in PDR across all tested node counts (50-200)."
- "Ablation study patterns are consistent between Python simulation and NS-3, with Gateway module showing the largest contribution in both platforms."
- "Trend-level validation was performed using NS-3 3.40; full numerical alignment is planned for future work."

### 4.2 禁写句式（当前状态禁止使用）

- ~~"NS-3 numerical validation confirms..."~~
- ~~"Results are validated against NS-3 with statistical significance..."~~
- ~~"Cross-platform numerical alignment achieved..."~~
- ~~"NS-3 验证完成"~~（中文论文中同样禁止）
- ~~"200 independent runs on NS-3..."~~

### 4.3 升级条件（满足后可改用更强句式）

满足以下**全部**条件后，可写 "numerical-level validation"：

1. 参数映射表 6/6 项全部 PASS
2. 每场景 n ≥ 30 seeds，显式记录在 JSON
3. PDR 差异 ≤ 5%（按场景报告）
4. 能耗差异 ≤ 10%（仅在能耗模型完全对齐时）
5. Welch + Hedges g + Holm 统计文件齐全
6. 双平台 git_commit 记录在 JSON 中

---

## 五、执行优先级

| 优先级 | 任务 | 阻塞关系 |
|--------|------|----------|
| P0 | 论文中使用 trend-level 句式 | 无阻塞，立即可用 |
| P1 | 修复 NS-3 参数对齐 | 需要 WSL2 环境 |
| P2 | 重跑 NS-3 n=30 实验 | 依赖 P1 |
| P3 | 统计检验 + 证据文件 | 依赖 P2 |
| P4 | 升级论文句式为 numerical | 依赖 P3 全部通过 |

**建议**：当前投稿使用 P0 的 trend-level 句式即可满足 Sensors 审稿要求。P1-P4 作为 revision 阶段的改进计划。

---

## 六、审计追溯

| 项目 | 值 |
|------|-----|
| 本文件创建 commit | bf59e4a |
| 依据规则版本 | .claude/RULES.md §12 (2026-02-10) |
| 现有证据文件 | ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md |
| 现有结果文件 | ns3_validation/results/ns3_realistic_validation.json |
| NS-3 版本 | 3.40 |
| NS-3 构建平台 | Ubuntu 24.04 (WSL2), g++ 13.2.0 |

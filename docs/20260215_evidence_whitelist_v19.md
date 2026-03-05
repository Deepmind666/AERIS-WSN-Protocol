# v19 论文证据白名单

生成日期: 2026-02-15
适用版本: v19 (AERIS_Sensors_MDPI_Submission_Draft_20260215_v19.tex)

---

## 1. 白名单规则

- **白名单 (WHITE)**: v19 论文表格/图表/正文直接引用的数据文件，数值必须与文件一致
- **支撑 (SUPPORT)**: 溯源/审查/复现用，不直接出现在论文数值中，但支撑白名单文件的可信度
- **禁止 (PROHIBITED)**: 诊断/冒烟/早期版本/已废弃，禁止在 v19 论文中引用

---

## 2. 白名单文件 (WHITE) — 共8个

| # | 文件路径 | v19 引用位置 | 指标 | 样本规模 |
|---|----------|-------------|------|----------|
| W1 | `results/mega_experiments/ablation_diag_multi_20260207_205448.json` | tab:ablation_gateway (L138-141) | pdr_expected | n=30, seeds 42001-42030, 仅含AERIS消融变体(720条) |
| W8 | `results/mega_experiments/env_sensitivity_20260207_205317.json` | tab:pdr100 (L111-114) | pdr_expected | n=30, seeds 42001-42030, 五协议×四环境(600条) |
| W2 | `results/mega_experiments/scalability_4env_s8_unified_20260215_descriptive.csv` | tab:scale1000 (L165-168), Abstract (L31), S5.3 Fig3 | pdr_expected | n=1000/cell |
| W3 | `results/mega_experiments/scalability_4env_s8_unified_20260215_significance.csv` | tab:robust_snapshot (L184-187) | delta_pdr, hedges_g, p_holm | n=1000/cell |
| W4 | `ns3_validation/results/ns3_scale_ext_1000_stats.csv` | tab:ns3_trend (L224-227) | pdr_expected | NS-3 n=30 |
| W5 | `ns3_validation/results/ns3_scale_ext_1000_significance.csv` | tab:ns3_trend (L224-227), S5.6 (L232) | p_holm | NS-3 n=30 |
| W6 | `results/mega_experiments/latency_hop_fix_20260209_074608_stats.csv` | S5.5 (L204) | avg_hops | n=30 |
| W7 | `results/mega_experiments/energy_lifetime_stats.csv` | Fig4 tradeoff panel | energy, lifetime | n=30 |

---

## 3. 支撑文件 (SUPPORT) — 共17个

### 3.1 NS-3 溯源文件

| # | 文件路径 | 用途 |
|---|----------|------|
| S1 | `ns3_validation/results/ns3_scale_ext_1000_20260211.json` | W4/W5 的原始 JSON 数据 |
| S2 | `ns3_validation/results/ns3_multienv_publication_v2_20260211.json` | 多环境 NS-3 原始结果 |
| S3 | `ns3_validation/results/ns3_scale_extension_20260211.json` | 规模扩展原始结果 |
| S4 | `ns3_validation/results/ns3_multienv_stats.csv` | 多环境统计 |
| S5 | `ns3_validation/results/ns3_multienv_significance.csv` | 多环境显著性 |
| S6 | `ns3_validation/results/ns3_scale_ext_stats.csv` | 规模扩展统计 |
| S7 | `ns3_validation/results/ns3_scale_ext_significance.csv` | 规模扩展显著性 |

### 3.2 NS-3 权威文档

| # | 文件路径 | 用途 |
|---|----------|------|
| S8 | `ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md` | 参数对齐证据 (权威) |
| S9 | `ns3_validation/results/NS3_CLAIM_GATE.md` | 投稿门控判定 (权威) |
| S10 | `ns3_validation/results/NS3_GATE_CONSISTENCY_CHECK_20260215.md` | 一致性检查报告 |

### 3.3 Python 溯源文件

| # | 文件路径 | 用途 |
|---|----------|------|
| S11 | `results/mega_experiments/ablation_diag_multi_20260207_205448.provenance.json` | W1 的 provenance |
| S12 | `results/mega_experiments/env_sensitivity_20260207_205317.provenance.json` | W8 的 provenance |
| S13 | `results/mega_experiments/env_sensitivity_20260207_205317.provenance.json` | W8 的 provenance (备份) |
| S14 | `results/mega_experiments/fact_table_5protocol_pdr.csv` | 5协议 PDR 事实表 |
| S15 | `results/mega_experiments/fact_table_ablation_pdr_pvalues.csv` | 消融 p 值事实表 |
| S16 | `results/mega_experiments/latency_hop_fix_20260209_074608_significance.csv` | W6 的显著性检验 |
| S17 | `results/mega_experiments/energy_lifetime_significance.csv` | W7 的显著性检验 |
| S18 | `results/mega_experiments/scalability_4env_s8_unified_20260215_s23_summary.md` | W2/W3 的汇总报告 |

---

## 4. 禁止引用文件 (PROHIBITED)

### 4.1 冒烟测试 (smoke)

禁止原因: 样本量不足，仅用于代码验证

| 文件路径 |
|----------|
| `results/smoke_baseline_no_mac.json` |
| `results/smoke_mac_relay_test.json` |
| `results/smoke_mac_v2.json` |
| `results/smoke_test_20260204_022943.json` |
| `results/smoke_verify_final.json` |
| `results/mega_experiments/ablation_diag_multi_smoke_20260207_133450.json` |
| `results/mega_experiments/ablation_diag_multi_smoke_20260207_133543.json` |
| `results/mega_experiments/env_sensitivity_smoke_20260207_133450.json` |
| `results/mega_experiments/env_sensitivity_smoke_20260207_133512.json` |
| `results/mega_experiments/latency_smoke_test.json` |
| `results/mega_experiments/latency_smoke_test2.json` |
| `results/mega_experiments/latency_smoke_resource_guard_20260209.json` |
| `results/mega_experiments/scalability_resource_guard_smoke.json` |
| `results/mega_experiments/scalability_resource_guard_smoke_v2.json` |
| `results/mega_experiments/_smoke_scalability_fix_20260210.json` |
| `results/mega_experiments/scalability_smoke_fix_indoor_factory_20260212.json` |
| `results/mega_experiments/scalability_smoke_fix2_indoor_factory_20260212.json` |
| `results/mega_experiments/scalability_smoke_fix2_indoor_office_20260212.json` |
| `results/mega_experiments/latency_gitmeta_smoke_20260211.json` |
| `ns3_validation/results/ns3_smoke_test_20260210.json` |

### 4.2 诊断/调试 (diagnostic)

禁止原因: 非 publication-tier，参数/种子/样本量不符合发表标准

| 文件路径 | 说明 |
|----------|------|
| `results/mega_experiments/ablation_diag_20260205_144709.json` | 消融初版 |
| `results/mega_experiments/ablation_diag_multi_20260206_020002.json` | 消融多环境初版 |
| `results/mega_experiments/ablation_diag_multi_20260207_192834.json` | 消融中间版本 |
| `results/mega_experiments/cas_weight_sweep_full_20260206_000736.json` | CAS 权重扫描 |
| `results/mega_experiments/fair_5protocol_20260206_000956.json` | 5协议测试 |
| `results/mega_experiments/env_sensitivity_20260206_013048.json` | 环境敏感性初版 |
| `results/mega_experiments/env_sensitivity_20260207_125440.json` | 环境敏感性中间版本 |
| `results/mega_experiments/scalability_20260206_121956.json` | 可扩展性初版 |
| `results/mega_experiments/scalability_20260206_122752.json` | 可扩展性 v2 |
| `results/mega_experiments/latency_indoor_office_20260208_234902.json` | 延迟旧版 |
| `results/mega_experiments/latency_indoor_factory_20260208_234929.json` | 延迟旧版 |
| `results/mega_experiments/latency_outdoor_urban_20260208_234952.json` | 延迟旧版 |
| `results/mega_experiments/latency_outdoor_suburban_20260209_071707.json` | 延迟旧版 |
| `results/mega_experiments/latency_indoor_office_smoke_fix_20260209.json` | 延迟修复冒烟 |
| `results/mega_experiments/latency_indoor_office_smoke_fix2_20260209.json` | 延迟修复冒烟 v2 |
| `results/mega_experiments/latency_hop_stats.csv` | 延迟统计初版 (被 W6 替代) |
| `results/mega_experiments/latency_hop_significance.csv` | 延迟显著性初版 (被 S16 替代) |
| `results/mega_experiments/pilot_rigor_20260215_descriptive.csv` | R2 试点 (非正式) |
| `results/mega_experiments/pilot_rigor_20260215_significance.csv` | R2 试点 (非正式) |
| `results/mega_experiments/pilot_rigor_indoor_factory_20260215_local.json` | R2 试点原始 |
| `results/mega_experiments/pilot_rigor_indoor_office_20260215_local.json` | R2 试点原始 |
| `results/mega_experiments/pilot_rigor_outdoor_suburban_20260215_local.json` | R2 试点原始 |
| `results/mega_experiments/pilot_rigor_outdoor_urban_20260215_local.json` | R2 试点原始 |

### 4.3 已被替代的可扩展性中间文件

禁止原因: 已被 S8 统一矩阵 (W2/W3) 替代

| 文件路径 | 说明 |
|----------|------|
| `results/mega_experiments/scalability_4env_550_20260211_103738_descriptive.csv` | 550 版本 |
| `results/mega_experiments/scalability_4env_550_20260211_103738_significance.csv` | 550 版本 |
| `results/mega_experiments/scalability_fix2_local2env_20260211_descriptive.csv` | fix2 本地 |
| `results/mega_experiments/scalability_fix2_local2env_20260211_significance.csv` | fix2 本地 |
| `results/mega_experiments/scalability_4env_mixed_20260213_s10_*.csv` | S10 混合 |
| `results/mega_experiments/scalability_4env_mixed_20260213_s11_*.csv` | S11 混合 |
| `results/mega_experiments/scalability_4env_mixed_20260213_s12_*.csv` | S12 混合 |
| `results/mega_experiments/scalability_4env_mixed_20260213_s13_*.csv` | S13 混合 |
| `results/mega_experiments/scalability_4env_mixed_20260213_s14_*.csv` | S14 混合 |
| `results/mega_experiments/scalability_4env_mixed_20260214_s15_*.csv` | S15 混合 |
| `results/mega_experiments/scalability_4env_mixed_20260214_s16_*.csv` | S16 混合 |
| `results/mega_experiments/scalability_4env_mixed_20260214_s17_*.csv` | S17 混合 |
| `results/mega_experiments/scalability_4env_mixed_20260214_s18_*.csv` | S18 混合 |
| 所有 `scalability_*_server_*.json` 和 `scalability_*_local_*.json` | 单环境分片 |
| 所有 `scalability_benchmark_fix2_*.json` | fix2 基准测试 |

### 4.4 NS-3 非 v19 范围

禁止原因: 5协议 NS-3 对比不在 v19 scope 内

| 文件路径 | 说明 |
|----------|------|
| `ns3_validation/results/ns3_5proto_merged.json` | 5协议合并 |
| `ns3_validation/results/ns3_5proto_significance.json` | 5协议显著性 |
| `ns3_validation/results/ns3_5proto_summary.json` | 5协议汇总 |
| `ns3_validation/results/ns3_ablation_results.json` | NS-3 消融 |
| `ns3_validation/results/ns3_aligned_publication_20260211.json` | 早期对齐版本 |
| `ns3_validation/results/ns3_aligned_stats.csv` | 早期对齐统计 |
| `ns3_validation/results/ns3_aligned_significance.csv` | 早期对齐显著性 |
| `ns3_validation/results/shards_5proto/*.json` | 分片中间文件 |

### 4.5 results/ 顶层诊断文件 (部分列举)

禁止原因: 早期实验/调试/非 publication-tier

| 文件路径 |
|----------|
| `results/ablation_fix_test.json` |
| `results/aeris_final_results.json` |
| `results/aeris_v2_results.json` |
| `results/aeris_v3_results.json` |
| `results/baseline_comparison.json` |
| `results/benchmark_decision_time.json` |
| `results/complete_integration_evidence.json` |
| `results/comprehensive_dynamic_experiments.json` |
| `results/deep_analysis.json` |
| `results/dynamic_*.json` (所有动态实验) |
| `results/fair_comparison_results.json` |
| `results/five_protocol_comparison.json` |
| `results/gateway_sweep*.json` (所有网关扫描) |
| `results/improved_aeris_results.json` |
| `results/inference_bench*.json` (所有推理基准) |
| `results/innovation_validation.json` |
| `results/integration_test_*.json` (所有集成测试) |
| `results/large_scale_long*.json` (所有长期测试) |
| `results/mega_8h_*.json` (所有8小时测试) |
| `results/monte_carlo_*.json` |
| `results/ns3_cross_validation_final.json` |
| `results/ns3_python_diff_*.json` |
| `results/ns3_validation_*.json` |
| `results/overnight_8h_*.json` |
| `results/publication_experiments.json` |
| `results/python_ns3_alignment_*.json` |
| `results/quick_verify_*.json` |
| `results/runtime_weights_*.json` |
| `results/sensitivity_anova_results.json` |
| `results/ultra_scale_*.json` |
| `results/unified_output_validation_*.json` |
| `results/unified_real_experiment_*.json` |

---

## 5. 已知异常 (来自 claim_source_matrix)

| 严重级别 | claim_id | 说明 |
|----------|----------|------|
| CRITICAL | C77 | indoor_factory PDR 随规模上升 (0.6031→0.9726)，物理不合理 |
| CRITICAL | C78 | outdoor_urban PDR 随规模上升 (0.3745→0.8846)，物理不合理 |
| CRITICAL | C79 | outdoor_suburban PDR 随规模上升 (0.7451→0.9896)，物理不合理 |
| WARNING | C75 | indoor_office n=200: AERIS < LEACH by 0.0004，方向性声明需修正 |

这些异常已在 `docs/20260215_v19_claim_source_matrix_v3.csv` 中标注，MAC 碰撞模型修复计划见 plan file。

**正文门控**: C75/C77/C78/C79 禁止在论文中作为正向主张引用。

---

## 6. 已废弃的 claim 矩阵版本 (SUPERSEDED)

| 文件 | 状态 | 废弃原因 |
|------|------|----------|
| `docs/20260215_v19_claim_source_matrix.csv` (v1) | SUPERSEDED | C01-C21 错误映射到 ablation 文件（仅含 AERIS 720 条） |
| `docs/20260215_v19_claim_source_matrix_v2.csv` (v2) | SUPERSEDED | 未修正 v1 核心错误，基于错误的子代理结论 |
| `docs/20260215_claim_matrix_v1_vs_v2_diff.md` | SUPERSEDED | 差异报告基于错误结论，已被 v1_vs_v3_diff.md 替代 |

当前有效版本: `docs/20260215_v19_claim_source_matrix_v3.csv`

---

## 6. 使用规范

1. v19 论文中所有数值必须且仅可来自 WHITE 文件
2. SUPPORT 文件用于审查溯源，不直接引用数值
3. PROHIBITED 文件禁止在论文任何位置引用
4. 新增实验结果需经过 claim_source_matrix 验证后方可加入白名单

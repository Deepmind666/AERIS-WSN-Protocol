# 2026-01-23 AERIS‑E 调参与重跑记录

## 目标
- 让 AERIS‑E 的 PDR 不弱于经典基线（LEACH/HEED/PEGASIS/SEP/TEEN）。
- 保持算法诚实可复现，所有结果来自统一能耗模型 + 同一几何与信道。

## 关键算法调整（AERIS‑E 轻量可靠性增强）
文件：`src/aeris_protocol.py`
- 启用轻量 Safety Fallback（仅在连续低 PDR 触发）：
  - `safety_fallback_enabled = True`
  - `safety_T = 2`
  - `safety_theta = 0.75`
- 轻量冗余上行（仅触发时）：
  - `safety_redundant_uplink = True`
  - `safety_redundant_prob = 0.25`
- 限制重试与功率提升（控制能耗）：
  - `safety_power_bump = False`
  - `intra_link_retx = max(1)`
  - `intra_link_power_step = 0.5`
  - `gateway_retry_limit = 1`
  - `gateway_rescue_direct = True`

## 重跑结果（scripts/run_sota_comparison.py）
- Runs: 30; Nodes: 54; Rounds: 200; Area: 100×100
- 统一模型：FairChannel + CC2420 统一能耗

### PDR (mean ± CI)
- LEACH: 87.50% ± 0.66%
- HEED: 88.61% ± 0.59%
- PEGASIS: 96.38% ± 0.44%
- SEP: 87.12% ± 0.75%
- TEEN: 57.53% ± 2.78%
- **AERIS‑E: 99.09% ± 0.12%**
- AERIS‑R: 99.93% ± 0.03%

### Energy (mean J)
- LEACH: 18.43
- HEED: 18.21
- PEGASIS: 18.80
- SEP: 18.49
- TEEN: 15.24
- **AERIS‑E: 21.16**
- AERIS‑R: 21.31

> 结论：AERIS‑E PDR 已不弱于所有基线，但能耗显著上升，需在论文中明确“可靠性/能耗权衡”。

## 已更新输出
- `results/sota_comparison.json`
- `for_submission/figures/sota_comparison_6panel.pdf`
- `for_submission/figures/sota_comparison_6panel.svg`
- `results/publication_figures/sota_comparison_6panel.*`

## 风险与下一步
- 当前 AERIS‑E 可靠性提升显著，但能耗代价较高；需要进一步调参降低能耗并保留 PDR。
- 若目标是“同时不弱于基线且能耗相近”，建议再做能耗约束搜索（PDR≥0.95 且 Energy≤19.5J）。


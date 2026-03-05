# 20260210 Publication Readiness Review (Local + Server)

## 1. Data Sources Checked
- C:\AERIS-WSN-Protocol\results\mega_experiments\env_sensitivity_20260207_205317.json
- C:\AERIS-WSN-Protocol\results\mega_experiments\ablation_diag_multi_20260207_205448.json
- C:\AERIS-WSN-Protocol\results\mega_experiments\pre_ns3_scalability_summary_20260210_231438.csv
- C:\AERIS-WSN-Protocol\results\mega_experiments\pre_ns3_scalability_aeris_vs_baselines_20260210_231438.csv
- C:\AERIS-WSN-Protocol\results\mega_experiments\latency_hop_v2_stats.csv
- C:\AERIS-WSN-Protocol\results\mega_experiments\latency_hop_v2_significance.csv

## 2. Current Evidence Quality

### 2.1 100-node multi-environment (n=30)
- AERIS is rank-1 in all 4 environments (pdr_expected).
- This supports scoped claims for 100-node scenarios only.

### 2.2 Scalability 100-1000 nodes (n=550, pre-NS3)
- AERIS rank-1 in 18/24 env-node cells.
- indoor_office is the exception: PEGASIS > AERIS at all tested scales.
- This supports a scoped claim: "AERIS leads in 3/4 environments at scale".

### 2.3 Latency proxy (hop count, n=30)
- AERIS avg_hops_to_bs is stable around 1.97-1.99.
- PEGASIS avg_hops_to_bs is 31-34 (chain routing cost).
- This supports a hop-based latency claim (not wall-clock ms claim).

## 3. Manuscript Gate Status
- Forbidden-claim grep on Section 1/8 target patterns: no direct hits after this update.
- Section 1 and Section 8 now use n=550 for scalability scope.
- Remaining strict scope rule: avoid any "universal best" wording.

## 4. Publication Decision (Current)
- Paper writing can proceed now under scoped, evidence-backed claims.
- Final submission-level claim is still blocked by NS-3 gate in project rules.
- Until NS-3 publication gate passes, only trend-level NS-3 wording is allowed.

## 5. Immediate Next Steps
1. Freeze manuscript text to one evidence set (20260207/20260210 files only).
2. Complete NS-3 publication gate artifacts (n>=30, aligned parameters, stats files).
3. Run final gate sweep before submission:
   - claim gate grep
   - JSON provenance completeness check
   - figure overlap/encoding/legend check

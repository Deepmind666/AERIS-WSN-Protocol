# Sensors Reviewer Attack-Defense Table (v4)

## Scope
- Target manuscript: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213_v4.tex`
- Goal: prepare direct, auditable defense lines for strict reviewers.
- Rule: each defense maps to explicit manuscript lines and source evidence files.

| ID | Likely Reviewer Attack | Defense Sentence to Keep in Manuscript | Manuscript Anchor | Evidence Anchor |
|---|---|---|---|---|
| R1 | "You overclaim universal superiority." | "AERIS is not uniformly best in every environment-scale pair; indoor\_office is a stable counterexample." | v4:225, v4:240 | `results/mega_experiments/scalability_4env_mixed_20260213_s11_descriptive.csv` |
| R2 | "Your 100-node claims are mixed with 1000-node claims." | "The 100-node claim is restricted to the 100-node matrix and is not generalized to the 1000-node regime." | v4:107 | `results/mega_experiments/env_sensitivity_20260207_205317.json`, `results/mega_experiments/scalability_4env_mixed_20260213_s11_descriptive.csv` |
| R3 | "Sample size differs by environment; pooled conclusions are invalid." | "Large-scale claims are reported per environment and are not pooled into a single global estimate." | v4:161, v4:235 | `results/mega_experiments/scalability_4env_mixed_20260213_s11_summary.md` |
| R4 | "Significance is inflated by large n." | "Practical interpretation follows both effect size and absolute PDR gap, not significance labels alone." | v4:183 | `results/mega_experiments/scalability_4env_mixed_20260213_s11_significance.csv` |
| R5 | "Gateway contribution is cherry-picked." | "Gateway is positive in three environments and near-neutral in indoor\_office." | v4:133-134 | `results/mega_experiments/ablation_diag_multi_20260207_205448.json` |
| R6 | "CAS is marketed as always beneficial." | "CAS does not show a consistent positive marginal effect across environments." | v4:133 | `results/mega_experiments/ablation_diag_multi_20260207_205448.json` |
| R7 | "Latency claims are not physically measured." | "Hop count is used as a latency proxy, not a wall-clock latency measurement." | v4:85, v4:194 | `results/mega_experiments/latency_hop_v3_20260211_stats.csv` |
| R8 | "NS-3 block is overstated as numerical validation." | "NS-3 is used as trend-level evidence only; numerical equivalence is not claimed." | v4:203-204, v4:222, v4:234 | `ns3_validation/results/ns3_multienv_stats.csv`, `ns3_validation/results/ns3_multienv_significance.csv`, `ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md` |
| R9 | "Your reproducibility can be contaminated by hidden reliability override." | "All publication-tier scripts enforce force\_ctp\_reliable=False." | v4:84-85 | `src/aeris_protocol.py`, `scripts/run_fair_5protocol.py`, `scripts/run_ablation_diag.py`, `scripts/run_scalability_experiment.py` |
| R10 | "Std definition is ambiguous; numbers may be inconsistent." | "Tables explicitly report population standard deviation (ddof=0)." | v4:90, v4:118 | `docs/20260213_Sensors_v3_Data_Consistency_Audit.md` |
| R11 | "Energy statements may ignore lifetime bias." | "Lower total energy can co-occur with shorter lifetime; claims are framed as trade-offs under fixed conditions." | v4:227 | `results/mega_experiments/energy_lifetime_stats.csv`, `results/mega_experiments/energy_lifetime_stats.md` |
| R12 | "Baseline set is selective." | "Claims are bounded to the tested protocol set (LEACH, PEGASIS, HEED, TEEN)." | v4:30, v4:107, v4:240 | `results/mega_experiments/env_sensitivity_20260207_205317.json`, `results/mega_experiments/scalability_4env_mixed_20260213_s11_descriptive.csv` |

## Rebuttal Usage Rule
- For any reviewer question, answer with:  
  1) scoped claim sentence (from v4),  
  2) exact evidence file,  
  3) one limitation sentence (if applicable).
- Never upgrade trend-level NS-3 evidence into numerical-equivalence wording.


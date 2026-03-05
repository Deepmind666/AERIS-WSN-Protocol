# v46 Figure Data Coverage Audit

Date: 2026-02-21
Draft: AERIS_Sensors_MDPI_Submission_Draft_20260221_v46.tex
Figure script: scripts/build_sensors_figures_s44.py

## Coverage Summary

- Core publication evidence files are now integrated into figure panels (Fig0-Fig10).
- New in S44: S8 significance heatmap, S9-vs-S11 consistency, S10 absolute tx-power profiles.
- Gate status: no forbidden-claim hit in v46 (see check_sensors_draft_gate.py output).

## Data-to-Figure Mapping

| Data file | Figure usage | Notes |
|---|---|---|
| results/mega_experiments/env_sensitivity_20260207_205317.json | Fig1, Fig4 | 100-node 4-env protocol comparison; reliability profile input |
| results/mega_experiments/ablation_diag_multi_20260207_205448.json | Fig2 | Full/no_gateway/no_cas/minimal ablation effects |
| results/mega_experiments/scalability_4env_s8_unified_20260215_descriptive.csv | Fig3 | S8 4-env scalability trends (100-1000 nodes) |
| results/mega_experiments/scalability_4env_s8_unified_20260215_significance.csv | Fig8 | S8 delta and Holm-corrected significance visualization |
| results/mega_experiments/s9_matched_4env_patch_vs_control_20260216_delta.csv | Fig9 | S9-vs-S11 consistency (AERIS delta) |
| results/mega_experiments/s11_matched_4env_patch_vs_control_20260217_delta.csv | Fig5, Fig9 | Matched patch-control deltas across 4 environments |
| results/mega_experiments/s11_matched_4env_patch_vs_control_20260217_significance.csv | Fig5 | Significance-aware S11 interpretation |
| results/mega_experiments/s10_4env_merged_descriptive_20260216.csv | Fig6, Fig10 | tx5/tx15 power-sensitivity (delta + absolute profiles) |
| results/mega_experiments/s10_4env_significance_tx5_vs_tx15_20260216.csv | Fig6 | Holm-corrected significance for tx-power sensitivity |
| ns3_validation/results/ns3_scale_ext_1000_significance.csv | Fig7 | NS-3 trend-level cross-platform evidence |
| results/mega_experiments/energy_lifetime_stats.csv | Fig4 | Energy/lifetime trade-off panel |
| results/mega_experiments/latency_hop_v3_20260211_stats.csv | Fig4 | Hop-latency profile panel |

## Remaining Gap (Non-blocking)

- No standalone figure yet for S11 significance matrix (only integrated in Fig5 narrative).
- If needed for rebuttal package, add one auxiliary heatmap (S11 p_holm + effect size).

## Conclusion

For v46, data utilization is sufficient for submission-level narrative: mainline (S8), rigor patch (S9/S11), power sensitivity (S10), and NS-3 trend validation are all represented in final figures.

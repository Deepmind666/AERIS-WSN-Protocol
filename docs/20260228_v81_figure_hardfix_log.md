# v81 Figure Hard-Fix Log (2026-02-28)

## Scope
- No new experiments.
- No algorithm/code-path changes in `src/`.
- Figure readability and unit-consistency fixes only.

## Data Sources
- `results/mega_experiments/s10r_4env_merged_descriptive_20260227.csv`
- `results/mega_experiments/s10r_4env_significance_tx5_vs_tx10_vs_tx15_20260227.csv`
- `results/mega_experiments/scalability_4env_v50rigor_20260222_descriptive.csv`
- `results/mega_experiments/scalability_4env_v50rigor_20260222_significance.csv`
- `ns3_validation/results/ns3_scale_ext_1000_significance.csv`
- `ns3_validation/results/ns3_5proto_fullnodes_descriptive_20260226.csv`

## Figure Script Changes
- Created `scripts/build_sensors_figures_s81.py` from `s77`.
- Updated suffix to `20260228_s81`.
- Fig1: changed outdoor_urban inset to baseline-only zoom (L/P/H/T), removing visual duplication.
- Fig3: added per-panel baseline zoom inset to improve low-PDR protocol separability.
- Fig5: y-axis unit corrected to `Delta PDR (absolute)`.
- Fig6: removed implicit *100 conversion; values plotted in absolute PDR units; colorbar/labels aligned.

## Regenerated Assets
- `for_submission/figures/fig0_aeris_workflow_20260228_s81.{pdf,svg,png}`
- `for_submission/figures/fig1_env_pdr_panel_20260228_s81.{pdf,svg,png}`
- `for_submission/figures/fig2_ablation_panel_20260228_s81.{pdf,svg,png}`
- `for_submission/figures/fig3_scalability_panel_20260228_s81.{pdf,svg,png}`
- `for_submission/figures/fig4_tradeoff_panel_20260228_s81.{pdf,svg,png}`
- `for_submission/figures/fig5_s11_patch_control_delta_20260228_s81.{pdf,svg,png}`
- `for_submission/figures/fig6_s10_delta_maps_20260228_s81.{pdf,svg,png}`
- `for_submission/figures/fig7_ns3_trend_panel_20260228_s81.{pdf,svg,png}`
- `for_submission/figures/fig8_s8_significance_heatmap_20260228_s81.{pdf,svg,png}`
- `for_submission/figures/fig9_s9_s11_consistency_20260228_s81.{pdf,svg,png}`
- `for_submission/figures/fig10_s10_absolute_profiles_20260228_s81.{pdf,svg,png}`
- `for_submission/figures/fig11_s11_significance_panel_20260228_s81.{pdf,svg,png}`

## Manuscript Update
- Created `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260228_v81.tex` from `v80`.
- Repointed all included figures from `_s79` to `_s81`.
- Updated captions to match unit semantics and inset behavior.
- Compiled: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260228_v81.pdf` (success).

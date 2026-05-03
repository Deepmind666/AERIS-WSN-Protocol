# Fig. 2-Fig. 5 redraw manifest

Date: 2026-05-03

## Style
- Conservative network-systems style: white background, black axes, light dashed grids.
- AERIS/proposed path: blue `#1F77B4`.
- Strong collection-style or competing baseline: orange `#FF7F0E` or dark gray `#4D4D4D`.
- Classical baselines: gray scale.
- Cost/failure signal: red `#D62728`.
- Mechanism secondary signal: green `#2CA02C`.

## Outputs
- `fig_lcn26_ns3_expanded_boundary.pdf/png`
- `fig_lcn26_strict_compact.pdf/png`
- `fig_lcn26_ns3_ablation_expanded.pdf/png`
- `fig_lcn26_mechanism_compact.pdf/png`

## Sources
- Fig. 2: `scripts/build_lcn26_expanded_boundary.py`
- Fig. 3: `scripts/build_lcn26_strict_compact.py`
- Fig. 4: `scripts/build_lcn26_ns3_ablation_figure.py`
- Fig. 5: `scripts/build_lcn26_compact_tail_figures.py`

## Data
- Fig. 2: `ns3_validation/results/lcn26_ns3_dual_combined_20260430_191527_191528/summary/ns3_focused_merged.json`
- Fig. 3: `results/mega_experiments/scalability_4env_v50rigor_20260222_descriptive.csv`
- Fig. 4: `ns3_validation/results/lcn26_ns3_ablation_combined_20260501_010355_011001/summary/ns3_ablation_environment_summary.csv`
- Fig. 5: `results/lcn26_targeted_20260420/mechanism_grid_fat/mechanism_summary.csv`

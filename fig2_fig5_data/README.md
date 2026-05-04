# Fig. 2--Fig. 5 Data Pack

This directory groups the data, plotting scripts, and exported figures used for
the current Fig. 2--Fig. 5 redraw.

## Exported Figures

Stored in `exported_figures/`:

- `fig_lcn26_ns3_expanded_boundary.pdf/png`
- `fig_lcn26_strict_compact.pdf/png`
- `fig_lcn26_ns3_ablation_expanded.pdf/png`
- `fig_lcn26_mechanism_compact.pdf/png`

## Plotting Scripts

Stored in `plot_scripts/`:

- `lcn26_style.py`
- `build_lcn26_expanded_boundary.py`
- `build_lcn26_strict_compact.py`
- `build_lcn26_ns3_ablation_figure.py`
- `build_lcn26_compact_tail_figures.py`

The scripts are copied for auditability. In the working repository they still
expect the original project-relative data paths.

## Fig. 2: Expanded Boundary Sweep

`fig2_boundary/source/` contains the original summary outputs from:

`ns3_validation/results/lcn26_ns3_dual_combined_20260430_191527_191528/summary/`

Files:

- `ns3_focused_merged.json`
- `ns3_focused_descriptive.csv`
- `ns3_focused_significance.csv`
- `ns3_focused_summary.md`

`fig2_boundary/derived/ns3_boundary_gap_summary.csv` is the derived gap table
written by the current plotting script.

## Fig. 3: Collision-Aware Stress Layer

`fig3_stress/scalability_4env_v50rigor_20260222_descriptive.csv` is copied from:

`results/mega_experiments/scalability_4env_v50rigor_20260222_descriptive.csv`

No deeper raw run file with the same experiment stem was found in the repository.

## Fig. 4: Expanded Ablation

`fig4_ablation/source/` contains the summary outputs from:

`ns3_validation/results/lcn26_ns3_ablation_combined_20260501_010355_011001/summary/`

Files:

- `ns3_ablation_environment_summary.csv`
- `ns3_ablation_delta.csv`
- `ns3_ablation_summary.md`
- `ns3_focused_descriptive.csv`
- `ns3_focused_merged.json`
- `ns3_focused_significance.csv`
- `ns3_focused_summary.md`

`fig4_ablation/raw/` contains the four raw shard files from:

`ns3_validation/results/lcn26_ns3_ablation_combined_20260501_010355_011001/raw/`

Files:

- `shard_ABLATION_indoor_factory.json`
- `shard_ABLATION_indoor_office.json`
- `shard_ABLATION_outdoor_suburban.json`
- `shard_ABLATION_outdoor_urban.json`

## Fig. 5: Mechanism Matrix

`fig5_mechanism/source/` contains the mechanism summary and raw merged output
from:

`results/lcn26_targeted_20260420/mechanism_grid_fat/`

Files:

- `mechanism_summary.csv`
- `mechanism_summary.md`
- `mechanism_raw_merged.json`

## Integrity Note

The original source directories were not modified while creating this pack.
The data files here are copies for review and audit.

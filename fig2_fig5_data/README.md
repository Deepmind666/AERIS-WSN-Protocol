# AERIS LCN 2026 Figure and Table Data Pack

This directory stores the data, plotting scripts, and exported artifacts needed
to reproduce the current manuscript figures and the data-backed trade-off table.
The package is intentionally self-contained: the plotting scripts first read the
copies in this directory and only fall back to the original experiment folders if
the packed copies are absent.

## Reproduction Commands

For the full figure rebuild and mirroring step, run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File fig2_fig5_data\rebuild_figures_from_pack.ps1
```

The script regenerates Fig. 2--Fig. 5, mirrors the PDFs into
`overleaf_upload_ready_20260503/figures/`, and updates `exported_figures/`.

The individual plotting commands are:

```powershell
C:\Users\admin\anaconda3\python.exe fig2_fig5_data\plot_scripts\build_lcn26_ns3_canonical_margin.py
C:\Users\admin\anaconda3\python.exe fig2_fig5_data\plot_scripts\build_lcn26_strict_compact.py
C:\Users\admin\anaconda3\python.exe fig2_fig5_data\plot_scripts\build_lcn26_ns3_ablation_figure.py
C:\Users\admin\anaconda3\python.exe fig2_fig5_data\plot_scripts\build_lcn26_compact_tail_figures.py
```

The individual commands write regenerated PDFs/PNGs to `_LCN26_AERIS/generated/`.

## Exported Manuscript Figures

`exported_figures/` contains current figure exports using the paper filenames:

- `fig1_workflow.png`
- `fig2_classical_margin.pdf/png`
- `fig3_stress.pdf/png`
- `fig4_ablation.pdf/png`
- `fig5_mechanism.pdf/png`

Older exported files are left in the directory only as audit history; the files
listed above are the current manuscript figures.

## Fig. 1: AERIS Workflow

`fig1_workflow/source/` contains:

- `fig1_workflow.png`: the current manuscript raster.
- `AERIS_fig1_flow_chart.drawio`: editable source drawing supplied for the final workflow.
- `AERIS_fig1_flow_chart.svg`: SVG export of the editable source.

Fig. 1 is a protocol schematic, so it is not generated from numeric experiment
data.

## Fig. 2: Canonical Classical Margin

`fig2_classical_margin/source/` contains:

- `ns3_5proto_fullnodes_descriptive_20260226.csv`

This is the five-protocol canonical NS-3 summary used by the current Fig. 2:
AERIS, LEACH, PEGASIS, HEED, and TEEN across four environments and seven node
scales. The plotting script computes AERIS minus the strongest classical
baseline for each cell.

`fig2_classical_margin/derived/` contains:

- `ns3_classical_margin_summary.csv`

This derived table records the plotted per-cell margin, approximate 95% CI,
best classical baseline, AERIS rank, top-two flag, and near-tie flag.

## Fig. 3: Collision-Aware Strict-Physics Stress Layer

`fig3_stress/` contains:

- `scalability_4env_v50rigor_20260222_descriptive.csv`

This file is copied from the strict-physics Python stress layer. It is stress
evidence only because the baselines are adapted with relay support.

## Fig. 4: NS-3 AERIS Ablation

`fig4_ablation/source/` contains the ablation summaries:

- `ns3_ablation_delta.csv`
- `ns3_ablation_environment_summary.csv`
- `ns3_ablation_summary.md`
- `ns3_focused_descriptive.csv`
- `ns3_focused_merged.json`
- `ns3_focused_significance.csv`
- `ns3_focused_summary.md`

`fig4_ablation/raw/` contains the four raw shard files:

- `shard_ABLATION_indoor_factory.json`
- `shard_ABLATION_indoor_office.json`
- `shard_ABLATION_outdoor_suburban.json`
- `shard_ABLATION_outdoor_urban.json`

The current figure plots full-minus-ablated PDR in percentage points, so positive
values mean the removed module helped the full AERIS configuration.

## Fig. 5: Mechanism Matrix

`fig5_mechanism/source/` contains:

- `mechanism_summary.csv`
- `mechanism_summary.md`
- `mechanism_raw_merged.json`

The current Fig. 5 is generated from the 12-cell mechanism study: four
environments at 100, 500, and 1000 nodes, with 400 replicates per cell.

## Table III: Fixed 100-Node Trade-Off Summary

`table3_pooled_summary/source/` contains the data used to compute the pooled
100-node PDR, consumed energy, lifetime, first-node death, and hop-count table:

- `energy_lifetime_stats.csv`
- `latency_hop_v3_20260211_stats.csv`

The table is pooled across the four environments and is separate from the
50--1000 node sweeps in Fig. 2 and Fig. 4.

## Legacy Boundary Data

`fig2_boundary/` is retained as audit history for the earlier expanded boundary
plot. It is not the current Fig. 2 in the manuscript. The expanded seven-protocol
boundary result is now summarized in prose, while the plotted Fig. 2 uses the
canonical five-protocol classical audit.

## Integrity Note

The files in this data pack are copies for review and reproducibility. The
original experiment directories were not modified while creating the pack.

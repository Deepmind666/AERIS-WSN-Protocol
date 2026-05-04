# AERIS LCN 2026 Figure/Table Data Pack

This directory is organized for review and reproducibility. The numbered
folders match the manuscript artifacts, and each formal figure has one primary
data folder.

## Quick Map

| Folder | Purpose |
|---|---|
| `00_final_outputs/` | Final exported PDFs/PNGs for manuscript Fig. 2--Fig. 5. |
| `01_fig2_classical_margin/` | Data for Fig. 2, the canonical five-protocol NS-3 classical-margin plot. |
| `02_fig3_stress/` | Data for Fig. 3, the collision-aware strict-physics stress layer. |
| `03_fig4_ablation/` | Data for Fig. 4, the NS-3 AERIS ablation heatmap. |
| `04_fig5_mechanism/` | Data for Fig. 5, the 12-cell mechanism study. |
| `05_table3_pooled_summary/` | Data for Table III, the fixed 100-node pooled trade-off summary. |
| `90_expanded_boundary_text_evidence/` | Extra seven-protocol boundary-sweep data used for prose claims, not for a current manuscript figure. |
| `scripts/` | Plotting scripts and shared style file. |

## Rebuild

Run this from the repository root:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File fig2_fig5_data\rebuild_figures_from_pack.ps1
```

The script regenerates Fig. 2--Fig. 5, mirrors the PDFs into
`LCN26_AERIS_overleaf/figures/`, and updates `00_final_outputs/`.

Individual commands:

```powershell
C:\Users\admin\anaconda3\python.exe fig2_fig5_data\scripts\build_lcn26_ns3_canonical_margin.py
C:\Users\admin\anaconda3\python.exe fig2_fig5_data\scripts\build_lcn26_strict_compact.py
C:\Users\admin\anaconda3\python.exe fig2_fig5_data\scripts\build_lcn26_ns3_ablation_figure.py
C:\Users\admin\anaconda3\python.exe fig2_fig5_data\scripts\build_lcn26_compact_tail_figures.py
```

## Final Manuscript Outputs

`00_final_outputs/` contains only the formal figure exports used by the current
paper:

- `fig2_classical_margin.pdf/png`
- `fig3_stress.pdf/png`
- `fig4_ablation.pdf/png`
- `fig5_mechanism.pdf/png`

Fig. 1 is a manually edited workflow schematic and is packaged directly in
`LCN26_AERIS_overleaf/figures/fig1_workflow.png`; it is not a numeric experiment
figure.

## Fig. 2: Canonical Classical Margin

Folder: `01_fig2_classical_margin/`

- `source/ns3_5proto_fullnodes_descriptive_20260226.csv`: canonical five-protocol
  NS-3 summary for AERIS, LEACH, PEGASIS, HEED, and TEEN across four environments
  and seven node scales.
- `derived/ns3_classical_margin_summary.csv`: plotted AERIS-minus-best-classical
  margins, confidence intervals, best classical baseline, AERIS rank, top-two
  flag, and near-tie flag.

This is the data behind the current Fig. 2.

## Fig. 3: Collision-Aware Strict-Physics Stress Layer

Folder: `02_fig3_stress/`

- `source/scalability_4env_v50rigor_20260222_descriptive.csv`

This is stress evidence only because LEACH, HEED, and TEEN are minimally
relay-enabled in this layer for multi-hop comparability.

## Fig. 4: NS-3 AERIS Ablation

Folder: `03_fig4_ablation/`

- `source/ns3_ablation_delta.csv`: plotted full-minus-ablated PDR deltas.
- `source/ns3_ablation_environment_summary.csv`
- `source/ns3_ablation_summary.md`
- `source/ns3_focused_descriptive.csv`
- `source/ns3_focused_merged.json`
- `source/ns3_focused_significance.csv`
- `source/ns3_focused_summary.md`
- `raw/shard_ABLATION_*.json`: raw environment shards.

Positive plotted values mean that removing the module hurt delivery.

## Fig. 5: Mechanism Matrix

Folder: `04_fig5_mechanism/`

- `source/mechanism_summary.csv`: plotted mechanism summary.
- `source/mechanism_summary.md`
- `source/mechanism_raw_merged.json`: merged raw mechanism records.

The figure covers 12 environment-scale cells: four environments at 100, 500, and
1000 nodes, with 400 replicates per cell.

## Table III: Fixed 100-Node Trade-Off Summary

Folder: `05_table3_pooled_summary/`

- `source/energy_lifetime_stats.csv`
- `source/latency_hop_v3_20260211_stats.csv`

The table is pooled across the four environments and is separate from the
50--1000 node sweeps in Fig. 2 and Fig. 4.

## Expanded Boundary Text Evidence

Folder: `90_expanded_boundary_text_evidence/`

This folder stores the expanded seven-protocol boundary-sweep data used for the
prose statement that AERIS remains strongest mainly in outdoor suburban cells
once simplified CTP and RPL-MRHOF baselines are included. It is intentionally not
named as Fig. 2 because the current manuscript Fig. 2 is the classical-margin
plot.

## Integrity

`MANIFEST.sha256` records SHA-256 hashes for the files in this pack. Regenerate
it after changing the data pack:

```powershell
Get-ChildItem -Path fig2_fig5_data -Recurse -File |
  Where-Object { $_.Name -ne 'MANIFEST.sha256' } |
  Sort-Object FullName |
  Get-FileHash -Algorithm SHA256 |
  ForEach-Object { "$($_.Hash)  $(Resolve-Path -Relative $_.Path)" } |
  Set-Content -Encoding ASCII fig2_fig5_data\MANIFEST.sha256
```

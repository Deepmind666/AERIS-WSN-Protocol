# Figure and Table Data Manifest

Use this manifest when checking whether the paper figures match the committed
data.

## Fig. 2: Classical NS-3 Margin

Output:

```text
paper/LCN26_AERIS_overleaf/figures/fig2_classical_margin.pdf
data/figure_reproduction/00_final_outputs/fig2_classical_margin.pdf
```

Source:

```text
data/figure_reproduction/01_fig2_classical_margin/source/ns3_5proto_fullnodes_descriptive_20260226.csv
data/figure_reproduction/01_fig2_classical_margin/derived/ns3_classical_margin_summary.csv
```

Script:

```text
data/figure_reproduction/scripts/build_lcn26_ns3_canonical_margin.py
```

## Fig. 3: Strict-Physics Stress Evidence

Output:

```text
paper/LCN26_AERIS_overleaf/figures/fig3_stress.pdf
data/figure_reproduction/00_final_outputs/fig3_stress.pdf
```

Source:

```text
data/figure_reproduction/02_fig3_stress/source/scalability_4env_v50rigor_20260222_descriptive.csv
```

Script:

```text
data/figure_reproduction/scripts/build_lcn26_strict_compact.py
```

## Fig. 4: NS-3 Ablation

Output:

```text
paper/LCN26_AERIS_overleaf/figures/fig4_ablation.pdf
data/figure_reproduction/00_final_outputs/fig4_ablation.pdf
```

Source:

```text
data/figure_reproduction/03_fig4_ablation/source/ns3_ablation_delta.csv
data/figure_reproduction/03_fig4_ablation/source/ns3_focused_descriptive.csv
data/figure_reproduction/03_fig4_ablation/raw/
```

Script:

```text
data/figure_reproduction/scripts/build_lcn26_ns3_ablation_figure.py
```

## Fig. 5: Mechanism/Trade-off Matrix

Output:

```text
paper/LCN26_AERIS_overleaf/figures/fig5_mechanism.pdf
data/figure_reproduction/00_final_outputs/fig5_mechanism.pdf
```

Source:

```text
data/figure_reproduction/04_fig5_mechanism/source/mechanism_summary.csv
data/figure_reproduction/04_fig5_mechanism/source/mechanism_raw_merged.json
```

Script:

```text
data/figure_reproduction/scripts/build_lcn26_compact_tail_figures.py
```

## Table III: Fixed 100-node Pooled Summary

Source:

```text
data/figure_reproduction/05_table3_pooled_summary/source/energy_lifetime_stats.csv
data/figure_reproduction/05_table3_pooled_summary/source/latency_hop_v3_20260211_stats.csv
```

## Expanded Seven-Protocol Boundary Text Evidence

Source:

```text
data/figure_reproduction/90_expanded_boundary_text_evidence/source/
data/figure_reproduction/90_expanded_boundary_text_evidence/derived/ns3_boundary_gap_summary.csv
```

This evidence is used for deployment-boundary text and should not be confused
with the classical Fig. 2 margin.

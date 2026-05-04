AERIS LCN 2026 Overleaf package

Included files:
- aeris_lcn2026.tex
- ref.bib
- aeris_lcn2026.pdf
- IEEEtran.cls
- figures/fig1_workflow.png
- figures/fig2_classical_margin.pdf
- figures/fig3_stress.pdf
- figures/fig4_ablation.pdf
- figures/fig5_mechanism.pdf

Source data and figure-rebuild scripts:
- fig2_fig5_data/README.md
- fig2_fig5_data/01_fig2_classical_margin/source/ns3_5proto_fullnodes_descriptive_20260226.csv
- fig2_fig5_data/02_fig3_stress/source/scalability_4env_v50rigor_20260222_descriptive.csv
- fig2_fig5_data/03_fig4_ablation/source/ns3_focused_descriptive.csv
- fig2_fig5_data/03_fig4_ablation/source/ns3_focused_significance.csv
- fig2_fig5_data/03_fig4_ablation/source/ns3_focused_merged.json
- fig2_fig5_data/04_fig5_mechanism/source/mechanism_summary.csv
- fig2_fig5_data/04_fig5_mechanism/source/mechanism_raw_merged.json
- fig2_fig5_data/05_table3_pooled_summary/source/energy_lifetime_stats.csv
- fig2_fig5_data/05_table3_pooled_summary/source/latency_hop_v3_20260211_stats.csv

Compile recipe used locally:
pdflatex -> bibtex -> pdflatex -> pdflatex

Notes:
- Final PDF length: 8 pages.
- The manuscript figures mirror the corresponding outputs in fig2_fig5_data/00_final_outputs/.
- Fig. 1 uses the hand-drawn workflow PNG provided by the author.

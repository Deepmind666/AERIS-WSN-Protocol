# Sensors v17 Editorial Changes

Date: 2026-02-14  
Scope: Refine manuscript with reviewer-safe scalability wording, author-order correction, and upgraded s22 figure style while waiting for final S8 unified outputs.

## Updated manuscript
- for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260214_v17.tex

## Main text changes
- Updated author order: Kangrui Li is now first author and corresponding author.
- Replaced figure set with s22 in all four figure panels.
- Reverted large-scale narrative to conservative audited-matrix wording: AERIS is first in 3/4 environments, with indoor office as a counterexample.
- Updated 1000-node table values and robustness snapshot to the audited pre-unified matrix.
- Clarified seed/sample statement to: n=550 in indoor office/outdoor urban/outdoor suburban and n=650 in indoor factory.
- Corrected limitations sample-size statement to `(n=550 or n=650)`.

## Figure/data sources in this revision
- for_submission/figures/fig1_env_pdr_panel_20260214_s22.pdf
- for_submission/figures/fig2_ablation_panel_20260214_s22.pdf
- for_submission/figures/fig3_scalability_panel_20260214_s22.pdf
- for_submission/figures/fig4_tradeoff_panel_20260214_s22.pdf
- results/mega_experiments/scalability_4env_mixed_20260213_s11_descriptive.csv
- results/mega_experiments/scalability_4env_mixed_20260213_s11_significance.csv

## Compilation and checks
- PDF compiled successfully:
  - for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260214_v17.pdf
- Gate check:
  - PASS via scripts/check_sensors_draft_gate.py
- DOI-hard used-reference audit:
  - docs/20260214_v17_used_refs_doi_hard.csv
  - docs/20260214_v17_used_refs_doi_hard.md
  - verified_doi=17
- Fast used-reference sanity audit:
  - docs/20260214_v17_used_refs_fastcheck.csv
  - docs/20260214_v17_used_refs_fastcheck.md
  - probable_real=17

## Notes
- This is an interim writing build using currently available audited data before final indoor_factory unified rebuild.
- No core source (`src/`) changes were made in this step.

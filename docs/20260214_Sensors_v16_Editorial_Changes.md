# Sensors v16 Editorial Changes

Date: 2026-02-14  
Scope: Reviewer-grade wording tightening, reference integrity hardening, and figure style refinement.

## Updated manuscript
- for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260214_v16.tex

## Main text changes
- Tightened scope language in Abstract/Introduction/Conclusion to keep claims bounded to protocol set, environment taxonomy, metric definition, and sample-size regime.
- Clarified scalability setup wording (`n=550`, indoor factory `n=650`) as a frozen matrix note.
- Updated trade-off figure caption to match actual plotting method (log-scale hop panel, no inset zoom wording).
- Kept all conclusions trend-safe and gate-compliant.

## Reference integrity repair
- Updated bibliography metadata in:
  - for_submission/bibliography.bib
- Corrected DOI/metadata keys:
  - `Rault2016Energy`
  - `Kandris2020`
  - `Ren2024`
  - `Okine2024`
  - `Chen2023Survey`
- Removed unstable context-aware citations from the active draft text:
  - `Liu2018Environment`
  - `Zhao2019Context`

## Figure refinement (s18)
- Added script:
  - scripts/build_sensors_figures_s18.py
- New figure set:
  - for_submission/figures/fig1_env_pdr_panel_20260214_s18.pdf
  - for_submission/figures/fig2_ablation_panel_20260214_s18.pdf
  - for_submission/figures/fig3_scalability_panel_20260214_s18.pdf
  - for_submission/figures/fig4_tradeoff_panel_20260214_s18.pdf

## Compilation and checks
- PDF compiled successfully:
  - for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260214_v16.pdf
- Gate check:
  - PASS via scripts/check_sensors_draft_gate.py
- Data consistency audit:
  - docs/20260214_Sensors_v16.csv
  - docs/20260214_Sensors_v16.md
  - result: fail=0, warn=27
- Used-reference audits:
  - docs/20260214_v16_used_refs_fastcheck.csv / .md
  - docs/20260214_v16_used_refs_doi_hard.csv / .md
  - DOI-hard status: verified_doi=17 (no mismatch/unresolved in used set)


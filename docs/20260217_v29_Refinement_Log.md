# Sensors v29 Refinement Log (2026-02-17)

## Scope
- Continue manuscript polishing while server-side experiment pipeline runs separately.
- Keep all claims bounded to existing audited evidence blocks.
- Upgrade figure visual quality without changing data sources.

## Completed Changes

### 1) Figure pipeline upgrade (S25 style pass)
- Added new plotting script: `scripts/build_sensors_figures_s25.py`.
- Generated new figure set:
  - `for_submission/figures/fig1_env_pdr_panel_20260217_s25.pdf`
  - `for_submission/figures/fig2_ablation_panel_20260217_s25.pdf`
  - `for_submission/figures/fig3_scalability_panel_20260217_s25.pdf`
  - `for_submission/figures/fig4_tradeoff_panel_20260217_s25.pdf`
- Style updates:
  - softer low-saturation palette,
  - stronger line readability for scalability curves,
  - tuned CI alpha bands,
  - rounded bars for panel consistency,
  - tighter legend framing and spacing.

### 2) Manuscript update to v29
- New draft file: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260217_v29.tex`.
- Updated all four figure references from S24 to S25 outputs.
- Compiled PDF: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260217_v29.pdf`.

### 3) Format check and compile status
- Ran BibTeX and multi-pass LaTeX compile.
- Citation resolution completed (no undefined citation warnings in final pass).
- Resolved prior table overflow by converting `tab:deployment_summary` to width-bounded `p{}` columns.
- Remaining warnings are minor underfull line breaks in narrow table cells.

## Data/Claim Policy
- No new data claims introduced.
- No cross-regime pooling introduced.
- NS-3 remains trend-level only.

## Next Recommended Step
- Once the pending experiment block is finalized, only update affected table rows/claim cells and regenerate final camera-ready figures from the same S25 style template.

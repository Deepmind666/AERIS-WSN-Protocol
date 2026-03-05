# v56 Figure & Submission Review Checklist (2026-02-22)

## Scope
- Draft: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260222_v56.tex`
- PDF: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260222_v56.pdf`
- Plot script: `scripts/build_sensors_figures_s56.py`
- Figures: `for_submission/figures/fig0..fig7_*_20260222_s56.(pdf|svg|png)`

## Hard-Issue Closure (from latest figure review)
| ID | Issue | Action | Status |
|---|---|---|---|
| F1 | Axis labels contained code variable (`pdr_expected`) | Replaced figure labels/captions with `PDR` / `mean PDR` | CLOSED |
| F2 | Fig.3 low-value protocols compressed | Added per-panel non-AERIS zoom insets in `plot_fig3()` | CLOSED |
| F3 | Fig.6 not grayscale-safe | Switched to grayscale-safe `|delta|` heatmaps + sign text + marker encoding | CLOSED |
| F4 | Fig.6 small readability | Increased font sizes and simplified visual channels | CLOSED |
| F5 | Fig.3 CI bands not visible | Caption now states CI can be narrower than line width at n=3200 | CLOSED |
| F6 | Fig.0 too sparse | Rebuilt compact workflow layout and reduced page width in tex | CLOSED |
| F7 | Hollow/filled significance markers weak | Replaced with explicit `x` markers for non-significant cells | CLOSED |
| F8 | `pp` undefined | Figure caption updated to `percentage points (pp)` | CLOSED |
| F9 | Fig.3 legend wasteful | Kept single shared legend for all subplots | CLOSED |
| F10 | Inconsistent typography | Unified style via updated `apply_style()` defaults | CLOSED |

## Gate & Consistency Checks
- `python scripts/check_sensors_draft_gate.py --draft for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260222_v56.tex` -> PASS
- Figure/table cross-reference coverage: 20/20 labels cited in main text -> PASS
- LaTeX build: `pdflatex + bibtex + pdflatex + pdflatex` -> PASS (no undefined references/citations in final run)

## Generated/Updated Files
- `scripts/build_sensors_figures_s56.py`
- `for_submission/figures/fig0_aeris_workflow_20260222_s56.pdf`
- `for_submission/figures/fig1_env_pdr_panel_20260222_s56.pdf`
- `for_submission/figures/fig2_ablation_panel_20260222_s56.pdf`
- `for_submission/figures/fig3_scalability_panel_20260222_s56.pdf`
- `for_submission/figures/fig4_tradeoff_panel_20260222_s56.pdf`
- `for_submission/figures/fig5_s11_patch_control_delta_20260222_s56.pdf`
- `for_submission/figures/fig6_s10_power_sensitivity_20260222_s56.pdf`
- `for_submission/figures/fig7_ns3_trend_panel_20260222_s56.pdf`
- `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260222_v56.tex`
- `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260222_v56.pdf`

## Residual Notes (non-blocking)
- Some underfull hbox warnings remain (line breaking only, not scientific or gate failures).
- If needed, next pass can reduce visual density further by moving Fig.0 to supplementary.

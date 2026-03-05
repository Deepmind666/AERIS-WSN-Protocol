# Sensors Draft Strict Reviewer Checklist (2026-02-13)

## Scope
- Target manuscript: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213.tex`
- Target figures: `for_submission/figures/fig1_env_pdr_panel_20260213_s11.pdf` to `for_submission/figures/fig4_tradeoff_panel_20260213_s11.pdf`

## Hard-Gate Status
- `PASS`: no internal file paths in manuscript body.
- `PASS`: forbidden claims not found (`100% PDR`, `200 independent runs`, `2500ms`, `<10ms`, `TDA metric`).
- `PASS`: MDPI template compile succeeds (`for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213.pdf`).
- `PASS`: figures regenerated with color-safe palette and no text overlap by design.

## Remaining Risks (Reviewer Perspective)
- `HIGH`: mixed sample-size scalability wording (`n=650` vs `n=550`) can be misread as a unified matrix without stratification.
  - Action: keep explicit per-environment sample-size statement in Results and Conclusion.
- `MEDIUM`: effect-size values in robustness snapshot are very large in harsh environments and may trigger methodology questions.
  - Action: in camera-ready text, add one sentence clarifying that large `g` is driven by large mean gaps and low within-group variance in large-`n` settings.
- `MEDIUM`: NS-3 remains trend-level validation.
  - Action: keep strict wording "trend-level only" and avoid any numerical-equivalence phrase.
- `LOW`: one bibliography line still has a non-blocking underfull hbox warning.
  - Action: optional; no scientific impact.

## Next Revision Actions (Directly Executable)
1. Add a one-paragraph "Statistical Methods" subsection describing Welch t-test, Holm correction, and Hedges' g interpretation bands.
2. Add one sentence in Discussion to explicitly separate:
   - "100-node all-environment ranking"
   - "large-scale mixed-sample ranking"
3. Keep the local zoom inset in Fig.3 (indoor office) and add one sentence in caption stating why the inset is used.

## Acceptance Criteria for Submission-Ready Draft
- No forbidden claim hits in manuscript.
- Compile clean with no errors and no hyperref token warnings.
- Figure captions are self-contained and scoped by sample size.
- NS-3 claims remain trend-level, not numerical-equivalence.

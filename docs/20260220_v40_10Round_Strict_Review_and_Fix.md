# 20260220 v40 Ten-Round Strict Review and Fix Log

## Scope
- Target draft: for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260220_v40.tex
- Figure script: scripts/build_sensors_figures_s40.py
- Output figures: for_submission/figures/fig1_env_pdr_panel_20260220_s40.pdf ... fig7_ns3_trend_panel_20260220_s40.pdf

## Round 1 — Layout and Template Compliance
- Check: MDPI Sensors class, section order, figure/table placement style.
- Action: kept MDPI class and tightened section transitions for readability in Introduction, Setup, Results.
- Result: PASS (no template-level break found).

## Round 2 — Abstract Clarity and Scientific Boundaries
- Check: density, claim scope, limitation visibility.
- Action: rewrote abstract into structured evidence flow (100-node matrix -> S8 baseline matrix -> S9/S10/S11 calibration -> NS-3 trend-only boundary).
- Result: PASS (claims remain scoped; no forbidden absolute language).

## Round 3 — Contribution Novelty Framing
- Check: innovation statement was too generic.
- Action: revised contribution bullets to explicitly separate (i) modular protocol design novelty, (ii) evidence-matrix novelty, (iii) cross-platform validation novelty.
- Result: PASS (innovation statement is clearer and less defensive).

## Round 4 — Mathematical Rigor (Method Section)
- Check: insufficient explicit equations for decision and statistics.
- Action:
  - Added CAS mode-selection scoring equation.
  - Added explicit Welch t-test equation.
  - Added explicit Hedges g equation with small-sample correction term.
- Result: PASS (method section now has auditable core formulas).

## Round 5 — Statistical Interpretation Tightening
- Check: risk of over-reading p-values in large n cells.
- Action: preserved joint interpretation rule (Holm p + effect size + absolute delta), kept Hedges g caveat.
- Result: PASS.

## Round 6 — Figure Aesthetic Upgrade
- Check: previous style considered coarse.
- Action in scripts/build_sensors_figures_s40.py:
  - muted publication palette re-tuned;
  - stronger visual hierarchy for AERIS line;
  - thicker lines and clearer CI bands;
  - consistent white-background export settings.
- Result: PASS (s40 figures regenerated).

## Round 7 — Evidence Utilization Depth
- Check: NS-3 evidence underused in visuals.
- Action:
  - Added Figure 7 generator (NS-3 trend panel over 50-1000 nodes, with significance marking).
  - Inserted Figure 7 into NS-3 results subsection in v40.
- Result: PASS (NS-3 evidence now has direct visual support).

## Round 8 — Caption Quality and Reader Guidance
- Check: captions needed stronger interpretability cues.
- Action: improved figure captions for S10/S11/NS-3 to explicitly define delta sign and significance markers.
- Result: PASS.

## Round 9 — Gate Compliance
- Check command:
  - python scripts/check_sensors_draft_gate.py --draft for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260220_v40.tex
- Result:
  - forbidden claims: PASS
  - path leakage: PASS
  - overall gate: PASS

## Round 10 — Build and Reproducibility Verification
- Build commands executed in for_submission:
  - pdflatex -> bibtex -> pdflatex -> pdflatex
- Output:
  - for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260220_v40.pdf
- Result: PASS (compiled PDF generated successfully).

## Final Status
- v40 is a direct improvement over v39 in method rigor, visual quality, and NS-3 evidence usage.
- Remaining work is editorial fine-polish, not structural correctness.

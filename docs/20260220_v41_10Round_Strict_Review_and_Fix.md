# 20260220 v41 Second-Cycle Ten-Round Strict Review and Fix Log

## Target
- Draft: for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260220_v41.tex
- Figure set: scripts/build_sensors_figures_s41.py

## Round 1 (Structure)
- Rechecked section sequencing for submission readability.
- Added stronger contribution framing to prevent novelty dilution.

## Round 2 (Claim Scope)
- Tightened wording so claims remain matrix-scoped.
- Kept NS-3 boundary as trend-level only.

## Round 3 (Method Clarity)
- Added explicit auditability subsection clarifying publication-tier claim constraints.

## Round 4 (Math Rigor)
- Preserved explicit Welch and Hedges-g formulas.
- Confirmed notation consistency with text interpretation.

## Round 5 (Discussion Logic)
- Strengthened practical interpretation sentence to avoid over-claiming under S8 limitations.

## Round 6 (Layout and Table Readability)
- Refactored deployment summary table columns to ragged-right p-columns.
- Reduced line-break artifacts and improved readability under MDPI layout.

## Round 7 (Figure Style)
- Produced S41 style pass: softer palette, lower visual harshness, clearer edge hierarchy.
- Kept white background and grayscale-safe line styles.

## Round 8 (Evidence Utilization)
- Retained S10 full matrix and S11 matched matrix visual evidence.
- Retained NS-3 full trend panel (50--1000 nodes, 4 environments).

## Round 9 (Gate)
- Gate command: python scripts/check_sensors_draft_gate.py --draft for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260220_v41.tex
- Result: PASS.

## Round 10 (Build)
- Compile chain executed: pdflatex -> bibtex -> pdflatex -> pdflatex.
- Output PDF produced successfully.

## Final
- v41 is publication-facing and stricter than v40 in wording discipline and visual polish.
- Remaining work is optional copy-edit polishing, not structural correction.

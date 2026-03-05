# Sensors v14 Editorial Changes

Date: 2026-02-14

## Scope
- Manuscript file: for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213_v14.tex
- Change type: reviewer-defense tightening + figure upgrade
- Numeric table values: unchanged

## Applied Changes
1. Abstract wording tightened
- "highest packet delivery ratio" -> "highest mean packet delivery ratio".

2. Related work hardened with recent context
- Added references to recent context-aware/hybrid routing studies:
  ElFouly2023ILP, Dan2024EMRAR, Bhukya2025Hybrid.

3. NS-3 boundary clarified
- Explicitly states NS-3 block is AERIS-vs-LEACH only and is not used for five-protocol cross-platform ranking.

4. Limitation expanded
- Added explicit future-work item: full five-protocol cross-platform ranking remains pending.

5. Data availability strengthened
- Includes figure-generation code and provenance sidecar records.

6. Figure quality upgrade (s15)
- New figure set generated with improved hop-panel readability:
  local zoom for non-PEGASIS protocols in trade-off panel.
- Updated manuscript figure references to s15 assets.

## Validation Status
- Compile: pdflatex+bibtex+pdflatex+pdflatex completed.
- Gate: PASS (forbidden claims/path leakage).
- Consistency audit: total=124, warn=27, fail=0.
- Residual warning: 1 Underfull \\hbox (bibliography line wrapping).

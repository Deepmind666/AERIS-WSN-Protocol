# Sensors v13 Editorial Changes (Draft in Progress)

Date: 2026-02-13

## Scope
- Manuscript file: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213_v13.tex`
- Change type: editorial hardening only
- Data and numerical results: unchanged

## Applied Changes
1. Abstract tightening
- Replaced broad closing sentence with stricter bounded-claim sentence.
- Preserved all reported regimes and metric definitions.

2. Evidence scope tightening
- Added explicit statement that cross-regime numerical pooling is avoided.

3. Discussion compression
- Consolidated repeated wording and emphasized why 100-node and 100--1000-node blocks are reported separately.

4. Conclusion tightening
- Replaced generic boundary wording with explicit "sample-size regimes" boundary.

5. Data availability hardening
- Updated the Data Availability Statement to explicitly include provenance sidecar records
  (commit identifiers and script hashes) for reproducibility metadata.

## Citation Status
- v12 citation strengthening retained in v13:
  - `Kandris2020`
  - `Zuniga2004`
  - `Zuniga2007Asymmetry`
  - `Baccour2012RLQE`
  - `Chen2023Survey`
  - `Liu2018Environment`
  - `Zhao2019Context`

## Validation Status
- Full LaTeX compile (`pdflatex+bibtex+pdflatex+pdflatex`) completed.
- Gate check passed:
  - `scripts/check_sensors_draft_gate.py --draft for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213_v13.tex`
- Consistency audit completed:
  - `scripts/audit_sensors_v3_consistency.py --draft for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213_v13.tex --out-prefix 20260213_Sensors_v13_Data_Consistency_Audit`
- Current results:
  - Gate: PASS
  - Audit: total=124, warn=27, fail=0
  - Compile residual: 1 Underfull `\hbox` warning

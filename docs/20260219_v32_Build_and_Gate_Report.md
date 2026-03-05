# v32 Build and Gate Report (2026-02-19)

## Scope
- Draft file: for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260219_v32.tex
- PDF output: for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260219_v32.pdf

## Build
- Command sequence:
  - bibtex AERIS_Sensors_MDPI_Submission_Draft_20260219_v32
  - pdflatex -interaction=nonstopmode -halt-on-error AERIS_Sensors_MDPI_Submission_Draft_20260219_v32.tex
  - pdflatex -interaction=nonstopmode -halt-on-error AERIS_Sensors_MDPI_Submission_Draft_20260219_v32.tex
- Result: PASS
- Output pages: 12
- Fatal errors: 0
- Notes: only Underfull hbox warnings remain (non-blocking).

## Gate Check
- Command:
  - python scripts/check_sensors_draft_gate.py --draft for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260219_v32.tex
- Result: PASS
- Forbidden-claim hits: 0
- Path-pollution hits: 0

## Claim Matrix Recheck
- Command:
  - python scripts/validate_claim_source_matrix.py --matrix docs/20260215_v19_claim_source_matrix_v3.csv --output docs/20260219_v32_claim_matrix_recheck.txt
- Result summary: PASS=78, FAIL=0, SKIP=2, TOTAL=80

## NS-3 Claim Gate Consistency
- Checked file: ns3_validation/results/NS3_CLAIM_GATE.md
- Current wording is consistent with 25/28 (no 26/28 residue found).

## Status
- v32 is internally consistent for submission-draft review.

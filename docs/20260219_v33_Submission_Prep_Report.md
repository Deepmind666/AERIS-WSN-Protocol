# v33 Submission Preparation Report (2026-02-19)

## 1) Manuscript Update
- Source updated: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260219_v33.tex`
- Main adjustment: abstract now explicitly states the single non-significant S10 case
  (LEACH at indoor_office, 1000 nodes), keeping the `59/60` statement auditable.

## 2) Build Status
- Main draft build: PASS
  - Output: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260219_v33.pdf`
  - Toolchain: `pdflatex -> bibtex -> pdflatex -> pdflatex`
  - Blocking errors: none
  - Non-blocking warnings: minor underfull hbox warnings only.

## 3) Gate Status
- Draft gate command:
  - `python scripts/check_sensors_draft_gate.py --draft for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260219_v33.tex`
- Result: PASS
  - Forbidden claim hits: 0
  - Path pollution hits: 0

## 4) Submission Bundle
- Bundle path: `for_submission/submission_bundle_v33_20260219`
- Included:
  - v33 tex + pdf
  - `bibliography.bib`
  - MDPI definition files (`mdpi.cls`, `mdpi.bst`, `journalnames.tex`, logo pdf)
  - Five referenced figure PDFs used by v33
  - SHA256 manifest:
    - `for_submission/submission_bundle_v33_20260219/submission_bundle_v33_20260219_manifest_sha256.csv`
- Standalone bundle build: PASS

## 5) Consistency Notes
- NS-3 claim-gate wording is consistent with `25/28` throughout
  (`ns3_validation/results/NS3_CLAIM_GATE.md`).
- S9 table values in v33 remain aligned with
  `results/mega_experiments/s9_matched_4env_patch_vs_control_20260216_merged.csv`.

## 6) Ready State
- v33 is ready for formal external review and teacher submission use.

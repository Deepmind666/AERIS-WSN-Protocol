# 2026-02-18 v31 Major-Fix Log

## Scope
Manuscript file fixed:
- `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260218_v31.tex`

## Fixed Items (from strict review)

### P0-1: S9 table source mismatch
- Section/Table: `tab:s9_patch_control`
- Action: Replaced four inconsistent values with values directly from:
  - `results/mega_experiments/s9_matched_4env_patch_vs_control_20260216_merged.csv`
- Corrected rows:
  - `indoor_factory, 100, patch: 0.9543 -> 0.9281`
  - `indoor_factory, 500, patch: 0.8413 -> 0.9060`
  - `outdoor_urban, 100, patch: 0.7943 -> 0.7482`
  - `outdoor_urban, 500, patch: 0.4067 -> 0.7299`

### P0-2: PEGASIS S11 delta=0 interpretation
- Section: S9/S11 discussion paragraph
- Action: Added explicit statement that PEGASIS exact-zero delta in `indoor_factory` (all six node scales) is treated as an implementation-coupling anomaly pending code-path audit, not as physical invariance evidence.

### P1: Abstract length
- Action: Rewrote abstract to concise version.
- Validation: word count reduced from 255 to 180 words (<= 200).

### P1: S8 non-physical upward trend explanation
- Action: Added root-cause hypothesis:
  - Missing explicit MAC-layer contention penalties in S8 path can inflate delivery probability at higher density.

### P1: Data availability formatting
- Action: Replaced overly long file-name list with grouped evidence categories to avoid line overflow and improve readability.

## Verification
- Build commands:
  - `pdflatex` -> `bibtex` -> `pdflatex` x2
- Output PDF:
  - `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260218_v31.pdf`
- Result:
  - Compile success, no blocking errors.
  - Remaining warnings are layout-level underfull hbox only.

## Notes
- `ns3_validation/results/NS3_CLAIM_GATE.md` currently already reports `25/28` significant comparisons, so no additional correction was needed for that file in this pass.

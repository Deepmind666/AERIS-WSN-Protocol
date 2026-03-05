# 20260228 v84 Harsh Self-Review (Sensors/APIN style)

## Scope
- Target: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260228_v84.tex`
- Build: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260228_v84.pdf`
- Method: static manuscript checks + gate checks + consistency scan.

## Hard checks executed
1. Draft gate: `scripts/check_sensors_draft_gate.py --draft ...v84.tex` -> PASS
2. Citation integrity: cited keys vs `bibliography.bib` -> 36/36 present, missing=0
3. Figure references: 9 includes -> all files exist in `for_submission/figures`
4. Terminology drift scan:
   - `Stress block` occurrences: 0
   - `stress-comparison` occurrences: 0
   - `matched confirmation matrix` occurrences: 0

## Reviewer-style findings
### P0 (blocking)
- None.

### P1 (must-fix before submission if aiming for low-risk review)
1. **Figure readability risk remains high for dense panels**.
   - Affected figures likely: stress/power multi-line panels and heatmap overlays (Fig 5/6/7/8 family).
   - Risk: low-PDR baseline curves remain visually compressed; reviewers can question interpretability even when data are correct.
   - Action: produce a publication-clean variant with split panels or small multiples for baseline-only views.

2. **Figure-to-claim interpretability still depends on caption density**.
   - Long captions carry essential caveats (e.g., inferential pair vs descriptive context).
   - Risk: readers miss caveat and interpret overlays as equal-level inference.
   - Action: move one-sentence inferential-scope note into body text immediately before each figure reference.

### P2 (optional but recommended)
1. Add one-line visual legend note for significance marker semantics in all multi-panel inferential figures.
2. Harmonize capitalization style for matrix naming in subsection titles/captions if journal copy-edit requires stricter title case.

## Verdict
- **Minor Revision** (technical consistency and gate checks pass; primary residual risk is figure readability/interpretability, not data integrity).

## Submission risk note
- Data integrity risk: low.
- Wording drift risk: low (fixed in v84).
- Visual acceptance risk: medium (depends on final figure legibility in print-scale review).

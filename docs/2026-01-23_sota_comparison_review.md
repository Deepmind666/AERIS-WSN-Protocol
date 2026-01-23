# 2026-01-23 SOTA Figure (6‑panel) Strict Review & Fix Log

## Scope
- Target: `for_submission/figures/sota_comparison_6panel.*`
- Focus: panel (c) anomalies, panel (f) readability, and overall visual rigor.

## Fixes Applied (2026-01-23)
1. **Panel (c) now shows all comparisons (10 points)**
   - Baselines: LEACH / HEED / PEGASIS / SEP / TEEN
   - Profiles: AERIS‑E and AERIS‑R
   - Added **paired “dumbbell” connectors** between AERIS‑E and AERIS‑R per baseline.
   - Removed any “vs” labeling from y‑axis (now plain protocol names).
2. **Panel (f) table readability**
   - Increased table font size and scaling.
   - Adjusted column widths and layout to prevent crowding.
   - Title typography normalized with proper “Shapiro–Wilk”.
3. **Layout spacing**
   - Adjusted GridSpec ratios and subplot margins to reduce crowding.
   - Reduced legend collision risk in panel (c).

## Reviewer‑Style Assessment (Strict)
- **Scientific validity:**
  - AERIS‑E underperforms LEACH/HEED/PEGASIS/SEP in this dataset (negative ΔPDR); only TEEN is worse. This is *not acceptable* unless the paper explicitly positions AERIS‑E as a low‑energy conservative profile or an ablation.
  - AERIS‑R shows strong positive ΔPDR, but energy cost is higher. This must be framed as a reliability‑focused profile, not a universal win.
- **Interpretability:**
  - Panel (c) is now interpretable (two profiles per baseline) and no longer “meaningless”.
  - Panel (f) is readable at paper scale.
- **Remaining risks:**
  - Results are still based on a single scenario/geometry; reviewers may demand multi‑dataset validation.
  - The paper must explicitly acknowledge AERIS‑E’s trade‑off to avoid over‑claiming.

## Output Files (updated)
- `for_submission/figures/sota_comparison_6panel.pdf`
- `for_submission/figures/sota_comparison_6panel.svg`
- `results/publication_figures/sota_comparison_6panel.*`

## Next Steps (if requested)
- Re‑run SOTA comparison under additional datasets to test robustness.
- Re‑balance AERIS‑E hyperparameters to avoid underperforming classical baselines.


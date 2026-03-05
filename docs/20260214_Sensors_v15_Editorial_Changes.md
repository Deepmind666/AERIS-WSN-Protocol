## Sensors v15 Editorial Changes

Date: 2026-02-14
Scope: Final wording tightening on top of v14 with no metric changes.

### Updated file
- for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213_v15.tex

### Changes applied
1. Abstract wording tightened to avoid broad ranking claims:
   - "highest PDR in three of four environments"
   - "second-highest in indoor office"
2. Statistical interpretation note added:
   - Large-n cells should be interpreted jointly by Holm-adjusted p-values, effect size, and absolute PDR delta.
3. Conclusion wording aligned with scope:
   - "highest mean PDR in three environments" replaces broader phrasing.

### Verification
- PDF compiled successfully:
  - for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213_v15.pdf
- Gate check passed:
  - scripts/check_sensors_draft_gate.py
- Consistency audit passed (fail=0):
  - docs/20260214_Sensors_v15_Data_Consistency_Audit.csv
  - docs/20260214_Sensors_v15_Data_Consistency_Audit.md

### Additional verification round
- Recompiled after reviewer-oriented wording refinement:
  - for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213_v15.pdf
- Gate check remains PASS:
  - scripts/check_sensors_draft_gate.py
- Consistency audit remains fail=0:
  - docs/20260214_Sensors_v15b_Data_Consistency_Audit.csv
  - docs/20260214_Sensors_v15b_Data_Consistency_Audit.md

### Figure style upgrade (s16)
- Figure generator updated:
  - scripts/build_sensors_figures_s15.py
  - changes: soft low-saturation palette, white background, sans-serif labels, Computer Modern math, rounded bars.
- New figure set generated:
  - for_submission/figures/fig1_env_pdr_panel_20260214_s16.pdf
  - for_submission/figures/fig2_ablation_panel_20260214_s16.pdf
  - for_submission/figures/fig3_scalability_panel_20260214_s16.pdf
  - for_submission/figures/fig4_tradeoff_panel_20260214_s16.pdf
- Draft switched from s15 figures to s16 figures:
  - for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213_v15.tex

### Reference reality fast-check
- Added script:
  - scripts/audit_used_references_fast.py
- Fast-check outputs:
  - docs/20260214_v15_used_refs_fastcheck.csv
  - docs/20260214_v15_used_refs_fastcheck.md
- Result summary:
  - 19/19 cited references: probable_real (Crossref title query match).

### Figure style refinement (s17)
- Added a softer publication style variant:
  - scripts/build_sensors_figures_s17.py
- Core style refinements:
  - lower-saturation palette (soft blue + soft orange family)
  - white background and light gray grid
  - sans-serif labels + Computer Modern math text
  - rounded bars for non-matrix bar panels
- New figure set:
  - for_submission/figures/fig1_env_pdr_panel_20260214_s17.pdf
  - for_submission/figures/fig2_ablation_panel_20260214_s17.pdf
  - for_submission/figures/fig3_scalability_panel_20260214_s17.pdf
  - for_submission/figures/fig4_tradeoff_panel_20260214_s17.pdf
- Draft switched from s16 to s17 figures:
  - for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213_v15.tex

### DOI-hard audit (used references only)
- Added script:
  - scripts/audit_used_references_doi_hard.py
- Outputs:
  - docs/20260214_v15_used_refs_doi_hard.csv
  - docs/20260214_v15_used_refs_doi_hard.md
- Result summary:
  - Initial hard-check identified DOI integrity issues and triggered the repair round below.

### Reference integrity repair (post-audit)
- Fixed DOI/metadata mismatches in:
  - for_submission/bibliography.bib
  - keys updated: `Rault2016Energy`, `Kandris2020`, `Ren2024`, `Okine2024`, `Chen2023Survey`
- Removed unresolved citations from current draft text:
  - for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213_v15.tex
  - `Liu2018Environment`, `Zhao2019Context` no longer cited in the v15 submission draft.
- Recompiled bibliography pipeline:
  - `bibtex` + `pdflatex` x2 on `AERIS_Sensors_MDPI_Submission_Draft_20260213_v15.tex`
- Re-audit outputs (used references only):
  - docs/20260214_v15_postfix_used_refs_fastcheck.csv
  - docs/20260214_v15_postfix_used_refs_fastcheck.md
  - docs/20260214_v15_postfix_used_refs_doi_hard.csv
  - docs/20260214_v15_postfix_used_refs_doi_hard.md
- Final reference status for the current draft:
  - `verified_doi=17`
  - `doi_title_mismatch=0`
  - `unresolved_network=0`

### Post-fix manuscript checks
- Gate check:
  - `python scripts/check_sensors_draft_gate.py --draft for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213_v15.tex`
  - Result: PASS
- Consistency audit:
  - docs/20260214_Sensors_v15e.csv
  - docs/20260214_Sensors_v15e.md
  - Result: fail=0, warn=27

### Figure style refinement (s18)
- Added final style pass script:
  - scripts/build_sensors_figures_s18.py
- Refinements applied:
  - lower-saturation protocol palette
  - panel-specific y-axis ranges for readability without data overlap
  - log-scale hop-latency ranking (removed inset block)
  - preserved white background + Sans-Serif labels + CM math glyphs
- New figure set:
  - for_submission/figures/fig1_env_pdr_panel_20260214_s18.pdf
  - for_submission/figures/fig2_ablation_panel_20260214_s18.pdf
  - for_submission/figures/fig3_scalability_panel_20260214_s18.pdf
  - for_submission/figures/fig4_tradeoff_panel_20260214_s18.pdf
- Draft switched from s17 to s18:
  - for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213_v15.tex
- Final audit after s18 switch:
  - docs/20260214_Sensors_v15f.csv
  - docs/20260214_Sensors_v15f.md
  - Result: fail=0, warn=27

# 20260228 v76 Figure Polish QA

## Scope
- Source script: `scripts/build_sensors_figures_s76.py`
- Manuscript: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260228_v76.tex`
- Compiled output: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260228_v76.pdf`

## Changes from v75
1. Increased global plotting typography floor (axis/title/legend) for print readability.
2. Fig1 (`fig1_env_pdr_panel_20260228_s76.pdf`):
   - Kept outdoor-urban low-range inset.
   - Added explicit inset title `Inset: 0-0.16 PDR` and low-range background shading in the main panel.
   - Updated manuscript caption to explain inset semantics.
3. Fig3 (`fig3_scalability_panel_20260228_s76.pdf`):
   - Slightly larger canvas.
   - Removed extra bottom footnote text from inside figure to avoid vertical compression.
   - Moved shared legend slightly upward to reduce clipping/compression.
4. Fig7 (`fig7_ns3_trend_panel_20260228_s76.pdf`):
   - Larger canvas and improved legend placement for context-line readability.
5. Fig10 (`fig10_s10_absolute_profiles_20260228_s76.pdf`):
   - Larger canvas.
   - Removed bottom footnote line inside figure to preserve plotting area.
   - Kept split design: top row AERIS full-scale, bottom row baseline zoom.

## Integrity checks
- Figure assets generated: 12/12 (fig0..fig11 with suffix `s76`).
- Manuscript figure references updated from `_s75` to `_s76`.
- LaTeX compile: PASS (`latexmk -gg -pdf -interaction=nonstopmode -halt-on-error`).
- Unresolved references: none after final compile pass.

## Remaining known limitations
- Dense multi-panel figures (especially Fig5/Fig6) remain information-heavy due required full coverage (4 env x 5 protocols x 6 nodes x 3 tx dimensions).
- If journal feedback requires larger in-panel labels, split Fig6 into two separate figures (AERIS row vs baselines row) is the next non-invasive option.

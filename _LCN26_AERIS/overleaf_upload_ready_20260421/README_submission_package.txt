AERIS LCN 2026 Submission Package

Contents:
- aeris_lcn2026.tex
- aeris_lcn2026.pdf
- refer.bib
- IEEEtran.cls
- IEEEtran.bst
- figures/

Figures included:
- fig0_aeris_workflow_temp_20260420.pdf
- fig_lcn26_ns3_canonical_compact.pdf
- fig_lcn26_strict_compact.pdf
- fig_lcn26_ablation_compact.pdf
- fig_lcn26_mechanism_compact.pdf
- fig_lcn26_tradeoff_cv.pdf

Compile sequence:
1. pdflatex -interaction=nonstopmode aeris_lcn2026.tex
2. bibtex aeris_lcn2026
3. pdflatex -interaction=nonstopmode aeris_lcn2026.tex
4. pdflatex -interaction=nonstopmode aeris_lcn2026.tex

Notes:
- The package is self-contained and does not require files outside this directory.
- The included PDF is the current compiled manuscript from the same source package.
- `reference_audit/` contains local citation-audit materials copied here for convenience.
- Do not upload `reference_audit/` to Overleaf unless you explicitly want those audit files in the project.

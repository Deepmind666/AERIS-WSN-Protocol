# v93 Reviewer Self-Check (Reject-Oriented)

Date: 2026-03-01  
Scope: `v92 -> v93` editorial/figure closure only (no new experiments, no algorithm changes).

## P0 Gate (Blocking)

| Check | Result | Evidence |
|---|---|---|
| Compilation succeeds with halt-on-error | PASS | `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260301_v93.pdf` generated via `latexmk -gg -pdf -halt-on-error` |
| Unresolved references/citations | PASS | `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260301_v93.log` contains no unresolved-reference warnings in final pass |
| Figure asset missing | PASS | All `\\includegraphics` targets exist (`10/10`) and point to `*_20260301_s93.pdf` |
| Data matrix coverage broken in Fig6 | PASS | `s10r_4env_significance_tx5_vs_tx10_vs_tx15_20260227.csv`: `360/360` cells (4 env x 3 pairs x 6 nodes x 5 protocols) |
| Strongest-baseline traceability missing in Fig8 | PASS | Figure text includes baseline tags (L/P/H/T), with mapping documented in `docs/20260301_v93_figure_traceability.csv` |

## P1 Gate (Should-fix)

| Check | Result | Action |
|---|---|---|
| NS-3 inferential boundary ambiguous | PASS | Figure caption + surrounding text explicitly restrict significance to AERIS vs LEACH; context overlays marked descriptive only |
| Figure read-order ambiguity (Fig6/Fig8/Fig9) | PASS | Added one-line read instructions around these figures in `v93.tex` |
| Mixed suffix references (`s92` + `s93`) | PASS | Global figure references switched to `s93` only |

## P2 Notes (Non-blocking)

1. `fig9_s9_s11_consistency_20260301_s93.*` remains generated as a diagnostic asset but is not included in the main manuscript figures.
2. Figure-0 source path in script still falls back to exported SVG/PNG when draw.io CLI export is unavailable; this does not affect compiled manuscript output.

## Quick Numeric Sanity Snapshot

- Table 1 source check sample: indoor_office-AERIS mean/std from `env_sensitivity_20260207_205317.json` = `0.9739 / 0.0048` (n=30), consistent with manuscript rounding.
- Table 3 and Table 4 values are unchanged from v92 baseline (no statistic edits in v93).

## Verdict

- **P0 = 0**
- **P1 = 0**
- **P2 = 2 (non-blocking)**

`v93` is suitable for external strict review (Claude/APIN/Sensors style) under the frozen-data policy.

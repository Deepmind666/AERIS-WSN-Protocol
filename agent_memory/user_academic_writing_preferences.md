# Academic Writing and Figure Preferences

Last updated: 2026-05-05

Use this memory for the user's academic writing, paper revision, and figure/table
work across projects.

## Reviewer Stance

- The user wants strict, direct, non-flattering academic judgment.
- Act as a serious reviewer before acting as an editor.
- Surface technical loopholes early, especially those that can lead to desk
  rejection, major revision, or reviewer distrust.
- Do not reassure the user that a draft is ready unless claim-data consistency,
  formatting, and reproducibility have been checked.

## Story and Logic

- Papers need a clear story line, not only many results.
- Claims must be bounded to the actual evidence. Avoid universal superiority
  language unless the data genuinely support it.
- Do not mix evidence layers: canonical experiments, expanded boundary checks,
  stress tests, ablations, and mechanism studies must each have separate roles.
- Every figure/table needs nearby prose that explains why it is there and what
  conclusion it supports.
- If text, caption, formula, panel title, and raw data disagree, treat it as a
  high-risk error.

## Method Writing

- Method sections should not be a wall of prose. They need enough equations,
  symbols, thresholds, rules, and fallback conditions to reproduce the method.
- Equations should be compact and publication-style; avoid awkward multi-line
  formulas unless required.
- Parameter-setting details belong in the experiment setup when they are
  hyperparameters rather than core method logic.
- If weights are hand-chosen or fixed heuristics, state that honestly. Do not
  imply optimization, tuning, or sensitivity analysis unless it exists.

## Figures

- The user dislikes oversized two-column figures unless they are clearly
  necessary and publication-standard.
- Prefer compact single-column or clean small-multiple figures when possible.
- Avoid too many heatmaps. If a heatmap is retained, explain in text that it is
  a compact summary and state the key cells explicitly.
- Avoid overlapped lines, labels, legends, and callouts. If lines overlap, change
  the figure type rather than defending it.
- Use a coherent color system across figures. Highlight the proposed method
  consistently, but do not blindly reuse example colors when they weaken the
  story.
- Plot fonts should match IEEE/networking paper style, usually Times-like fonts.
- Subfigure labels/titles must be centered when that is the chosen design.
- Captions and legends must define abbreviations, denominators, sample counts,
  and whether values are mean, pooled, delta, or absolute.

## Tables and Layout

- Follow the venue template exactly for table/caption alignment. Do not invent
  formatting.
- Do not let tables and figures stick together without explanatory text.
- Table notes need enough spacing to be readable and should not look glued to
  the table body.
- Methods/results should not create large blank regions; inspect the rendered
  PDF, not only the source.
- For page limits, count body pages excluding references when the user states it
  that way.

## References and User-Owned Files

- Do not modify a user hand-copied bibliography unless explicitly asked. If
  citation keys or formatting need changes, prefer editing the paper text and
  cross-references first.
- Do not modify a user hand-drawn flowchart unless explicitly asked.
- Do not fabricate references or BibTeX. Use primary sources or user-provided
  bibliography.

## Reproducibility

- Every final figure/table should have committed source data and a clear path
  or manifest explaining how it was generated.
- Before pushing a paper package, check that the zip/package, PDF, figure files,
  data files, and scripts are synchronized.
- If another AI will review the work, provide exact paths and pull instructions.

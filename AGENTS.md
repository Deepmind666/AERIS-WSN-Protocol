# Agent Rules for AERIS

Read this file before editing the repository.

## Project Role

This repo is now a clean handoff tree for:

- the final AERIS LCN 2026 submission package;
- figure/table reproducibility;
- source code and selected NS-3 evidence;
- patent-writing handoff on another computer.

## Hard Rules

- Do not reintroduce old Sensors/DCN drafts or temporary PDF render folders.
- Do not modify the final bibliography or hand-drawn/selected figures unless the
  user explicitly asks.
- Do not claim AERIS is universally best.
- Keep AERIS positioned as reliability-first, rule-based, auditable, and bounded
  by deployment conditions.
- Keep evidence layers separate: classical NS-3 audit, expanded seven-protocol
  boundary sweep, NS-3 ablation, strict-physics stress layer, and mechanism
  study.
- If a figure/table is changed, update its source data, generated output, and
  manifest together.

## Preferred Review Format

Use severity labels:

- `P0`: contradiction, wrong data mapping, unreproducible artifact, or likely
  rejection trigger.
- `P1`: misleading claim, missing setup detail, baseline fairness issue, or
  serious layout risk.
- `P2`: style, readability, or minor formatting issue.

## Patent Work

For patent drafting, use `docs/PATENT_WRITING_HANDOFF.md`. The patent should
focus on the technical solution and effects, not on paper-style reviewer
language.

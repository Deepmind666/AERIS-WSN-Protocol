# User Academic Writing Preferences

Last updated: 2026-05-05

This file records stable user preferences for academic paper writing, figures,
layout, and review style so future Claude sessions can avoid re-learning them.

## READ THIS FIRST - MoE Thesis Current State 2026-05-19

- For the Astra-Sim MoE thesis project, first read `C:\astra-sim\.claude\rules\moe-current-effective-state-20260519.md`.
- Current repo: `C:\astra-sim\SimAI-moe-cluster-modeling`, branch `lkr_毕设`.
- Current thesis DOCX: `C:\astra-sim\experiments\毕业论文\面向MoE 推理的动态负载建模与仿真评估研究.docx`.
- Current project root: `C:\astra-sim\SimAI-moe-cluster-modeling\MoE_Workload_Simulation`.
- Older Q1-Q6, `load_generator_v2`, `experiments/modeling`, and April V3 figure-package memories are historical unless the user explicitly asks for them.

## Review Style

- Be strict, concrete, and honest. The user wants real reviewer risk, not praise.
- Put serious technical and formatting problems first.
- Use P0/P1/P2 severity when reviewing a paper:
  - P0: contradiction, unreproducible result, fatal claim-data mismatch, or
    likely rejection trigger.
  - P1: misleading claim, weak fairness explanation, missing setup detail, or
    risky formatting issue.
  - P2: style, flow, polish, or minor layout issue.

## Paper Logic

- A strong paper needs a clear story, not a pile of figures.
- Separate evidence layers and never mix their roles.
- Each result must answer a question and connect back to the story.
- Text, formulas, captions, panel titles, tables, and raw data must use the same
  denominator, baseline set, and metric definition.
- Bounded claims are preferred over aggressive claims.

## Methods

- Method sections should include enough equations and definitions for
  reproducibility.
- Avoid long dry prose without formulas or clear rules.
- Keep parameter settings and experiment hyperparameters in the experiment setup
  unless they are part of the algorithm definition.
- If parameters are heuristic or hand-selected, say so directly.

## Figures and Tables

- Figures must be readable in the final PDF, not just in isolation.
- Avoid oversized two-column figures unless necessary.
- Avoid overlapped lines; if a line chart overlaps badly, switch to a clearer
  figure type.
- Do not overuse heatmaps. If a heatmap remains, explain the key cells in prose.
- Use coherent colors across the paper and make the proposed method visually
  identifiable.
- Use Times-like fonts in academic plots.
- Keep subfigure titles/labels centered when the design uses centered titles.
- Do not let tables and figures stick together without prose between them.
- Follow the venue template for caption/table alignment exactly.

## File Safety

- Do not modify a user hand-copied `.bib` unless explicitly requested.
- Do not modify a user hand-drawn flowchart unless explicitly requested.
- When preparing a submission package, ensure PDF, source, figures, data, scripts,
  and zip/package are all synchronized.

## Deadline Mode

- When time is short, prioritize contradiction removal, claim narrowing,
  reproducibility checks, and obvious layout fixes.
- Do not start large new experiments unless the user explicitly asks and the
  result can finish in time.

# AERIS WSN Protocol

This repository is the cleaned handoff tree for the AERIS LCN 2026 paper and
follow-up patent writing.

The main branch intentionally keeps only the current submission package,
reproduction code/data, and handoff documents. Old Sensors/DCN drafts, temporary
PDF renders, obsolete figure variants, and stale experiment dumps were removed
from the branch tree.

## What Is Included

```text
paper/
  LCN26_AERIS_overleaf/       final Overleaf-ready submission folder
  LCN26_AERIS_overleaf.zip    same package as a zip for upload

data/
  figure_reproduction/        source data and scripts for Fig. 2-Fig. 5/Table III
  LCN26_AERIS_fig2_fig5_data.zip

src/                          Python AERIS/WSN protocol code
ns3_validation/               NS-3 standalone validation harness and selected results
scripts/                      final LCN figure/analysis scripts only
configs/                      simulation configuration files
tests/                        smoke and protocol tests
docs/                         reproduction, handoff, and patent-writing notes
agent_memory/                 Codex/Claude rules and AERIS writing memory
patent_seed_20260306/         earlier patent seed material, for reference only
```

## Current Paper

Final manuscript package:

```text
paper/LCN26_AERIS_overleaf/
paper/LCN26_AERIS_overleaf.zip
```

Main PDF:

```text
paper/LCN26_AERIS_overleaf/aeris_lcn2026.pdf
```

Main TeX:

```text
paper/LCN26_AERIS_overleaf/aeris_lcn2026.tex
```

## Reproduce Figures

The compact figure data package is already unpacked:

```text
data/figure_reproduction/
```

To rebuild publication figures from the packed data:

```powershell
cd data/figure_reproduction
powershell -ExecutionPolicy Bypass -File .\rebuild_figures_from_pack.ps1
```

See:

```text
docs/REPRODUCTION_GUIDE.md
docs/FIGURE_DATA_MANIFEST.md
```

## Patent Writing Handoff

For another computer or another AI agent, start with:

```text
docs/PROJECT_HANDOFF_PROMPT.md
docs/PATENT_WRITING_HANDOFF.md
agent_memory/user_academic_writing_preferences.md
agent_memory/AERIS_project_memory.md
```

The intended patent-writing workflow is:

1. Pull this repository.
2. Load the user's local patent-writing skill.
3. Use the language level and structure of the user's MoE patent as the style
   target.
4. Use this repo for AERIS technical content, evidence boundaries, and diagrams.

## Core Scientific Position

AERIS should be described as a reliability-first, rule-based, auditable WSN
routing design. It improves delivery over classical WSN baselines in selected
harsh heterogeneous-channel regimes, but it is not a universal replacement for
collection-tree/RPL-style stacks and it pays a reliability-lifetime trade-off.

Do not overstate the Skeleton module: in the audited configuration it is mostly
a reserve fallback, while Gateway-assisted uplinks carry the main measured gain.

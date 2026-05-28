# Reproduction Guide

This guide covers the artifacts needed to reproduce the paper figures and audit
the main evidence chain.

## Environment

Recommended local Python setup:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

For the NS-3 harness, use the bundled NS-3.40 source archive:

```text
ns3_validation/ns-allinone-3.40.tar.bz2
```

The paper's NS-3 evidence is produced by a standalone harness, not by a full
standards-compliant RPL/CTP stack.

## Final Paper Package

```text
paper/LCN26_AERIS_overleaf/
paper/LCN26_AERIS_overleaf.zip
```

The Overleaf package contains:

- `aeris_lcn2026.tex`
- `aeris_lcn2026.pdf`
- `ref.bib`
- `IEEEtran.cls`
- `figures/`

## Figure Reproduction

The source data and plotting scripts for the final paper figures are in:

```text
data/figure_reproduction/
```

Run:

```powershell
cd data/figure_reproduction
powershell -ExecutionPolicy Bypass -File .\rebuild_figures_from_pack.ps1
```

Expected regenerated outputs:

```text
data/figure_reproduction/00_final_outputs/fig2_classical_margin.pdf
data/figure_reproduction/00_final_outputs/fig3_stress.pdf
data/figure_reproduction/00_final_outputs/fig4_ablation.pdf
data/figure_reproduction/00_final_outputs/fig5_mechanism.pdf
```

## NS-3 Evidence

Selected NS-3 evidence is retained in:

```text
ns3_validation/results/lcn26_ns3_audit_20260420_012811/
ns3_validation/results/lcn26_ns3_dual_combined_20260430_191527_191528/
ns3_validation/results/lcn26_ns3_ablation_combined_20260501_010355_011001/
```

These directories preserve the current paper's classical audit, expanded
seven-protocol boundary evidence, and ablation evidence.

## Important Boundaries

- The strict-physics Python stress layer is stress evidence, not canonical
  protocol ranking evidence.
- The expanded CTP/RPL-MRHOF baselines are collection-style baselines inside the
  standalone harness, not full LLN protocol stacks.
- TEEN results use a PDR-expected denominator and must be interpreted together
  with energy/lifetime behavior.

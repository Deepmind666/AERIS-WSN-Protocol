# Core Code Inventory (AERIS-WSN-Protocol)
**Date:** 2026-01-23
**Purpose:** Identify and summarize the *core* code paths for protocol logic, baselines, experiment runners, and figure generation.

---

## 1) Core Protocol Logic (Python)
**Primary implementation**
- `src/aeris_protocol.py` — Main AERIS protocol implementation (CAS + Skeleton + Gateway + Safety/Fairness). This is the *canonical* Python protocol path used by most experiments.

**Key subcomponents (called by `aeris_protocol.py`)**
- `src/cas_selector.py` — Context-Adaptive Switching (Direct / Chain / Two-Hop).
- `src/skeleton_selector.py` — Skeleton backbone construction and routing constraints.
- `src/gateway_selector.py` — Gateway selection logic and multi-criteria scoring.
- `src/multi_objective_gateway.py` — Gateway scoring variants / multi-objective selection.
- `src/fairness_metrics.py` — Jain’s fairness and related metrics.
- `src/adaptive_reliability.py` — Reliability hooks (fallback, redundancy, thresholds).
- `src/realistic_channel_model.py` — Log-normal shadowing & environment-aware link reliability.
- `src/improved_energy_model.py` — CC2420-style energy accounting / model parameters.
- `src/intel_dataset_loader.py` — Intel Lab trace loader & preprocessing.
- `src/benchmark_protocols.py` — Shared Node/Network types used across protocols.

**Variants / auxiliary**
- `src/enhanced_aeris_protocol.py` — Alternate AERIS variant (less used in recent runs).
- `src/simplified_cas.py` — Lightweight CAS variant.
- `src/distilled_cas_selector.py` — Distilled/learned CAS selector.

---

## 2) Baseline Protocols (Python)
**Canonical baselines used in comparisons**
- `src/baseline_protocols/leach_protocol.py`
- `src/baseline_protocols/heed_protocol.py`
- `src/baseline_protocols/pegasis_protocol.py`
- `src/baseline_protocols/teen_protocol.py`
- `src/baseline_protocols/sep_protocol.py`

**Note:** these are the primary baseline implementations used in the SOTA comparisons and paper figures.

---

## 3) Core Experiment Runners (Python)
**Main entry points for experiments & JSON outputs**
- `scripts/run_sota_comparison.py` — Generates SOTA comparison data (n=30 runs), used for 6‑panel figure.
- `scripts/run_intel_replay.py` — Intel replay runs (real trace scenarios).
- `scripts/run_monte_carlo_uniform.py` — Monte Carlo topology tests.
- `scripts/run_dynamic_corridor_compare.py` — Dynamic corridor stress test.
- `scripts/run_dynamic_moving_bs_compare.py` — Moving BS scenario.
- `scripts/run_dynamic_dropout_compare.py` — Random dropout stress test.
- `scripts/run_large_scale_long.py` — Large-scale long runs (300/500 nodes).
- `scripts/run_scalability_experiment.py` — Scalability tests across node counts.
- `scripts/run_intel_ablation_parallel.py` — Parallel ablation runs (Intel).
- `scripts/run_intel_sensitivity_parallel.py` — Parallel sensitivity runs (Intel).

**Statistical post‑processing**
- `scripts/compute_dynamic_significance.py`
- `scripts/compute_monte_carlo_stats.py`
- `scripts/run_stats_multitest.py`
- `scripts/run_stats_bootstrap.py`

---

## 4) Core Figure Generation (Python)
**Paper-facing figure outputs**
- `scripts/plot_paper_figures.py` — Main paper figures
- `scripts/generate_sota_figures.py` — 6‑panel SOTA comparison figure
- `scripts/plot_pdr_breakdown_diagnostics.py` — PDR breakdowns
- `scripts/plot_dynamic_comparisons.py` — Dynamic scenario plots

**Output locations**
- `for_submission/figures/` — Paper-ready figures
- `results/plots/` — Intermediate plots
- `results/publication_figures/` — Batch-exported publication figures

---

## 5) NS‑3 Cross‑Validation (C++)
**Module path**
- `ns3_validation/src/aeris/model/aeris-protocol.cc`
- `ns3_validation/src/aeris/model/aeris-protocol.h`
- `ns3_validation/src/aeris/model/aeris-helper.cc`
- `ns3_validation/src/aeris/model/aeris-helper.h`

**Runner scripts**
- `ns3_validation/scripts/run_ns3_experiments.py`
- `ns3_validation/scripts/compare_results.py`

---

## 6) Paper Sources
- `for_submission/aeris_paper_final.tex` — Current main manuscript
- `for_submission/figures/` — Figures included in paper
- `references/main.bib` or `for_submission/bibliography.bib` — Bibliography

---

## 7) Recommended “Core Set” for Review
If a reviewer only reads the minimal set to understand the project:
1. `src/aeris_protocol.py`
2. `src/cas_selector.py`
3. `src/skeleton_selector.py`
4. `src/gateway_selector.py`
5. `src/realistic_channel_model.py`
6. `src/improved_energy_model.py`
7. `src/intel_dataset_loader.py`
8. `src/baseline_protocols/*`
9. `scripts/run_sota_comparison.py`
10. `scripts/generate_sota_figures.py`

---

## 8) Notes
- Several historical/legacy paths exist (e.g., `src/Enhanced-EEHFR-WSN-Protocol/`) and should be treated as *legacy*, not the core pipeline.
- The SOTA comparison figure is currently driven by `scripts/run_sota_comparison.py` + `scripts/generate_sota_figures.py`.


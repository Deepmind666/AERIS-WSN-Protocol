# AERIS v18 Stage Plan (Local + Server Split)

Date: 2026-02-15  
Owner split: Codex (local) + Claude (server)

## 1) Stage objective (for advisor progress review)

Deliver a reviewable Sensors-format draft that is technically honest and evidence-bounded, while continuing rigorous simulation upgrades in parallel.

- Draft track: stable, bounded claims from completed evidence.
- Experiment track: rigorous simulator fix + NS-3 strengthening.

## 2) Current evidence baseline used in v18

### Python (S8 unified, publication tier)
- File: `results/mega_experiments/scalability_4env_s8_unified_20260215_descriptive.csv`
- File: `results/mega_experiments/scalability_4env_s8_unified_20260215_significance.csv`
- Matrix: 4 environments x 6 node counts x 5 protocols x n=1000 per cell.

### NS-3 (trend-level only)
- File: `ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md`
- Scope: AERIS vs LEACH trend validation; numerical equivalence not claimed.

## 3) Work split and compute allocation

## 3.1 Codex (local machine)

### Tasks
1. Maintain manuscript mainline (`for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260215_v18.tex`).
2. Maintain figure mainline (`scripts/build_sensors_figures_s23.py`).
3. Prepare claim-gate and advisor-facing stage notes.
4. Run local smoke and pilot tests for simulator-rigor patch only (small matrix), not full production reruns.

### Resource policy
- CPU budget: 65% target, hard cap 80%.
- Memory budget: 70% target, hard cap 80%.
- Concurrency: adaptive; reduce workers when RAM > 75%.

### ETA (local)
- v18 editorial polish + figure consistency pass: 2-3 hours.
- local smoke/pilot for rigor fix (after code patch approval): 3-5 hours.

## 3.2 Claude (server)

### Tasks
1. Continue/execute long-run matrices and NS-3 expansion jobs.
2. Produce provenance + significance tables only; no manuscript text edits unless assigned.
3. Return strict completion packs (raw + stats + significance + sidecar + integrity check).

### Resource policy
- CPU budget: 80-90% allowed (server-only).
- Memory budget: <= 85% preferred; if > 90% sustained, reduce workers.
- Scheduling: long jobs in detached mode with checkpoint logs every 20-30 min.

### ETA (server)
- NS-3 extension with additional baselines at 3 scales (100/500/1000, n=30): 1-2 days (depends on compile/run queue).
- full rigor rerun after simulator patch (4 env x 6 nodes x 5 protocols x n=1000): 1-2 days wall time with split execution.

## 4) Quality gates (must pass before submission)

1. Claim gate: no forbidden claims (`100% PDR`, `numerical equivalence`, etc.).
2. Metric gate: all main claims tied to `pdr_expected` and explicit sample-size regime.
3. Reproducibility gate: every production JSON has provenance sidecar and consistent config hash.
4. Figure gate: no overlap, no clipping, low-saturation palette, white/light background, MDPI-friendly typography.

## 5) Next actions

### Immediate (today)
- [x] Publish advisor-view draft v18 PDF.
- [x] Switch scalability figure source to S8 unified matrix.
- [ ] Run one final manuscript consistency scan (claims vs tables vs captions).

### Next (after advisor feedback)
- [ ] Implement simulator-rigor patch in staged mode (P0/P1/P2), not all-at-once.
- [ ] Run pilot matrix and compare with NS-3 trend envelope.
- [ ] Decide go/no-go for full rerun.

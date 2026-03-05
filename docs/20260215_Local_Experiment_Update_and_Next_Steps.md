# Local Experiment Update and Next Steps (2026-02-15)

## 1) Local pilot completed (publication-tier, resource-capped)

- Matrix: 4 environments x nodes {100, 500, 1000} x 5 protocols x n=60
- Command policy used: `--max-cpu-percent 80 --max-mem-percent 80`
- Files:
  - `results/mega_experiments/pilot_rigor_pub_indoor_office_20260215_local.json`
  - `results/mega_experiments/pilot_rigor_pub_indoor_factory_20260215_local.json`
  - `results/mega_experiments/pilot_rigor_pub_outdoor_urban_20260215_local.json`
  - `results/mega_experiments/pilot_rigor_pub_outdoor_suburban_20260215_local.json`

## 2) Integrity and statistics artifacts

- Integrity check (expected cell n=60): PASS with warnings only (`git_dirty=True`)
  - Tool: `scripts/check_scalability_regime_integrity.py`
- Aggregated outputs:
  - `results/mega_experiments/pilot_rigor_pub_20260215_descriptive.csv`
  - `results/mega_experiments/pilot_rigor_pub_20260215_significance.csv`
  - `results/mega_experiments/pilot_rigor_pub_20260215_summary.md`

## 3) Key findings (for Go/No-Go gate)

- AERIS remains top-1 in all 12 cells (4 env x 3 node scales).
- Significance: 46/48 AERIS-vs-baseline comparisons are significant.
- Non-significant cells are both:
  - indoor_office @ 500: AERIS vs PEGASIS (p=0.1426)
  - indoor_office @ 1000: AERIS vs PEGASIS (p=0.0742)
- Physical-rigor issue remains:
  - AERIS PDR increases with node scale in 3/4 environments (factory/urban/suburban).
  - This still violates expected scalability monotonicity and confirms the need for simulator-rigor patches before final claims.

## 4) Figure pipeline fix applied

- Fixed fallback scalability source path in:
  - `scripts/build_sensors_figures_s23.py`
- Change:
  - `indoor_factory` now points to `scalability_indoor_factory_server_s8_20260215.json` (not old local S9 file).

## 5) Immediate next steps

### Local (Codex)
1. Implement rigor patch set in simulator (collision/load fairness and baseline fairness alignment).
2. Re-run the same pilot matrix first (n=60) under the same CPU/MEM caps.
3. Promote to larger run only if monotonicity and physical sanity improve.

### Server (Claude)
1. No heavy restart while node stability is uncertain.
2. Use existing NS-3/S8 evidence for manuscript support only (trend-level scope).
3. Prepare a resource-safe launch profile for future runs:
   - target CPU <= 75%
   - memory ceiling <= 80%
   - staged workers with backoff on memory pressure


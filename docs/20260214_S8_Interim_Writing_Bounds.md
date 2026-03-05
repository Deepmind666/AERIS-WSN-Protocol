# S8 Interim Writing Bounds (2026-02-14)

## Verified available files
- results/mega_experiments/scalability_outdoor_urban_server_s8_20260213.json (30000, error_runs=0)
- results/mega_experiments/scalability_outdoor_suburban_server_s8_20260213.json (30000, error_runs=0)
- results/mega_experiments/scalability_indoor_office_server_s8_20260213.json (30000, error_runs=1)
- results/mega_experiments/scalability_indoor_factory_local_small_s8.json (15000, nodes=100/200/300)
- results/mega_experiments/scalability_indoor_factory_local_500_s8.json (5000, nodes=500)

## Interim claim boundaries (before final indoor_factory 800/1000 merge)
- Allowed: per-environment statements for outdoor_urban/outdoor_suburban/indoor_office at nodes 100-1000.
- Allowed: mixed-regime wording with explicit sample-size caveat.
- Not allowed: final 4-environment unified significance statements for indoor_factory at 800/1000.
- Not allowed: pooled cross-environment single-number claims.

## Blocking deliverables required from server pipeline
- scalability_indoor_factory_server_s8_20260214.json
- scalability_indoor_factory_server_s8_20260214.provenance.json
- results/mega_experiments/s8_unified_20260214_descriptive.csv
- results/mega_experiments/s8_unified_20260214_significance.csv
- integrity-check pass report

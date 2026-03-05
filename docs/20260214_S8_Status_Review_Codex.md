## S8 Status Review (Codex)

Date: 2026-02-14
Scope: verify current local S8 artifacts before v16 manuscript merge.

### Machine-verified file state
1) Present and readable:
- results/mega_experiments/scalability_outdoor_urban_server_s8_20260213.json
- results/mega_experiments/scalability_outdoor_suburban_server_s8_20260213.json
- results/mega_experiments/scalability_indoor_office_server_s8_20260213.json

2) Missing:
- results/mega_experiments/scalability_indoor_factory_server_s8_20260214.json

3) Sidecar state:
- present:
  - results/mega_experiments/scalability_outdoor_urban_server_s8_20260213.provenance.json
  - results/mega_experiments/scalability_outdoor_suburban_server_s8_20260213.provenance.json
- missing:
  - results/mega_experiments/scalability_indoor_office_server_s8_20260213.provenance.json

### Verified metadata snapshot
- outdoor_urban:
  - raw_results=30000, error_runs=0, run_tier=publication, primary_metric=pdr_expected, git_commit=bf59e4a
- outdoor_suburban:
  - raw_results=30000, error_runs=0, run_tier=publication, primary_metric=pdr_expected, git_commit=bf59e4a
- indoor_office:
  - raw_results=30000, error_runs=1, run_tier=publication, primary_metric=pdr_expected, git_commit=b6b2e5e

### Integrity notes
1) indoor_office has metadata inconsistency:
- top-level error_runs=1
- but raw_results contains no explicit error rows
- per-cell counts are complete (30 cells, each n=1000)

2) Mixed commit risk remains:
- bf59e4a (outdoor files) vs b6b2e5e (indoor_office)
- must be explicitly documented in final provenance note.

3) Unified 4-environment S8 tables are not yet buildable locally:
- indoor_factory file not present
- cannot produce final s8_unified descriptive/significance CSV.

### What is safe to write now
- only partial statements for 3 environments based on available S8 files
- no full four-environment S8 claim
- no final Holm-adjusted four-environment significance claims

### Next gate before v16 merge
All conditions must be satisfied:
1) indoor_factory S8 JSON available
2) indoor_office + indoor_factory provenance sidecars available
3) unified S8 descriptive/significance CSV generated
4) regime integrity gate pass (or pass with one audited exception)

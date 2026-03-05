# Server-Side Reconciliation Report (v50-rigor)

Date: 2026-02-22
Auditor: Claude (server role)
Commit: `0dddcf4` (v50-rigor fairness fix)

## Reconciliation Summary

| Field | outdoor_urban | indoor_factory |
|-------|--------------|----------------|
| total_runs | 96000 | 96000 |
| error_runs | 0 | 0 |
| run_tier | publication | publication |
| primary_metric | pdr_expected | pdr_expected |
| git_commit | 0dddcf4 | 0dddcf4 |
| cell_check (30 cells x 3200) | PASS | PASS |
| data_sha256 | 8f55af...53c21 | 3ec187...f39d |

## AERIS PDR Means

| Nodes | outdoor_urban | indoor_factory |
|-------|--------------|----------------|
| 100 | 0.7479 | 0.9278 |
| 500 | 0.7304 | 0.9061 |
| 1000 | 0.1359 | 0.7284 |

## Provenance Sidecar Files

- `scalability_outdoor_urban_v50rigor_20260222_server.provenance.json` — verified
- `scalability_indoor_factory_v50rigor_20260222_server.provenance.json` — verified

## Verdict

Both server authoritative files pass all acceptance criteria. SHA256 hashes match provenance records. Cell completeness confirmed (30 cells x 3200 replicates each).

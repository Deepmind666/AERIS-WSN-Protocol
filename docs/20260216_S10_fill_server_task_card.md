# S10 Fill Server Task Card (CPU Target 90%)

## Goal
Complete missing S10 tx-sensitivity runs for:
- `indoor_office` (`tx=5/15`)
- `outdoor_suburban` (`tx=5/15`)

Then rebuild the full four-environment S10 bundle for manuscript sync.

## Run Script
Server-side script:
- `scripts/run_server_s10_fill_2env_90.ps1`

Core runtime parameters:
- `replicates=600`
- `nodes=100,500,1000`
- `workers=20`
- `max_cpu_percent=90`
- `max_mem_percent=88`
- `mac_collision=true`
- `multihop_relay=true`

Outputs:
- `scalability_indoor_office_server_s10_tx5_fill_20260216.json`
- `scalability_indoor_office_server_s10_tx15_fill_20260216.json`
- `scalability_outdoor_suburban_server_s10_tx5_fill_20260216.json`
- `scalability_outdoor_suburban_server_s10_tx15_fill_20260216.json`

## Postprocess Script
After all four fill JSON files are complete, run:
- `python scripts/postprocess_s10_full4env.py`

Expected outputs:
- `s10_4env_merged_descriptive_20260216.csv`
- `s10_4env_significance_tx5_vs_tx15_20260216.csv`
- `s10_4env_summary_20260216.md`
- sidecar `*.provenance.json` for missing fill files

## Acceptance Checks
1. Each new JSON has:
   - `raw_results=9000`
   - `error_runs=0`
   - `run_tier=publication`
   - `primary_metric=pdr_expected`
2. Full S10 merged file row count:
   - `4 env x 2 tx x 3 nodes x 5 protocols = 120`
3. Full significance row count:
   - `4 env x 3 nodes x 5 protocols = 60`

## ETA Basis
- Prior S10 2-env bundle (`4 groups`) took ~141 min at lower utilization.
- This fill run is also `4 groups`, with higher CPU target (90%).
- Practical ETA window:
  - Run stage: `100-140 min`
  - Postprocess + integrity check: `10-15 min`
  - Total: `110-155 min`

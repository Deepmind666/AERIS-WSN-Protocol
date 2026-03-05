# Claude Server Task Update - Outdoor Urban Takeover

Date: 2026-02-22

## Task

Run and own `outdoor_urban` for v50-rigor on server as authoritative result.

## Command (server)

```powershell
cd C:\Users\sshuser\AERIS-WSN
C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe scripts/run_scalability_experiment.py `
  --env outdoor_urban --replicates 3200 --seed 42001 `
  --nodes 100,200,300,500,800,1000 --rounds 300 --workers 20 `
  --run-tier publication --tx-power 10.0 `
  --max-cpu-percent 90 --max-mem-percent 90 --resource-check-sec 1 `
  --mac-collision --multihop-relay --allow-partial `
  --output results/mega_experiments/scalability_outdoor_urban_v50rigor_20260222_server.json
```

## Required outputs

1. `results/mega_experiments/scalability_outdoor_urban_v50rigor_20260222_server.json`
2. `results/mega_experiments/scalability_outdoor_urban_v50rigor_20260222_server.provenance.json`
3. Summary fields: `raw_results`, `error_runs`, `run_tier`, `primary_metric`, `git_commit`
4. AERIS PDR means at node counts 100/500/1000

## Acceptance

- `raw_results == 96000` (3200 replicates x 6 nodes x 5 protocols)
- `error_runs == 0` preferred (or explicit failed-cell list)
- `run_tier == publication`
- `primary_metric == pdr_expected`


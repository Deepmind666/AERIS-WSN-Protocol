# Claude Server Task Card (v50-rigor, 4h Window)

Date: 2026-02-22  
Owner: Claude (server)  
Purpose: keep server utilized for the next ~4 hours with non-duplicate rigor data.

## Task scope

Run **indoor_factory** v50-rigor full matrix on server (authoritative run).

- environment: `indoor_factory`
- replicates: `3200`
- nodes: `100,200,300,500,800,1000`
- protocols: `AERIS, LEACH, PEGASIS, HEED, TEEN`
- rounds: `300`
- seed: `42001`
- required flags: `--mac-collision --multihop-relay`

## Execution command (server)

```powershell
cd C:\Users\sshuser\AERIS-WSN
C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe scripts/run_scalability_experiment.py `
  --env indoor_factory --replicates 3200 --seed 42001 `
  --nodes 100,200,300,500,800,1000 --rounds 300 --workers 20 `
  --run-tier publication --tx-power 10.0 `
  --max-cpu-percent 90 --max-mem-percent 90 --resource-check-sec 1 `
  --mac-collision --multihop-relay --allow-partial `
  --output results/mega_experiments/scalability_indoor_factory_v50rigor_20260222_server.json
```

## Server usage reminders (must follow)

1. Use full Python path; do not depend on `conda activate` in one-line SSH calls.
2. Keep only one heavy scalability run active on server at a time.
3. Target utilization: CPU around `85-90%` (not 100% hard lock).
4. Write stdout/err logs if using `Start-Process`; keep run traceable.

## Expected 4-hour checkpoint

At t≈4h, report:

- current progress (`completed/30000`, `failed`)
- current throughput (tasks/s)
- remaining ETA based on current throughput

## Final deliverables

1. `results/mega_experiments/scalability_indoor_factory_v50rigor_20260222_server.json`
2. `results/mega_experiments/scalability_indoor_factory_v50rigor_20260222_server.provenance.json`
3. Metadata summary:
   - `raw_results`
   - `error_runs`
   - `run_tier`
   - `primary_metric`
   - `git_commit`
4. AERIS PDR means at `n=100/500/1000`.

## Acceptance criteria

- `raw_results == 96000` (3200 x 6 x 5)
- `error_runs == 0` (or explicit failed-cell list)
- `run_tier == publication`
- `primary_metric == pdr_expected`
- sidecar exists and includes data SHA256 + script SHA256 + config hash.

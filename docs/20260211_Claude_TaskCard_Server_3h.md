# Claude Task Card: Server 3-Hour Run (S7)

Date: 2026-02-11  
Owner: Claude (server execution)  
Window: ~3 hours  
CPU cap: <= 90%  
Memory cap: <= 70%

## Goal

Run a new server-side scalability batch for two environments with publication settings, so we can unify 4-env evidence at consistent scale.

## Do Not

- Do not modify `src/` in this task.
- Do not run any extra experiment not listed below.
- Do not edit manuscript files in this task.

## Experiment Commands (exact)

Run sequentially on server (same commit for both runs):

1) `indoor_office`:

```bash
python scripts/run_scalability_experiment.py \
  --replicates 550 \
  --workers 12 \
  --seed 42001 \
  --nodes 100,200,300,500,800,1000 \
  --rounds 300 \
  --env indoor_office \
  --tx-power 10.0 \
  --run-tier publication \
  --max-cpu-percent 90 \
  --max-mem-percent 70 \
  --resource-check-sec 2 \
  --output results/mega_experiments/scalability_indoor_office_server_s7_20260211.json
```

2) `outdoor_suburban`:

```bash
python scripts/run_scalability_experiment.py \
  --replicates 550 \
  --workers 12 \
  --seed 42001 \
  --nodes 100,200,300,500,800,1000 \
  --rounds 300 \
  --env outdoor_suburban \
  --tx-power 10.0 \
  --run-tier publication \
  --max-cpu-percent 90 \
  --max-mem-percent 70 \
  --resource-check-sec 2 \
  --output results/mega_experiments/scalability_outdoor_suburban_server_s7_20260211.json
```

## Post-Run Provenance

Generate sidecars immediately:

```bash
python scripts/generate_scalability_provenance.py \
  --files \
  results/mega_experiments/scalability_indoor_office_server_s7_20260211.json \
  results/mega_experiments/scalability_outdoor_suburban_server_s7_20260211.json
```

If `--files` is unsupported, use the script's equivalent single-file mode and produce two sidecars.

## Acceptance Criteria

For each JSON:

- `error_runs == 0`
- `run_tier == publication`
- `primary_metric == pdr_expected`
- `len(raw_results) == 16500` (550 * 6 * 5)
- `config.node_counts == [100,200,300,500,800,1000]`

For sidecars:

- 2 sidecar files exist (one per JSON)
- include `git_commit`, `git_dirty`, `git_diff_stat`, `script_sha256`, `config_hash`

## ETA Reporting (required)

At start and every 30 minutes, report:

- current stage
- elapsed time
- completed/total (from logs)
- remaining ETA and basis (throughput from latest progress lines)

## Final Report Format (strict)

1. File list (absolute paths)  
2. What was done  
3. What still needs verification  
4. Runtime summary (elapsed + ETA basis + resource usage range)

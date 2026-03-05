# v70 Night Run Status (2026-02-27)

## Local queue (Codex)

- Launcher: `scripts/start_local_s10r_queue_9h_force_20260227.ps1`
- Runner: `scripts/run_local_s10r_queue_9h_force_20260227.ps1`
- PID file: `logs/s10r_local_force9h_queue_launcher_20260227_002111.pid`
- Runtime log: `logs/s10r_local_force9h_queue_20260227_002111.log`
- Start time (T0): `2026-02-27 00:21:11`
- Stop rule: 9-hour cutoff for *new* tasks, but do not interrupt running task.

### Local task list (fixed order)

1. `indoor_office tx5`
2. `indoor_office tx10`
3. `indoor_office tx15`
4. `indoor_factory tx5`
5. `indoor_factory tx10`
6. `indoor_factory tx15`

### Local ETA basis

- Bootstrap estimate at launch: `~1h50m/task` and `~11h global` (based on previous local S10R long-tail rates near 3.7/s).
- Real-time task ETA source: `[Scalability] ... rate=... ETA=...` lines in the runtime log.
- At launch+few minutes snapshot, first task (`indoor_office tx5`) ran in `~21-23 min ETA` early stage and then slows in large-node stage.

## Server queue (Claude)

### Completed JSON snapshots

- `scalability_outdoor_urban_server_s10r_tx5_20260226.json`
- `scalability_outdoor_urban_server_s10r_tx10_20260226.json`
- `scalability_outdoor_urban_server_s10r_tx15_20260226.json`
- `scalability_outdoor_suburban_server_s10r_tx5_20260226.json`
- `scalability_outdoor_suburban_server_s10r_tx10_20260226.json`

### Remaining server task at snapshot

- `outdoor_suburban tx15` (plus server reconciliation/significance output).

## Acceptance checks to run after all 12 tasks

- `raw_results == 30000`
- `error_runs == 0`
- `run_tier == publication`
- `primary_metric == pdr_expected`

Then build:

- `results/mega_experiments/s10r_4env_merged_descriptive_20260226.csv`
- `results/mega_experiments/s10r_4env_significance_tx5_vs_tx10_vs_tx15_20260226.csv`
- `results/mega_experiments/s10r_4env_reconciliation_20260226.md`

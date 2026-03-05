# Claude Server Execution Card (v20260209)

Owner split:
- Codex: local run owner
- Claude: server run owner

## 1) Scope (server only)

Run only the remaining scalability environments:

1. indoor_office
2. outdoor_suburban

Do not run indoor_factory or outdoor_urban on server (already running locally).

## 2) Hard limits

- CPU <= 65%
- MEM <= 65%
- workers = 12

## 3) Commands (server)

Working directory:

C:\Users\sshuser\AERIS-WSN

Run command pattern:

conda run -n aether-wsn python scripts/run_scalability_experiment.py --nodes 100,200,300,500,800,1000 --replicates 550 --workers 12 --rounds 300 --env <ENV> --max-cpu-percent 65 --max-mem-percent 65 --run-tier publication --output results/mega_experiments/scalability_<ENV>_20260209_server550.json

Run both envs:

1) indoor_office
2) outdoor_suburban

After both runs finish:

python scripts/generate_scalability_provenance.py --overnight-dir results/mega_experiments

## 4) Acceptance

For each environment:

- error_runs == 0
- raw_results count == 16500
- run_tier == publication
- primary_metric == pdr_expected
- provenance sidecar exists

## 5) Return format (strict)

Return only:

1. full output file paths
2. elapsed time per environment
3. ETA basis used during run
4. metadata:
   - git_commit
   - git_dirty
   - git_diff_stat
   - script_sha256
   - config_hash
   - run_tier
   - primary_metric
5. 1000-node PDR ranking per environment

Do not modify src or manuscript in this task.

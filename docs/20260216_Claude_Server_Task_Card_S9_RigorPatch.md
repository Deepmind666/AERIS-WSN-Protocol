# 2026-02-16 Claude Server Task Card (S9 Rigor Patch Bundle)

## 0) Scope and Boundaries

- This card is **server-side only**.
- Do not modify `src/` in this task.
- Do not run unrelated experiments.
- Use existing scripts and keep outputs in `results/mega_experiments/`.
- Report in Chinese, concise, with exact file paths.

## 1) Goal

Build an 8-hour server bundle for method comparison:

1. Rigor-patch scalability in benign environments (indoor_office, outdoor_suburban).  
2. Matched control runs (same settings, no `--mac-collision` / no `--multihop-relay`).  
3. Produce a clean result set for direct delta analysis (`patched - control`) at each node scale.

## 2) Resource Policy (must obey)

- `workers`: start at 14.
- `--max-cpu-percent 88`
- `--max-mem-percent 82`
- If memory exceeds threshold repeatedly, reduce workers to 10 and restart that stage once.
- No tight polling loops. Check once per 20-30 minutes or at stage boundaries.

## 3) Execution Plan

Use base seed offsets to avoid accidental overlap.

### S9-A Patched runs (publication)

```bash
python scripts/run_scalability_experiment.py --env indoor_office --replicates 1000 --seed 62001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 10.0 --max-cpu-percent 88 --max-mem-percent 82 --mac-collision --multihop-relay --output results/mega_experiments/scalability_indoor_office_server_s9_patch_20260216.json

python scripts/run_scalability_experiment.py --env outdoor_suburban --replicates 1000 --seed 72001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 10.0 --max-cpu-percent 88 --max-mem-percent 82 --mac-collision --multihop-relay --output results/mega_experiments/scalability_outdoor_suburban_server_s9_patch_20260216.json
```

### S9-B Control runs (publication)

```bash
python scripts/run_scalability_experiment.py --env indoor_office --replicates 600 --seed 82001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 10.0 --max-cpu-percent 88 --max-mem-percent 82 --output results/mega_experiments/scalability_indoor_office_server_s9_control_20260216.json

python scripts/run_scalability_experiment.py --env outdoor_suburban --replicates 600 --seed 92001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 10.0 --max-cpu-percent 88 --max-mem-percent 82 --output results/mega_experiments/scalability_outdoor_suburban_server_s9_control_20260216.json
```

## 4) Expected Runtime (for coordination)

Estimated 7.5-9.0 hours total:

- S9-A indoor_office patch: ~2.5-3.0 h  
- S9-A outdoor_suburban patch: ~2.5-3.0 h  
- S9-B two control runs (600 replicates): ~2.0-3.0 h

Basis: previous S8/S9 logs with 30k-cell runs and heavy 800/1000-node tail behavior.

## 5) Required Post-processing

After all 4 JSON files finish:

1. Generate sidecars for each JSON (same metadata fields as existing S8 sidecars: commit, run_tier, primary_metric, script_sha256, config_hash, raw_results_count, error_runs).  
2. Build one merged comparison CSV:
   - columns: `environment,num_nodes,protocol,mode(patch/control),n,pdr_mean,pdr_std`
3. Build one delta CSV for AERIS and each baseline:
   - `delta = pdr_patch - pdr_control`
4. Run Welch + Hedges g + Holm correction for key comparisons and save significance CSV.

## 6) Acceptance Criteria

- 4 output JSON files exist and each has `run_tier=publication`, `primary_metric=pdr_expected`.
- No missing node/protocol cells.
- Sidecar files exist for all 4 JSON files.
- Merged descriptive CSV + delta CSV + significance CSV all generated.
- Report exact file paths and one-line interpretation per environment.

## 7) Final Reply Format (strict)

1. 文件清单（完整路径）  
2. 本次完成（每步一句）  
3. 仍需核对（最多3条）  
4. ETA复盘（实际耗时 vs 预估耗时）


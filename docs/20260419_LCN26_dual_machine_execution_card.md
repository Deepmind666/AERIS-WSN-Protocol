# 2026-04-19 LCN26 Dual-Machine Execution Card

Project: `C:\AERIS-WSN-Protocol`  
Goal: fix the suspicious NS-3 PEGASIS path, rerun a focused cross-platform audit, and add mechanism-focused AERIS evidence using both local and FatMachine compute.

## 1. Why We Are Rerunning

### P0: suspicious NS-3 PEGASIS office trend
- Current file: `ns3_validation/results/ns3_5proto_fullnodes_descriptive_20260226.csv`
- Symptom: PEGASIS in `indoor_office` stays near `1.0` across `50..1000` nodes with range about `0.0002`
- Root cause under audit: `PegasisProtocolNs3` uses a random leader proxy and scales `leader->BS` distance by `0.1`

### P1: missing mechanism-depth evidence
- Need AERIS-only evidence for:
  - CAS mode usage
  - Gateway uplink hit/success
  - Skeleton assignments/backbone use
  - first-node and half-node death

## 2. Output Layout

### Server / NS-3
- `ns3_validation/results/lcn26_ns3_audit_<timestamp>/raw/`
- `ns3_validation/results/lcn26_ns3_audit_<timestamp>/logs/`
- `ns3_validation/results/lcn26_ns3_audit_<timestamp>/ns3_focused_merged.json`
- `ns3_validation/results/lcn26_ns3_audit_<timestamp>/ns3_focused_descriptive.csv`
- `ns3_validation/results/lcn26_ns3_audit_<timestamp>/ns3_focused_significance.csv`
- `ns3_validation/results/lcn26_ns3_audit_<timestamp>/ns3_focused_summary.md`

### Local / mechanism
- `results/mega_experiments/lcn26_aeris_mechanism_<timestamp>/mechanism_raw.json`
- `results/mega_experiments/lcn26_aeris_mechanism_<timestamp>/mechanism_summary.csv`
- `results/mega_experiments/lcn26_aeris_mechanism_<timestamp>/mechanism_summary.md`

## 3. Machine Split

### FatMachine (server, WSL / NS-3 owner)
- Code fix target:
  - `ns3_validation/aeris-validation-standalone.cc`
- Run target:
  - focused 5-protocol audit matrix
- Matrix:
  - protocols: `AERIS, LEACH, HEED, PEGASIS, TEEN`
  - environments: `indoor_office, indoor_factory, outdoor_urban, outdoor_suburban`
  - nodes: `100, 500, 1000`
  - seeds: `42001..42030`
- Total:
  - `5 x 4 x 3 x 30 = 1800` experiments
- Runner:
  - `ns3_validation/run_lcn26_focused_matrix.sh`
- Merge:
  - `ns3_validation/merge_lcn26_focused_results.py`

### Local (Python mechanism owner)
- Run target:
  - focused AERIS-only mechanism matrix
- Matrix:
  - environments: `indoor_office, indoor_factory, outdoor_urban, outdoor_suburban`
  - nodes: `100, 500, 1000`
  - replicates: `400` recommended initial run
  - rounds: `300`
  - strict flag: `--mac-collision`
- Total:
  - `4 x 3 x 400 = 4800` runs
- Runner:
  - `scripts/run_lcn26_aeris_mechanism_matrix.py`
- Summary:
  - `scripts/summarize_lcn26_aeris_mechanism.py`

## 4. ETA Strategy

### Server ETA
- No trustworthy prior benchmark exists for the fixed focused NS-3 matrix
- Required:
  1. run a single shard smoke (`AERIS, indoor_factory, 100/500/1000`)
  2. measure elapsed time
  3. extrapolate `20 shards total`
- Initial planning estimate:
  - smoke shard: `5-15 min`
  - full focused matrix: `2-8 h`
  - merge/statistics: `<10 min`

### Local ETA
- Existing project notes suggest full scalability Python batches on local are much slower than server
- Initial planning estimate for `4800` AERIS-only runs:
  - `6-12 h` depending on node-size skew
- Required:
  1. run `--smoke` first
  2. record throughput after first `200` completed tasks
  3. update ETA from measured rate

## 5. Launch Order

1. Patch and sync the NS-3 source to FatMachine.
2. Build NS-3 binary on FatMachine.
3. Run one server smoke shard and one local smoke mechanism job.
4. Recalculate ETA from measured throughput.
5. Launch both full jobs so expected finish windows overlap as closely as possible.
6. When local finishes first, use local to summarize and begin manuscript updates.
7. When server finishes, merge focused NS-3 results and update canonical claims.

## 6. Commands

### Local smoke
```powershell
python scripts/run_lcn26_aeris_mechanism_matrix.py `
  --smoke `
  --envs indoor_office,indoor_factory,outdoor_urban,outdoor_suburban `
  --nodes 100,500,1000 `
  --replicates 400 `
  --workers 12 `
  --max-cpu-percent 80 `
  --max-mem-percent 80 `
  --mac-collision
```

### Local full
```powershell
python scripts/run_lcn26_aeris_mechanism_matrix.py `
  --envs indoor_office,indoor_factory,outdoor_urban,outdoor_suburban `
  --nodes 100,500,1000 `
  --replicates 400 `
  --workers 12 `
  --max-cpu-percent 80 `
  --max-mem-percent 80 `
  --mac-collision
```

### Local summary
```powershell
python scripts/summarize_lcn26_aeris_mechanism.py `
  --input results/mega_experiments/lcn26_aeris_mechanism_<timestamp>/mechanism_raw.json
```

### Server focused NS-3 merge
```bash
python ns3_validation/merge_lcn26_focused_results.py \
  --input-dir ns3_validation/results/lcn26_ns3_audit_<timestamp>/raw \
  --output-dir ns3_validation/results/lcn26_ns3_audit_<timestamp>
```

### Server wrappers from local PowerShell
```powershell
# 1. sync files to FatMachine
powershell -File scripts/sync_lcn26_rerun_to_fatmachine.ps1

# 2. rebuild NS-3 on server
powershell -File scripts/start_lcn26_server_ns3_build.ps1

# 3. launch focused NS-3 audit on server
powershell -File scripts/start_lcn26_server_ns3_audit.ps1

# 4. launch local mechanism matrix
powershell -File scripts/start_lcn26_local_mechanism.ps1
```

## 7. Acceptance

### NS-3 audit pass
- focused rerun completes `20/20` shards
- merged experiment count = `1800`
- `indoor_office PEGASIS` range is no longer near-flat for implausible reasons, or the audit note clearly explains why it remains flat
- new canonical claims are based on rerun files only

### Mechanism pass
- raw JSON has `4800` successful runs (or all expected runs for the chosen replicate count)
- summary includes:
  - `pdr_expected`
  - `first_node_death_round`
  - `half_nodes_death_round`
  - `gateway_uplink_*`
  - `skeleton_assignments`
  - `cas_mode_usage_stats`

## 8. High-Risk Points

1. Server WSL quoting and remote launch reliability.
2. NS-3 rebuild path mismatch between Windows host and WSL path.
3. Local long-run slowdown at `1000` nodes.
4. Dirty worktree: do not overwrite unrelated manuscript assets.

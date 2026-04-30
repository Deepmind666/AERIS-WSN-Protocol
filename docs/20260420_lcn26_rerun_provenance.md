# 2026-04-20 LCN26 Rerun Provenance

Project root: `C:\AERIS-WSN-Protocol`

## 1. Valid NS-3 audit to use in the paper

- Local synced path:
  - `ns3_validation/results/lcn26_ns3_audit_20260420_012811/summary/ns3_focused_descriptive.csv`
  - `ns3_validation/results/lcn26_ns3_audit_20260420_012811/summary/ns3_focused_significance.csv`
  - `ns3_validation/results/lcn26_ns3_audit_20260420_012811/summary/ns3_focused_summary.md`
- FatMachine build chain was corrected to copy
  - `C:\Users\sshuser\AERIS-WSN-Protocol\ns3_validation\aeris-validation-standalone.cc`
  - into WSL scratch:
  - `/home/ns3user/ns-allinone-3.40/ns-3.40/scratch/aeris-validation-standalone.cc`
- Confirmed synced scratch MD5 during valid rebuild:
  - `ad384bcbe24fcc0d10da0fda270aaa12`

## 2. Invalid NS-3 rerun to discard

- Remote output:
  - `C:\Users\sshuser\AERIS-WSN-Protocol\ns3_validation\results/lcn26_ns3_audit_20260420_010122`
- Reason invalid:
  - The old `server_build_lcn26_ns3.ps1` rebuilt NS-3 in WSL without copying the updated repo source into `scratch/`.
  - Therefore that rerun still used the stale WSL source and cannot be cited as the corrected PEGASIS audit.

## 3. Mechanism matrix to use in the paper

- Local synced path:
  - `results/lcn26_targeted_20260420/mechanism_grid_fat/mechanism_raw_merged.json`
  - `results/lcn26_targeted_20260420/mechanism_grid_fat/mechanism_summary.csv`
  - `results/lcn26_targeted_20260420/mechanism_grid_fat/mechanism_summary.md`
- Remote merged source:
  - `C:\Users\sshuser\AERIS-WSN-Protocol\results\lcn26_targeted_20260420\mechanism_grid_fat\merged_20260420_084617`
- Execution mode:
  - FatMachine Windows host
  - 12 cell jobs total
  - `env x nodes = 4 x 3`
  - `400` replicates per cell
  - `8` concurrent jobs max
  - each cell job ran `scripts/run_lcn26_aeris_mechanism_matrix.py` with `--workers 1`

## 3b. Targeted follow-up mechanism confirmation

- Local synced path:
  - `results/lcn26_targeted_20260421_followup_b/mechanism_grid_fat_targeted/merged_20260421_135826/mechanism_raw_merged.json`
  - `results/lcn26_targeted_20260421_followup_b/mechanism_grid_fat_targeted/merged_20260421_135826/mechanism_summary.csv`
  - `results/lcn26_targeted_20260421_followup_b/mechanism_grid_fat_targeted/merged_20260421_135826/mechanism_summary.md`
- Remote merged source:
  - `C:\Users\sshuser\AERIS-WSN-Protocol\results\lcn26_targeted_20260421_followup_b\mechanism_grid_fat_targeted\merged_20260421_135826`
- Scope:
  - `indoor_factory, 1000`
  - `outdoor_suburban, 1000`
  - `outdoor_urban, 1000`
  - `400` replicates per cell
- Purpose:
  - confirm the harsh 1000-node bottleneck pattern in a separate rerun
- Outcome:
  - the targeted follow-up reproduced the active publication matrix exactly for the checked means:
    - `pdr_expected_mean`
    - `first_node_death_round_mean`
    - `half_nodes_death_round_mean`
    - `gateway_uplink_pdr_total_mean`
    - `gateway_uplink_attempts_total_mean`
    - `cas_TWO_HOP_mean`
  - observed delta versus `results/lcn26_targeted_20260420/mechanism_grid_fat/mechanism_summary.csv`:
    - all checked deltas = `0.0` at stored floating-point precision

## 4. Local machine load policy

- User requirement:
  - local utilization should stay below `90%`
- Measured after offloading:
  - local CPU around `10%` to `16%`
  - local memory around `24%` to `30%`
- Effective policy used:
  - local machine no longer carries the main mechanism run
  - FatMachine carries the full publication-grade mechanism matrix

## 5. Key numerical deltas after valid reruns

### NS-3 PEGASIS office trend

- Old suspicious focused rerun:
  - `[0.999987, 0.999887, 0.9998]`
  - range `0.000187`
- Valid corrected focused rerun:
  - `[0.999953, 0.99969, 0.99869]`
  - range `0.001263`

### Mechanism summary anchors

- `indoor_factory, 100`: `PDR 0.928`, `FND 3.6`
- `indoor_office, 100`: `PDR 0.971`, `FND 13.7`
- `outdoor_urban, 1000`: `PDR 0.138`, `GW uplink PDR 0.286`
- follow-up confirmation on harsh `1000` cells:
  - `indoor_factory`: `PDR 0.728`, `HND 4.5`, `GW attempts 136.9`
  - `outdoor_suburban`: `PDR 0.725`, `HND 2.6`, `GW attempts 171.4`
  - `outdoor_urban`: `PDR 0.138`, `HND 3.3`, `GW attempts 63.3`

## 6. Next manuscript update targets

1. Replace the canonical NS-3 figure/table inputs with `lcn26_ns3_audit_20260420_012811`.
2. Cite the old `010122` rerun only as a discarded audit attempt, if at all.
3. Build the mechanism figure/table from `mechanism_grid_fat/mechanism_summary.csv`.
4. Tighten the text so PEGASIS in `indoor_office` is described as strong but no longer presented as an implausibly scale-invariant flat line.

## 7. Final rebuild path

- Use:
  - `scripts/rebuild_lcn26_final_assets.ps1`
- This wrapper rebuilds:
  1. the base figure set
  2. the corrected canonical NS-3 figure
  3. the corrected tradeoff/mechanism figure
  4. the IEEE PDF in `_LCN26_AERIS`

Key local outputs after the latest successful rebuild:
- `_LCN26_AERIS/generated/fig_lcn26_ns3_canonical.pdf`
- `_LCN26_AERIS/generated/fig_lcn26_tradeoff_cv.pdf`
- `_LCN26_AERIS/aeris_lcn2026.pdf`

## 8. Python environment note

- Use `C:\Users\admin\anaconda3\python.exe` for figure generation.
- Do not use `C:\Users\admin\anaconda3\envs\aether-wsn\python.exe` for matplotlib save operations in this workflow.
- Reason:
  - that environment can import `matplotlib`, but in this session it crashed at `savefig` time with process exit code `-1066598273`.

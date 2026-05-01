# 2026-05-01 LCN26 NS-3 Ablation Evidence

Purpose:
- replace the weaker frozen 100-node ablation figure with a full four-environment, seven-scale NS-3 attribution test
- identify which AERIS module actually carries the measured PDR gain

Source outputs:
- `ns3_validation/results/lcn26_ns3_ablation_local_office_factory_20260501_010355/`
- `ns3_validation/results/lcn26_ns3_ablation_fat_suburban_urban_20260501_011001/`
- `ns3_validation/results/lcn26_ns3_ablation_combined_20260501_010355_011001/summary/`

Run scope:
- `3360` NS-3 experiments
- protocols: `AERIS-FULL`, `AERIS-noGW`, `AERIS-noCAS`, `AERIS-noFair`
- environments: `indoor_office`, `indoor_factory`, `outdoor_suburban`, `outdoor_urban`
- node counts: `50, 100, 200, 300, 500, 800, 1000`
- seeds: `30` per environment-node-variant cell
- local machine ran office/factory; FatMachine ran suburban/urban

Analysis artifacts:
- `ns3_ablation_delta.csv`
- `ns3_ablation_environment_summary.csv`
- `ns3_ablation_summary.md`
- `_LCN26_AERIS/generated/fig_lcn26_ns3_ablation_expanded.pdf`

Key results:
- Gateway removal is the main negative ablation:
  - `indoor_factory`: `-5.82` percentage-point mean delta, significant in `7/7` scales
  - `outdoor_suburban`: `-7.58` percentage-point mean delta, significant in `7/7` scales
  - `indoor_office`: `0.00` mean delta, significant in `0/7` scales
  - `outdoor_urban`: `-0.41` mean delta, significant in `0/7` scales
- CAS is environment dependent:
  - removing CAS helps in office/suburban in this configuration
  - removing CAS hurts urban by `0.91` percentage points on average, significant in `5/7` scales
- CH-score removal is near-neutral:
  - significant cells are at most `1/7` in any environment

Writing implication:
- claim Gateway-assisted uplinks as the strongest measured module
- describe CAS as conditional rather than uniformly beneficial
- do not attribute the measured PDR gain mainly to CH scoring or Skeleton

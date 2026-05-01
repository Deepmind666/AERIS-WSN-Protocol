# 2026-04-20 LCN26 Claim Audit

Draft:
- `_LCN26_AERIS/aeris_lcn2026.tex`

## Canonical NS-3 numbers used in text

Source:
- `ns3_validation/results/lcn26_ns3_audit_20260420_012811/summary/ns3_focused_descriptive.csv`

Office:
- `AERIS`: `0.919960`, `0.912763`, `0.912013`
- `PEGASIS`: `0.999953`, `0.999690`, `0.998690`

1000-node harsh cells:
- `indoor_factory`: `AERIS 0.593067`, `PEGASIS 0.324430`
- `outdoor_suburban`: `AERIS 0.769790`, `PEGASIS 0.548553`
- `outdoor_urban`: `AERIS 0.200407`, `PEGASIS 0.019123`

## Strict-physics numbers used in text

Source:
- `results/mega_experiments/scalability_4env_v50rigor_20260222_descriptive.csv`

Office:
- `PEGASIS`: `0.990700` at `100`
- `PEGASIS`: `0.988406` at `1000`

1000-node harsh cells:
- `indoor_factory`: `AERIS 0.728424`, `PEGASIS 0.299902`
- `outdoor_suburban`: `AERIS 0.727533`, `PEGASIS 0.472524`
- `outdoor_urban`: `AERIS 0.135935`, `PEGASIS 0.098670`

## Mechanism numbers used in text

Source:
- `results/lcn26_targeted_20260420/mechanism_grid_fat/mechanism_summary.csv`
- targeted confirmation:
  - `results/lcn26_targeted_20260421_followup_b/mechanism_grid_fat_targeted/merged_20260421_135826/mechanism_summary.csv`

Anchors:
- `indoor_office, 100`
  - `PDR 0.9714425`
  - `FND 13.72`
  - `GW uplink PDR 0.9998888`
- `indoor_factory, 100`
  - `PDR 0.9282383`
  - `FND 3.65`
  - `GW uplink PDR 0.9988512`
- `outdoor_urban, 1000`
  - `PDR 0.1375278`
  - `FND 1.0`
  - `GW uplink PDR 0.2855185`
- all `12` cells:
  - `skeleton_assignments_mean = 0.0`

Interpretation used in the draft:
- Gateway is the active gain carrier in the current audited publication configuration.
- CAS is environment dependent.
- Skeleton is inactive in the current audited publication configuration.
- the harsh `1000`-node mechanism means were reproduced in a separate targeted FatMachine rerun.

## Frozen 100-node pooled cross-protocol values

Sources:
- `results/mega_experiments/energy_lifetime_stats.csv`
- `results/mega_experiments/latency_hop_v3_20260211_stats.csv`

Used values:
- `AERIS`: `PDR 0.674`, `Energy 131.1`, `Lifetime 215.5`, `FND 15.8`, `Hops 1.98`
- `PEGASIS`: `PDR 0.373`, `Energy 137.6`, `Lifetime 300.0`, `FND 70.7`, `Hops 32.37`
- `LEACH`: `PDR 0.260`, `Energy 165.4`, `Lifetime 300.0`, `FND 0.0`, `Hops 1.71`
- `HEED`: `PDR 0.414`, `Energy 165.2`, `Lifetime 300.0`, `FND 11.4`, `Hops 2.00`
- `TEEN`: `PDR 0.432`, `Energy 159.4`, `Lifetime 299.0`, `FND 0.0`, `Hops 1.28`

## Current audit judgment

- The draft has been rewritten to promote the 2026-04-30 seven-protocol NS-3 sweep as a boundary-mapping figure.
- The draft now also uses the 2026-05-01 expanded NS-3 AERIS ablation as the main attribution figure.
- The old five-protocol numbers remain valid as the classical-baseline repair anchor.
- The main claim is no longer that AERIS is the global harsh-environment winner.
- Active rewritten claim:
  - AERIS strongly beats classical WSN baselines in many harsh cells.
  - AERIS is strongest mainly in outdoor suburban cells after CTP/RPL-MRHOF are added.
  - CTP leads office; RPL-MRHOF leads most factory and urban cells.
  - AERIS's measured gain is mostly carried by Gateway-assisted uplinks, while CAS is environment dependent.

## Expanded seven-protocol numbers now used in text

Source:
- `ns3_validation/results/lcn26_ns3_dual_combined_20260430_191527_191528/summary/ns3_focused_descriptive.csv`
- `ns3_validation/results/lcn26_ns3_dual_combined_20260430_191527_191528/summary/ns3_focused_significance.csv`

Used values:
- Classical-only AERIS rank:
  - rank-1 in `21/28`
  - top-2 in `28/28`
- All-seven AERIS rank:
  - rank-1 in `7/28`
  - top-2 in `8/28`
- Outdoor suburban:
  - AERIS wins `50,100,200,300,500,800`
  - RPL-MRHOF leads at `1000` by about `0.0001` PDR
- Environment-level mean gaps to winner:
  - office: about `-8.16` percentage points
  - factory: about `-1.27` percentage points
  - suburban: about `0.00` percentage points
  - urban: about `-5.69` percentage points

## Expanded NS-3 ablation numbers now used in text

Source:
- `ns3_validation/results/lcn26_ns3_ablation_combined_20260501_010355_011001/summary/ns3_ablation_delta.csv`
- `ns3_validation/results/lcn26_ns3_ablation_combined_20260501_010355_011001/summary/ns3_ablation_environment_summary.csv`
- `ns3_validation/results/lcn26_ns3_ablation_combined_20260501_010355_011001/summary/ns3_ablation_summary.md`

Run scope:
- `3360` NS-3 experiments
- full AERIS plus `AERIS-noGW`, `AERIS-noCAS`, and `AERIS-noFair`
- four environments, seven node counts, `n=30` per environment-node-variant cell
- local shard: `indoor_office`, `indoor_factory`
- FatMachine shard: `outdoor_suburban`, `outdoor_urban`

Used values:
- Gateway removal:
  - `indoor_factory`: `-5.82` percentage-point mean delta, significant in `7/7` scales
  - `outdoor_suburban`: `-7.58` percentage-point mean delta, significant in `7/7` scales
  - `indoor_office`: `0.00` mean delta, significant in `0/7` scales
  - `outdoor_urban`: `-0.41` mean delta, significant in `0/7` scales
- CAS removal:
  - `indoor_office`: `+1.39` mean delta, significant in `7/7` scales
  - `outdoor_suburban`: `+0.76` mean delta, significant in `6/7` scales
  - `outdoor_urban`: `-0.91` mean delta, significant in `5/7` scales
- CH-score removal:
  - near-neutral in all four environments; significant cells are at most `1/7`

Interpretation used in the draft:
- negative delta means the removed module hurts PDR relative to full AERIS.
- Gateway is the main NS-3 gain carrier in factory and suburban regimes.
- CAS is not universally helpful; it helps mainly where harsh links still leave viable short detours.
- CH-score removal is near-neutral in this audited configuration, so the paper should not attribute the observed PDR gain mainly to head scoring.

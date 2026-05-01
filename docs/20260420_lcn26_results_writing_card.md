# 2026-04-20 LCN26 Results Writing Card

Use this file as the short-form writing anchor for the current conference draft.

## Canonical NS-3

Source:
- `ns3_validation/results/lcn26_ns3_audit_20260420_012811/summary/ns3_focused_descriptive.csv`
- `ns3_validation/results/lcn26_ns3_audit_20260420_012811/summary/ns3_focused_significance.csv`

Main environment-scoped claim:
- `PEGASIS` wins in `indoor_office`
- `AERIS` wins in `indoor_factory`, `outdoor_suburban`, and `outdoor_urban`

Key corrected office numbers:
- `PEGASIS`: `0.999953 -> 0.999690 -> 0.998690` for `100, 500, 1000`
- `AERIS`: `0.919960 -> 0.912763 -> 0.912013`
- Do not describe PEGASIS office as an implausibly flat line anymore.

Key 1000-node harsh numbers:
- `indoor_factory`: `AERIS 0.593` vs `PEGASIS 0.324`
- `outdoor_suburban`: `AERIS 0.770` vs `PEGASIS 0.549`
- `outdoor_urban`: `AERIS 0.200` vs `PEGASIS 0.019`

Interpretation:
- Canonical baseline ranking now supports harsh-channel AERIS claims without depending only on the custom Python simulator.

## Expanded Seven-Protocol NS-3 Boundary

Source:
- `ns3_validation/results/lcn26_ns3_dual_combined_20260430_191527_191528/summary/ns3_focused_descriptive.csv`
- `ns3_validation/results/lcn26_ns3_dual_combined_20260430_191527_191528/summary/ns3_focused_significance.csv`

Role in the rewritten paper:
- main boundary figure
- stronger scientific framing than the five-protocol-only story
- not an `AERIS is globally best` result

Key result:
- Against classical WSN baselines only:
  - AERIS rank-1 in `21/28` cells
  - AERIS top-2 in `28/28` cells
- Against all seven protocols:
  - AERIS rank-1 in `7/28` cells
  - AERIS top-2 in `8/28` cells

Environment-level boundary:
- `indoor_office`: CTP/RPL-MRHOF dominate; AERIS mean gap to winner is about `-8.16` percentage points.
- `indoor_factory`: AERIS wins only at `50` nodes; RPL-MRHOF leads the larger scales; AERIS mean gap to winner is about `-1.27` points.
- `outdoor_suburban`: AERIS wins `50,100,200,300,500,800`; RPL-MRHOF leads by only about `0.0001` PDR at `1000`; this is AERIS's strongest expanded-baseline regime.
- `outdoor_urban`: RPL-MRHOF leads all scales; AERIS mean gap to winner is about `-5.69` points.

Interpretation:
- AERIS remains a strong rule-based alternative to classical WSN baselines.
- CTP/RPL-MRHOF reveal the deployment boundary: richer collection-tree parent selection narrows or removes the AERIS advantage in office, factory, and urban regimes.
- The paper should be framed as a boundary-mapping engineering paper, not a universal winner claim.

## Strict-Physics Python Matrix

Source:
- `results/mega_experiments/scalability_4env_v50rigor_20260222_descriptive.csv`

Role in the paper:
- stress-test layer only
- not the sole fairness anchor
- confirms the harsh-channel AERIS advantage under collision and relay stress

## Expanded NS-3 AERIS Ablation

Source:
- `ns3_validation/results/lcn26_ns3_ablation_combined_20260501_010355_011001/summary/ns3_ablation_delta.csv`
- `ns3_validation/results/lcn26_ns3_ablation_combined_20260501_010355_011001/summary/ns3_ablation_environment_summary.csv`
- `ns3_validation/results/lcn26_ns3_ablation_combined_20260501_010355_011001/summary/ns3_ablation_summary.md`

Role in the paper:
- main module-attribution figure
- replaces the weaker frozen 100-node ablation figure
- connects the boundary result to concrete AERIS internals

Run scope:
- `3360` NS-3 experiments
- full AERIS plus `AERIS-noGW`, `AERIS-noCAS`, and `AERIS-noFair`
- four environments, seven node counts, `n=30` per cell
- local shard covers office/factory; FatMachine shard covers suburban/urban

Key result:
- Gateway removal costs:
  - `5.82` percentage points on average in `indoor_factory`, significant in `7/7` scales
  - `7.58` percentage points on average in `outdoor_suburban`, significant in `7/7` scales
  - near-neutral in `indoor_office`
  - weak and not Holm-significant in `outdoor_urban`
- CAS removal:
  - improves office/suburban PDR in this audited configuration
  - hurts `outdoor_urban` by `0.91` points on average, significant in `5/7` scales
- CH-score removal:
  - near-neutral in every environment

Interpretation:
- Gateway-assisted uplinks are the strongest measured AERIS module.
- CAS is conditional rather than universally beneficial.
- The current draft should not claim that CH scoring or Skeleton is the primary measured source of the PDR gain.

## Mechanism Matrix

Source:
- `results/lcn26_targeted_20260420/mechanism_grid_fat/mechanism_summary.csv`
- `results/lcn26_targeted_20260420/mechanism_grid_fat/mechanism_raw_merged.json`
- targeted confirmation:
  - `results/lcn26_targeted_20260421_followup_b/mechanism_grid_fat_targeted/merged_20260421_135826/mechanism_summary.csv`

Main mechanism claim:
- `Gateway` is the dominant reliability mechanism in the current publication configuration
- `CAS` is environment dependent
- `Skeleton` is inactive in this configuration

Numerical anchors:
- gateway uplink PDR stays `0.996 ~ 1.000` in office, factory, and suburban cells
- gateway uplink PDR drops to `0.286` at `outdoor_urban, 1000`
- end-to-end PDR at `outdoor_urban, 1000` is `0.138`
- first-node death is:
  - `13.7` at `indoor_office, 100`
  - `3.6` at `indoor_factory, 100`
  - `4.3` at `outdoor_suburban, 100`
  - `2.0` at `outdoor_urban, 100`
  - mostly `1.0 ~ 4.5` in the `500/1000` harsh cells
- `skeleton_assignments = 0` in all `12` audited cells

Interpretation:
- AERIS reliability comes from strong gateway-aided uplinks and aggressive two-hop use in selected cells
- the cost is concentrated relay burden and very early node attrition
- the harsh `1000`-node bottleneck pattern was reproduced in a separate targeted FatMachine rerun, so the urban/factory/suburban `1000` mechanism means are no longer resting on only one merged batch.

## Frozen 100-Node Publication Block

Source:
- `results/mega_experiments/energy_lifetime_stats.csv`
- `results/mega_experiments/latency_hop_v3_20260211_stats.csv`

Cross-protocol pooled numbers used in the tradeoff table:
- `AERIS`: `PDR 0.674`, `Energy 131.1`, `Lifetime 215.5`, `FND 15.8`, `Hops 1.98`
- `PEGASIS`: `PDR 0.373`, `Energy 137.6`, `Lifetime 300.0`, `FND 70.7`, `Hops 32.37`

Interpretation:
- AERIS is energy-efficient but does not maximize longevity
- PEGASIS is longevity-first but incurs extreme path length

## Writing Constraints

- Use `012811` as the only valid corrected NS-3 rerun.
- Do not use `010122` for claims; it was invalid because the WSL scratch source was stale.
- Do not overstate `Skeleton`; the corrected mechanism matrix does not support that.
- Use the 2026-05-01 NS-3 ablation for module attribution, not the older frozen 100-node ablation figure.
- Keep the claim scoped:
  - harsh-channel reliability-first -> `AERIS`
  - benign office-like or longevity-first -> `PEGASIS`

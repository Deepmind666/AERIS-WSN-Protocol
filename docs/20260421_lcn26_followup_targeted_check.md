# 2026-04-21 LCN26 Targeted Mechanism Follow-Up Check

Purpose:
- confirm the harsh 1000-node mechanism pattern with a separate targeted rerun on FatMachine

Remote source:
- `C:\Users\sshuser\AERIS-WSN-Protocol\results\lcn26_targeted_20260421_followup_b\mechanism_grid_fat_targeted\merged_20260421_135826\`

Local copy:
- `results/lcn26_targeted_20260421_followup_b/mechanism_grid_fat_targeted/merged_20260421_135826/`

Cells rerun:
- `indoor_factory, 1000`
- `outdoor_suburban, 1000`
- `outdoor_urban, 1000`

Replicates:
- `400` per cell

Checked metrics:
- `pdr_expected_mean`
- `first_node_death_round_mean`
- `half_nodes_death_round_mean`
- `gateway_uplink_pdr_total_mean`
- `gateway_uplink_attempts_total_mean`
- `cas_TWO_HOP_mean`

Result:
- the targeted rerun reproduced the active publication matrix exactly for all checked means in all three cells
- observed delta versus `results/lcn26_targeted_20260420/mechanism_grid_fat/mechanism_summary.csv`:
  - all checked deltas = `0.0` at the stored floating-point precision

Per-cell reproduced means:
- `indoor_factory, 1000`: `PDR 0.7280548040`, `FND 1.0`, `HND 4.5275`, `GW uplink PDR 0.9693294861`, `GW attempts 136.8725`, `CAS two-hop 436.325`
- `outdoor_suburban, 1000`: `PDR 0.7249323733`, `FND 1.0`, `HND 2.5775`, `GW uplink PDR 0.9897376183`, `GW attempts 171.4175`, `CAS two-hop 375.6875`
- `outdoor_urban, 1000`: `PDR 0.1375277960`, `FND 1.0`, `HND 3.26`, `GW uplink PDR 0.2855184772`, `GW attempts 63.2925`, `CAS two-hop 10.4825`

Interpretation:
- the harsh-cell bottleneck pattern is stable under a separate rerun
- this follow-up is a confirmatory check only
- it does not replace the full 12-cell audited mechanism matrix

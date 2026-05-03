# LCN26 NS-3 AERIS Ablation Summary

- input: `ns3_validation\results\lcn26_ns3_ablation_combined_20260501_010355_011001\summary\ns3_focused_merged.json`
- experiments: `3360`
- reference: `AERIS-FULL`

Delta is variant minus full AERIS in percentage points; negative values mean the removed module hurts delivery.

| Environment | Variant | Mean delta pts | Range pts | Significant cells |
|---|---|---:|---:|---:|
| indoor_factory | AERIS-noGW | -5.82 | -6.55 to -5.23 | 7/7 |
| indoor_factory | AERIS-noCAS | 0.07 | -0.32 to 0.47 | 0/7 |
| indoor_factory | AERIS-noFair | 0.04 | -0.35 to 0.52 | 0/7 |
| indoor_office | AERIS-noGW | 0.00 | -0.09 to 0.13 | 0/7 |
| indoor_office | AERIS-noCAS | 1.39 | 0.94 to 1.89 | 7/7 |
| indoor_office | AERIS-noFair | 0.01 | -0.13 to 0.14 | 0/7 |
| outdoor_suburban | AERIS-noGW | -7.58 | -7.97 to -7.18 | 7/7 |
| outdoor_suburban | AERIS-noCAS | 0.76 | 0.45 to 1.21 | 6/7 |
| outdoor_suburban | AERIS-noFair | -0.01 | -0.24 to 0.38 | 1/7 |
| outdoor_urban | AERIS-noGW | -0.41 | -0.91 to 0.02 | 0/7 |
| outdoor_urban | AERIS-noCAS | -0.91 | -1.55 to -0.72 | 5/7 |
| outdoor_urban | AERIS-noFair | -0.12 | -0.85 to 0.59 | 0/7 |

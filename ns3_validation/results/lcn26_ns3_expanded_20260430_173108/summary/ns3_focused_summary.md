# LCN26 Focused NS-3 Audit Summary

- input_dir: `/mnt/c/Users/sshuser/AERIS-WSN-Protocol/ns3_validation/results/lcn26_ns3_expanded_20260430_173108/raw`
- merged experiments: `2520`
- shards: `28`

## Winner by environment-node cell

| Environment | Nodes | Winner | PDR |
|---|---:|---|---:|
| indoor_factory | 100 | RPL-MRHOF | 0.605 |
| indoor_factory | 500 | RPL-MRHOF | 0.608 |
| indoor_factory | 1000 | RPL-MRHOF | 0.611 |
| indoor_office | 100 | CTP | 1.000 |
| indoor_office | 500 | CTP | 1.000 |
| indoor_office | 1000 | CTP | 1.000 |
| outdoor_suburban | 100 | AERIS | 0.777 |
| outdoor_suburban | 500 | AERIS | 0.770 |
| outdoor_suburban | 1000 | RPL-MRHOF | 0.771 |
| outdoor_urban | 100 | RPL-MRHOF | 0.253 |
| outdoor_urban | 500 | RPL-MRHOF | 0.258 |
| outdoor_urban | 1000 | RPL-MRHOF | 0.261 |

## PEGASIS office trend check

- indoor_office PEGASIS PDR values: `[0.999953, 0.99969, 0.99869]`
- range across tested scales: `0.001263`

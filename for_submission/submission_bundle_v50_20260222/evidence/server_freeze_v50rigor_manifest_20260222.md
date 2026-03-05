# Server Freeze Manifest (v50-rigor)

Date: 2026-02-22
Branch: v50-rigor
Commit: 0dddcf4

## Authoritative Result Files (4 environments)

| # | Environment | File | Owner | SHA256 | Provenance Sidecar |
|---|-------------|------|-------|--------|-------------------|
| 1 | indoor_office | scalability_indoor_office_v50rigor_20260222_032955.json | Codex (local) | c1947750...edad4f | VERIFIED |
| 2 | outdoor_suburban | scalability_outdoor_suburban_v50rigor_20260222_103921.json | Codex (local) | 8aca47ab...530a53 | VERIFIED |
| 3 | outdoor_urban | scalability_outdoor_urban_v50rigor_20260222_server.json | Claude (server) | 8f55af34...753c21 | VERIFIED |
| 4 | indoor_factory | scalability_indoor_factory_v50rigor_20260222_server.json | Claude (server) | 3ec187a3...77f39d | VERIFIED |

## Full SHA256 Hashes

```
c1947750d0e445bb3433400ef5367e46bf8e7ba0ccf478440d173b4879edad4f  scalability_indoor_office_v50rigor_20260222_032955.json
8aca47aba9941c75c362d67c008053d8e5b84268736d0c3f8c64cd72e0530a53  scalability_outdoor_suburban_v50rigor_20260222_103921.json
8f55af3451bd9a75f2eeb01dcff8e5359027973bf20f54cb698c78ea7f753c21  scalability_outdoor_urban_v50rigor_20260222_server.json
3ec187a35b724ef52bbe2d9eb2d86e14dd23b568cfea20ba010cf001a577f39d  scalability_indoor_factory_v50rigor_20260222_server.json
```

## Provenance Sidecar Status

| File | Status |
|------|--------|
| scalability_outdoor_urban_v50rigor_20260222_server.provenance.json | EXISTS, verified |
| scalability_indoor_factory_v50rigor_20260222_server.provenance.json | EXISTS, verified |
| scalability_indoor_office_v50rigor_20260222_032955.provenance.json | EXISTS, verified |
| scalability_outdoor_suburban_v50rigor_20260222_103921.provenance.json | EXISTS, verified |

## Common Parameters (all 4 files)

- replicates: 3200
- seed: 42001
- nodes: 100, 200, 300, 500, 800, 1000
- rounds: 300
- run_tier: publication
- primary_metric: pdr_expected
- flags: --mac-collision --multihop-relay
- tx_power: 10.0

## Lock Conditions

All 4 files must have:
- raw_results == 96000
- error_runs == 0
- provenance sidecar with matching data_sha256

Current: 4/4 provenance verified. Freeze package is complete for final manuscript lock.

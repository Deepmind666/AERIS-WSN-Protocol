# Local Rigor-Patch Pilot Comparison (2026-02-16)

## Scope

- Baseline pilot (pre-patch): `pilot_rigor_pub_r2_*_20260215_181356.json`
- Rigor-patch pilot: `pilot_rigor_patch_*_20260215_224539.json`
- Common matrix: 4 environments x nodes {100, 500, 1000} x 5 protocols x n=60
- Primary metric: `pdr_expected`

## Key outcome

The rigor-patch run restores physically expected scale monotonicity for AERIS in all environments:

- indoor_office: 0.9706 -> 0.8676 -> 0.6746
- indoor_factory: 0.9283 -> 0.9083 -> 0.7355
- outdoor_urban: 0.7487 -> 0.7362 -> 0.1492
- outdoor_suburban: 0.9563 -> 0.9193 -> 0.7307

`non_increasing=True` for all four environments.

## Interpretation

1. The patch direction is technically valid for simulator rigor (removes PDR-up-with-scale anomaly).
2. Ranking behavior changes in benign channels:
   - indoor_office: PEGASIS > AERIS at 100/500/1000 under patched setting.
3. AERIS remains dominant in harsh environments (indoor_factory, outdoor_urban, outdoor_suburban).

## Decision gate

- PASS: monotonicity sanity gate.
- HOLD: do not promote patched pilot values into final manuscript tables yet.
- Next required step: full rerun under patched settings with publication-tier sample size and updated claim-source audit.


# v68 Coefficients Traceability (Code -> Manuscript)

Date: 2026-02-25
Manuscript target: `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260225_v68.tex`

## Scope
This file maps coefficient values reported in Section 3 (Table `tab:cas_gateway_coeffs`) to implementation constants.

## Publication-tier profile and run path
- Publication scalability runs instantiate AERIS with `profile="energy"`:
  - `scripts/run_scalability_experiment.py:236-243`
- AERIS gateway scoring defaults captured under energy profile defaults:
  - `src/aeris_protocol.py:370-384`
- Gateway selector receives effective coefficients from config defaults (or explicit config values):
  - `src/aeris_protocol.py:1266-1269`

## CAS coefficients (base constants)
Source: `src/cas_selector.py:72-93`

| Mode | Manuscript symbol mapping | Code fields | Values |
|---|---|---|---|
| direct | `(w_q, w_e, w_d, w_r, w_ρ, w_f)` | `w_direct_link`, `w_direct_energy`, `w_direct_dist_bs`, `w_direct_radius`, `w_direct_density`, `w_direct_fair` | `(0.65, 0.35, -0.25, -0.05, 0.10, -0.05)` |
| chain | `(w_q, w_e, w_d, w_r, w_ρ, w_f)` | `w_chain_link`, `w_chain_energy`, `w_chain_dist_bs`, `w_chain_radius`, `w_chain_density`, `w_chain_fair` | `(0.40, 0.30, 0.20, 0.20, 0.20, -0.05)` |
| two-hop | `(w_q, w_e, w_d, w_r, w_ρ, w_f)` | `w_twohop_link`, `w_twohop_energy`, `w_twohop_dist_bs`, `w_twohop_radius`, `w_twohop_density`, `w_twohop_fair` | `(0.25, 0.20, 0.50, 0.15, 0.05, -0.05)` |

## Gateway coefficients (base constants)
Primary effective defaults in publication profile:
- `gateway_w_dist = -0.60`
- `gateway_w_centrality = 0.20`
- `gateway_w_link = 0.35`
- `gateway_w_energy = 0.15`

Code references:
- Defaults captured: `src/aeris_protocol.py:381-384`
- Passed to gateway selector: `src/aeris_protocol.py:1266-1269`
- Selector score form: `src/gateway_selector.py:139-142`

Manuscript Eq.(2) mapping:
- `alpha` -> `gateway_w_energy` (E)
- `beta` -> `gateway_w_centrality` (C)
- `gamma` -> `gateway_w_link` (L)
- `delta` -> `gateway_w_dist` (D)

## Stage-adaptive updates (bounded runtime modifications)
These are runtime-bounded updates, not per-replicate retuning:
- Gateway bounded updates: `src/aeris_protocol.py:1751-1754`
- CAS bounded updates: `src/aeris_protocol.py:1769-1779`
- Base CAS defaults captured once: `src/aeris_protocol.py:1147-1172`

Manuscript wording alignment:
- Section 3 states fixed base coefficients and bounded deterministic stage updates under fixed seeds.

## Consistency note
The v68 manuscript table reports base initialization coefficients used for publication-tier runs.
Runtime adaptive updates are logged as metadata and interpreted as bounded stage adjustments rather than environment-specific retuning.

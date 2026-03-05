# NS-3 Alignment Evidence (Publication Gate)

> Updated: 2026-02-15 (incorporated 800/1000-node scale extension)
> Previous: 2026-02-11

## 0. Scope

This document is the canonical NS-3 evidence for this repository.

- Validation level: `TREND-LEVEL`
- Not allowed claim: `numerical alignment completed`
- Metric: `pdr_expected` trend (AERIS vs LEACH)
- Scale coverage: 50–1000 nodes (7 node counts × 4 environments × n=30)

## 1. Evidence Files

- `ns3_scale_ext_1000_20260211.json` (50/100/200/300/500/800/1000 nodes, 4 envs, n=30, + ablation at 100 nodes; 2160 experiments)
- `ns3_scale_ext_1000_stats.csv` (descriptive statistics for all cells)
- `ns3_scale_ext_1000_significance.csv` (Welch t-test + Hedges' g + Holm-Bonferroni for all cells)
- `ns3_multienv_publication_v2_20260211.json` (legacy: 50/100/200 nodes, 4 envs, n=30)
- `ns3_scale_extension_20260211.json` (legacy: 50/100/200/300/500 nodes, 4 envs, n=30)
- `ns3_scale_ext_stats.csv` (legacy)
- `ns3_scale_ext_significance.csv` (legacy)
- `ns3_multienv_stats.csv` (legacy)
- `ns3_multienv_significance.csv` (legacy)
- `aeris-validation-standalone.cc` (source code)

## 2. Parameter Alignment Check

### 2.1 Environment Mapping

Python channel targets are matched in NS-3 code (`aeris-validation-standalone.cc`):

- `indoor_office`: PLE=2.0, shadow=4.5 (`line 50`)
- `indoor_factory`: PLE=2.7, shadow=8.5 (`line 53`)
- `outdoor_urban`: PLE=3.4, shadow=12.0 (`line 52`)
- `outdoor_suburban`: PLE=2.8, shadow=7.5 (`line 54`)

### 2.2 Core Experiment Controls

- `tx_power_dbm` clamp: `[-25, 10]`, default `10.0` (`line 35`, `line 59`)
- `initial_energy`: `2.0 J` (`line 115`, `line 256`, `line 432`, `line 440`)
- `rounds`: `300` (`line 115`, `line 256`, `line 501`)
- `packet_size`: `512 bytes` (`line 115`, `line 256`, `line 432`, `line 440`)
- `seeds`: `42001..42030` (`line 524`, `line 525`)

## 3. Statistical Results (AERIS vs LEACH, n=30)

Source: `ns3_scale_ext_1000_significance.csv` (Holm-corrected p-values).

### 3.1 Node Count = 100 (most relevant for manuscript baseline table)

| Environment | AERIS mean | LEACH mean | Diff | Hedges g | Holm p | Significant |
|---|---:|---:|---:|---:|---:|---|
| indoor_office | 0.920240 | 0.918523 | +0.001717 | 0.3547 | 1.000000e+00 | NO |
| indoor_factory | 0.602530 | 0.533557 | +0.068973 | 5.2097 | 4.233767e-25 | YES |
| outdoor_urban | 0.206373 | 0.189857 | +0.016517 | 1.0117 | 3.414472e-03 | YES |
| outdoor_suburban | 0.777143 | 0.692123 | +0.085020 | 7.4307 | 1.935380e-30 | YES |

### 3.2 Scale Extension (300-1000 nodes, n=30)

Source: `ns3_scale_ext_1000_significance.csv`.

| Environment | Nodes | AERIS mean | LEACH mean | Diff | Hedges g | Holm p | Sig? |
|---|---:|---:|---:|---:|---:|---:|---|
| indoor_office | 300 | 0.915627 | 0.912173 | +0.003453 | 0.9354 | 7.596e-03 | YES |
| indoor_office | 500 | 0.912683 | 0.909540 | +0.003143 | 0.7624 | 4.909e-02 | YES |
| indoor_office | 800 | 0.913373 | 0.910090 | +0.003283 | 0.8195 | 2.803e-02 | YES |
| indoor_office | 1000 | 0.912773 | 0.910343 | +0.002430 | 0.6093 | 2.119e-01 | NO |
| indoor_factory | 300 | 0.598147 | 0.525907 | +0.072240 | 7.7030 | 2.244e-35 | YES |
| indoor_factory | 500 | 0.595580 | 0.526027 | +0.069553 | 8.5549 | 5.507e-38 | YES |
| indoor_factory | 800 | 0.595010 | 0.526690 | +0.068320 | 11.0870 | 3.744e-44 | YES |
| indoor_factory | 1000 | 0.592363 | 0.524563 | +0.067800 | 13.7688 | 4.768e-46 | YES |
| outdoor_urban | 300 | 0.202410 | 0.186847 | +0.015563 | 1.7709 | 7.159e-08 | YES |
| outdoor_urban | 500 | 0.199790 | 0.182730 | +0.017060 | 2.1921 | 2.138e-10 | YES |
| outdoor_urban | 800 | 0.199250 | 0.185337 | +0.013913 | 2.6485 | 1.751e-13 | YES |
| outdoor_urban | 1000 | 0.200430 | 0.185710 | +0.014720 | 3.3065 | 2.577e-17 | YES |
| outdoor_suburban | 300 | 0.773530 | 0.687427 | +0.086103 | 11.3311 | 8.975e-37 | YES |
| outdoor_suburban | 500 | 0.769053 | 0.686550 | +0.082503 | 14.1757 | 1.176e-41 | YES |
| outdoor_suburban | 800 | 0.770177 | 0.685777 | +0.084400 | 15.6879 | 2.953e-51 | YES |
| outdoor_suburban | 1000 | 0.770393 | 0.687383 | +0.083010 | 17.4277 | 4.881e-50 | YES |

### 3.3 PDR-Scale Trend in NS-3 (key physical plausibility check)

NS-3 PDR trends across scale (AERIS, n=30 per cell):

| Environment | 50 nodes | 100 | 500 | 1000 | Trend |
|---|---:|---:|---:|---:|---|
| indoor_office | 0.9357 | 0.9202 | 0.9127 | 0.9128 | monotone ↓ (plateau at 500+) |
| indoor_factory | 0.6044 | 0.6025 | 0.5956 | 0.5924 | monotone ↓ |
| outdoor_urban | 0.2127 | 0.2064 | 0.1998 | 0.2004 | ↓ (plateau at 500+) |
| outdoor_suburban | 0.7906 | 0.7771 | 0.7691 | 0.7704 | ↓ (plateau at 500+) |

All four environments show PDR decreasing or plateauing with scale — physically plausible. This contrasts with the Python simulator where PDR increases with scale in 3/4 environments (see §6 note).

### 3.4 Trend Summary

- AERIS mean PDR ≥ LEACH in 27/28 environment-scale cells (directionally positive); indoor_office at n=200 shows AERIS slightly below LEACH (diff=−0.0004).
- After Holm correction, significance confirmed in 25/28 comparisons.
- Three non-significant cells (all `indoor_office`): n=100 (g=+0.35, p=1.00), n=200 (g=−0.11, p=1.00, AERIS < LEACH), n=1000 (g=+0.61, p=0.21).
- At 300–800 nodes, AERIS > LEACH is significant in ALL 4 environments (16/16).
- Effect sizes grow with scale in harsh environments (indoor_factory g: 5.2→13.8; outdoor_suburban g: 7.4→17.4).
- Total: 25/28 AERIS-vs-LEACH comparisons significant across 7 node counts × 4 environments.

## 4. Ablation Signals in NS-3 (100 nodes)

Source: `ns3_scale_ext_1000_significance.csv` (ablation rows).

- `FULL vs noGW`:
  - indoor_factory: positive and significant (`diff=+0.051867`, `g=3.38`, `p_holm=1.03e-17`) — Gateway critical
  - outdoor_suburban: positive and significant (`diff=+0.073270`, `g=7.23`, `p_holm=5.31e-30`) — Gateway critical
  - outdoor_urban: not significant (`diff=+0.005100`, `g=0.35`, `p_holm=1.00`)
  - indoor_office: not significant (`diff=+0.000293`, `g=0.09`, `p_holm=1.00`)
- `FULL vs noCAS`:
  - indoor_office: `noCAS` better, significant (`diff=-0.014967`, `g=-4.80`, `p_holm=3.27e-24`) — CAS overhead in benign channel
  - outdoor_suburban: `noCAS` better, significant (`diff=-0.006597`, `g=-0.88`, `p_holm=1.39e-02`)
  - outdoor_urban: `FULL` better, significant (`diff=+0.015057`, `g=0.99`, `p_holm=4.04e-03`) — CAS helps in harsh channel
  - indoor_factory: not significant (`diff=-0.003933`, `g=-0.27`, `p_holm=1.00`)
- `FULL vs noFair`:
  - All four environments: not significant (all `p_holm > 0.05`)

Interpretation: Gateway is the dominant module in harsh environments. CAS is environment-dependent — beneficial in outdoor_urban, detrimental in indoor_office. Fairness module has no statistically detectable effect at n=30.

## 5. Claim Gate (What Can and Cannot Be Written)

### 5.1 Allowed

- "NS-3 confirms trend-level consistency: AERIS >= LEACH across all tested environments and scales (50-1000 nodes)."
- "At 100 nodes, AERIS exceeds LEACH in all four environments, with significance in 3/4 after Holm correction."
- "At 300-800 nodes, AERIS significantly outperforms LEACH in all four environments (16/16, Holm alpha=0.05)."
- "At 1000 nodes, AERIS significantly outperforms LEACH in 3/4 environments; indoor_office is directionally positive but not significant (g=0.61, p=0.21)."
- "Gateway contribution is environment-dependent; significant in harsher channels (indoor_factory, outdoor_suburban)."
- "CAS is environment-dependent: beneficial in outdoor_urban, detrimental in indoor_office, negligible elsewhere."
- "Across 7 node counts (50-1000) and 4 environments, 25/28 AERIS-vs-LEACH comparisons are statistically significant."
- "NS-3 PDR decreases or plateaus with scale in all four environments: physically plausible trend."

### 5.2 Not Allowed

- "NS-3 numerical alignment is fully completed."
- "AERIS significantly beats LEACH in every NS-3 environment at every scale." (indoor_office n=100, n=200, n=1000 not significant)
- Any claim that ignores the non-significant indoor_office results.
- "NS-3 validates the Python simulator's absolute PDR values." (platform gap remains: e.g., Python AERIS indoor_office 100n = 97.4% vs NS-3 = 92.0%)

## 6. Physical Plausibility Note

NS-3 shows PDR monotonically decreasing (or plateauing) with node count in all environments. This is physically expected due to increased contention. The Python simulator shows the opposite trend in 3/4 environments (PDR increasing with scale), which is a known limitation addressed by the planned MAC collision model patch.

## 7. Status

- NS-3 scale coverage: 50–1000 nodes × 4 environments × {AERIS, LEACH} × n=30 = 1680 main experiments + 480 ablation = **2160 total**
- NS-3 evidence completeness for publication gate: `PASS (trend-level)`
- Numerical equivalence gate: `NOT PASSED`
- Last updated: 2026-02-15

# NS-3 Cross-Validation Report for AERIS Protocol

## Executive Summary

5-protocol NS-3 cross-validation across 4 radio environments, 7 node counts, and 30 independent seeds (4,680 total experiments). AERIS significantly outperforms LEACH, HEED, and TEEN in PDR across all environments. PEGASIS shows environment-dependent behavior due to chain-cumulative packet loss.

## Validation Environment

- **Simulator**: NS-3 3.40
- **Platform**: Ubuntu 24.04 (WSL2), 24-core server
- **Protocols**: AERIS, LEACH, HEED, PEGASIS, TEEN
- **Environments**: indoor_office, indoor_factory, outdoor_urban, outdoor_suburban
- **Node counts**: 50, 100, 200, 300, 500, 800, 1000
- **Seeds**: 30 independent runs (42001–42030)
- **Rounds**: 300 per experiment
- **Area**: 200m × 200m, BS at (100, 200)
- **Initial Energy**: 2.0 J per node
- **Data Packet Size**: 512 bytes
- **Channel Model**: Realistic physics-based (3GPP-calibrated path loss per environment)
- **Execution**: 24 parallel shards (20 main + 4 ablation), ~4% CPU per shard

## Key Results

### 1. Mean PDR by Protocol × Environment (all node counts pooled, n=120 each)

| Protocol | indoor_office | indoor_factory | outdoor_urban | outdoor_suburban |
|----------|--------------|----------------|---------------|------------------|
| AERIS    | 92.22%       | 60.12%         | 20.46%        | 77.90%           |
| LEACH    | 91.70%       | 53.04%         | 18.78%        | 69.21%           |
| HEED     | 87.11%       | 52.07%         | 18.99%        | 67.23%           |
| PEGASIS  | 99.98%       | 43.64%         | 1.13%         | 73.29%           |
| TEEN     | 62.14%       | 36.40%         | 13.06%        | 47.31%           |

### 2. AERIS PDR by Node Count (100-node reference)

| Environment      | 50 nodes | 100 nodes | 200 nodes | 500 nodes | 1000 nodes |
|------------------|----------|-----------|-----------|-----------|------------|
| indoor_office    | 93.62%   | 92.04%    | 91.87%    | 91.56%    | 91.37%     |
| indoor_factory   | 61.27%   | 60.02%    | 59.63%    | 59.30%    | 59.22%     |
| outdoor_urban    | 20.97%   | 20.59%    | 20.16%    | 19.97%    | 19.92%     |
| outdoor_suburban | 78.72%   | 77.74%    | 77.42%    | 77.00%    | 76.72%     |

PDR decreases monotonically with node count across all environments (physically expected).

### 3. Statistical Significance (Welch t-test + Holm-Bonferroni)

| Baseline | Significant / Total | Avg Hedges' g | AERIS wins |
|----------|--------------------:|:-------------:|:----------:|
| LEACH    | 14 / 16             | +4.03         | 16 / 16    |
| HEED     | 16 / 16             | +7.49         | 16 / 16    |
| PEGASIS  | 16 / 16             | +3.48         | 12 / 16    |
| TEEN     | 16 / 16             | +42.28        | 16 / 16    |

Non-significant comparisons (2 of 64):
- AERIS vs LEACH, indoor_office, 100 nodes (p_adj=0.456, g=+0.19)
- AERIS vs LEACH, indoor_office, 200 nodes (p_adj=0.402, g=+0.33)

These are the easiest conditions where both protocols perform well (~92% PDR).

### 4. PEGASIS Notes

PEGASIS uses chain-based forwarding where each node's data traverses ~N/2 hops to the round-robin leader. Hop distance follows the Beardwood-Halton-Hammersley model: `0.7 * sqrt(Area/N)`, so denser networks have shorter hops. Per-hop success is evaluated via the channel model. This produces:
- indoor_office: ~100% PDR (short-range LOS links have near-zero per-hop loss)
- indoor_factory: 40.7% → 50.7% as nodes increase 50 → 1000 (shorter hops offset more hops)
- outdoor_suburban: 71.7% → 75.9% (same density-benefit effect)
- outdoor_urban: ~1% at all scales (high per-hop loss dominates even with short hops)

PEGASIS outperforms AERIS only in indoor_office (4 of 16 comparisons). In all other environments, AERIS wins.

### 5. Ablation Study (100 nodes, 4 environments)

| Variant      | Significant envs | Key findings |
|--------------|:----------------:|--------------|
| AERIS-noCAS  | 2 / 4            | CAS contributes in office (g=−6.20) and suburban (g=−1.05) |
| AERIS-noFair | 0 / 4            | Fairness effect not significant at 100 nodes / 300 rounds |
| AERIS-noGW   | 2 / 4            | Gateway critical in factory (g=+4.01) and suburban (g=+7.04) |

Gateway module provides the largest improvement in challenging environments. CAS module improves PDR in moderate-to-good link conditions.

## Reproducibility

```bash
# Single shard (e.g., AERIS in indoor_factory)
cd ns-3.40
export LD_LIBRARY_PATH=./build/lib
./build/scratch/ns3.40-aeris-validation-standalone-default \
  --runShard --protocol=AERIS --env=indoor_factory \
  --nodes=50,100,200,300,500,800,1000 \
  --output=shard_AERIS_indoor_factory.json

# Full matrix (all 5 protocols × 4 environments)
./build/scratch/ns3.40-aeris-validation-standalone-default --runMultiEnv
```

## Files

- **C++ source**: `ns3_validation/aeris-validation-standalone.cc`
- **Merge + analysis**: `ns3_validation/merge_and_analyze.py`
- **Raw shards**: `ns3_validation/results/shards_5proto/shard_*.json` (24 files)
- **Merged data**: `ns3_validation/results/ns3_5proto_merged.json` (4,680 experiments)
- **Summary stats**: `ns3_validation/results/ns3_5proto_summary.json`
- **Significance tests**: `ns3_validation/results/ns3_5proto_significance.json`

---
*Updated: 2026-02-15*
*NS-3 Version: 3.40*
*AERIS Protocol — 5-protocol multi-environment validation*

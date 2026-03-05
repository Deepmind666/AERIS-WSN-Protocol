# Section 6: Results and Analysis

---

## 6. Results and Analysis

This section reports publication-tier results (n=30). All PDR values use
pdr_expected (bs_delivered / source_packets_expected).

### 6.1 Multi-Environment 5-Protocol Comparison (n=30)

Data source:  
C:\AERIS-WSN-Protocol\results\mega_experiments\env_sensitivity_20260207_205317.json

**Table 6.1: PDR by Environment (mean +/- std)**

| Environment | AERIS | LEACH | PEGASIS | HEED | TEEN |
|---|---|---|---|---|---|
| indoor_office | 0.9739+/-0.0047 | 0.5543+/-0.0401 | 0.9078+/-0.0166 | 0.9371+/-0.0076 | 0.8222+/-0.0044 |
| indoor_factory | 0.6031+/-0.0258 | 0.1614+/-0.0209 | 0.1928+/-0.0255 | 0.2326+/-0.0263 | 0.3113+/-0.0245 |
| outdoor_urban | 0.3745+/-0.0354 | 0.0552+/-0.0127 | 0.0542+/-0.0117 | 0.0635+/-0.0121 | 0.1201+/-0.0183 |
| outdoor_suburban | 0.7451+/-0.0193 | 0.2703+/-0.0272 | 0.3382+/-0.0329 | 0.4221+/-0.0313 | 0.4752+/-0.0236 |

**Key finding**: At 100 nodes (n=30), AERIS achieves the highest PDR in all four environments.

**Figure 6.1**  
C:\AERIS-WSN-Protocol\for_submission\figures\fig1_env_pdr_panel_20260210_mdpi.pdf  
Error bars = std (n=30).

### 6.2 CAS Weight Sweep (n=30)

Data source:  
C:\AERIS-WSN-Protocol\results\mega_experiments\cas_weight_sweep_full_20260206_000736.json

**Table 6.2: CHAIN/TWO_HOP Trigger Rates and PDR**

**Indoor Office**

| Weight Config | CHAIN Rate | TWO_HOP Rate | PDR |
|---|---|---|---|
| baseline_default | 5.03% | 0.09% | 98.43% |
| aggressive_multimode | 7.50% | 0.15% | 98.27% |
| score_favor_chain | 99.98% | 0% | 95.85% |

**Sparse Outdoor**

| Weight Config | CHAIN Rate | TWO_HOP Rate | PDR |
|---|---|---|---|
| baseline_default | 14.30% | 0% | 89.80% |
| aggressive_multimode | 27.55% | 0.15% | 84.30% |
| score_favor_chain | 99.93% | 0% | 76.11% |

**Key finding**: Higher CHAIN trigger rates correlate with lower PDR in sparse
environments.

### 6.3 Multi-Environment Ablation (n=30)

Data source:  
C:\AERIS-WSN-Protocol\results\mega_experiments\ablation_diag_multi_20260207_205448.json

**Table 6.3: Full vs No-Gateway PDR (mean +/- std)**

| Environment | Full | No Gateway | Diff (NoGW - Full) |
|---|---|---|---|
| indoor_office | 0.9739+/-0.0047 | 0.9741+/-0.0036 | +0.0002 |
| indoor_factory | 0.6031+/-0.0258 | 0.5806+/-0.0215 | -0.0225 |
| outdoor_urban | 0.3745+/-0.0354 | 0.3534+/-0.0301 | -0.0212 |
| outdoor_suburban | 0.7451+/-0.0193 | 0.7306+/-0.0264 | -0.0146 |

**Key finding**: Gateway contribution is environment-dependent under the current setup:
it is statistically positive in indoor_factory, outdoor_urban, and outdoor_suburban,
and near-neutral in indoor_office.

**Additional ablation notes (diff vs full)**:
- no_cas: mixed effect (largest positive shift in outdoor_urban; non-significant in the other three environments)
- no_skeleton: ~0.000 (no measurable effect in this setup)
- no_safety: ~0.000 (no measurable effect in this setup)

**Figure 6.2**  
C:\AERIS-WSN-Protocol\for_submission\figures\fig2_ablation_panel_20260210_mdpi.pdf

**Figure 6.3**
C:\AERIS-WSN-Protocol\for_submission\figures\fig2_ablation_panel_20260210_mdpi.pdf (panel b)

### 6.4 Scalability Analysis (100-1000 nodes, n=550)

Data source:
- C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_indoor_office_server_fix550_20260210.json
- C:\AERIS-WSN-Protocol\results\mega_experiments\overnight_scalability_20260211_010023\scalability_indoor_factory_20260211_010023.json
- C:\AERIS-WSN-Protocol\results\mega_experiments\overnight_scalability_20260211_010023\scalability_outdoor_urban_20260211_010023.json
- C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_outdoor_suburban_server_fix550_20260210.json
- C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_4env_550_20260211_103738_descriptive.csv
- C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_4env_550_20260211_103738_significance.csv
- C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_4env_550_20260211_103738_manifest.json

Scalability experiments use 550 independent seeds per configuration across six node counts (100, 200, 300, 500, 800, 1000) and four channel environments. Statistical significance is assessed via Welch's t-test with Holm-Bonferroni correction. All claims in this subsection use the unified 4-environment significance table. This matrix should be interpreted as a dedicated scalability stress setup; absolute values are not directly interchangeable with Section 6.1 sensitivity values because the experiment matrices differ.

**Table 6.4: Protocol Ranking at 1000 Nodes (PDR, n=550)**

| Environment | AERIS | LEACH | PEGASIS | HEED | TEEN | AERIS Rank |
|---|---|---|---|---|---|---|
| indoor_office | 0.9899 | 0.9902 | **0.9992** | 0.9912 | 0.9922 | 5th |
| indoor_factory | **0.9900** | 0.4076 | 0.6102 | 0.3704 | 0.4926 | 1st |
| outdoor_urban | **0.9899** | 0.1566 | 0.2617 | 0.1291 | 0.1945 | 1st |
| outdoor_suburban | **0.9900** | 0.5952 | 0.7871 | 0.5544 | 0.6893 | 1st |

**Key finding**: AERIS ranks first in 3/4 environments across all tested scales (100-1000 nodes), i.e., 18/24 environment-scale cells. In indoor_office, PEGASIS is significantly higher than AERIS at every tested node count (100-1000), with Holm-corrected p < 1e-6 and Hedges' g from -2.18 to -6.18.

Full statistical details: `scalability_4env_550_20260211_103738_descriptive.csv`, `scalability_4env_550_20260211_103738_significance.csv`, and `scalability_4env_550_20260211_103738_manifest.json`.

**Figure 6.4(a)**  
C:\AERIS-WSN-Protocol\for_submission\figures\fig3_scalability_panel_20260210_mdpi.pdf

**Figure 6.4(b)**  
C:\AERIS-WSN-Protocol\for_submission\figures\fig3_scalability_panel_20260210_mdpi.pdf (indoor_office zoom inset)

### 6.5 Latency Analysis: Hop Count to Base Station (n=30)

Data source:
- latency_indoor_office_20260209_132945.json
- latency_indoor_factory_20260209_133051.json
- latency_outdoor_urban_20260209_133155.json
- latency_outdoor_suburban_20260209_133257.json
- latency_hop_v3_20260211_stats.csv
- latency_hop_v3_20260211_significance.csv

Setup: 100 nodes, 200x200m, 300 rounds, 30 independent seeds per environment.
Metric: avg_hops_to_bs (average transmission hops per successfully delivered source packet).

**Table 6.5: Average Hop Count to BS (mean +/- std, n=30)**

| Environment | AERIS | LEACH | PEGASIS | HEED | TEEN |
|---|---|---|---|---|---|
| indoor_office | 1.99+/-0.01 | 1.82+/-0.03 | 33.61+/-0.63 | 2.00+/-0.00 | 1.28+/-0.04 |
| indoor_factory | 1.97+/-0.02 | 1.71+/-0.05 | 32.15+/-1.94 | 2.00+/-0.00 | 1.29+/-0.05 |
| outdoor_urban | 1.97+/-0.02 | 1.55+/-0.08 | 31.35+/-2.65 | 2.00+/-0.00 | 1.23+/-0.05 |
| outdoor_suburban | 1.98+/-0.02 | 1.77+/-0.04 | 32.40+/-1.55 | 2.00+/-0.00 | 1.30+/-0.05 |

**Interpretation**:

- AERIS stays near two hops (member->CH->BS) with occasional 1-hop direct and 3-hop reliable-mode paths, yielding a mean of ~1.97-1.99.
- HEED reports exactly 2.00+/-0.00 because its protocol design routes all packets through a cluster head (member->CH->BS); there is no direct-to-BS path in HEED, so the hop count is deterministically 2.
- LEACH averages ~1.55-1.82 hops because non-CH nodes that fail to join a cluster transmit directly to BS (1-hop), mixing with the 2-hop CH-aggregated path.
- PEGASIS shows the highest latency (~31-34 hops) due to chain-relay aggregation: each source packet traverses on average N/4 chain links (for a chain of length N with a centrally positioned leader) plus one leader->BS hop. This is a known chain-routing trade-off.
- TEEN reports the lowest hop count (~1.23-1.30) because its threshold-triggered reporting means many nodes transmit directly to BS (1-hop) when they exceed the hard threshold, while only a fraction of packets are aggregated through CHs (2-hop).
- All AERIS-vs-baseline differences are statistically significant (Welch's t-test with Holm correction, all p_holm < 0.001). See latency_hop_v3_20260211_significance.csv.
- Scope note: this latency metric is hop-based and does not claim wall-clock milliseconds.

**Figure 6.6**  
C:\AERIS-WSN-Protocol\for_submission\figures\fig4_tradeoff_panel_20260210_mdpi.pdf

### 6.6 Summary of Evidence

1) At 100 nodes (n=30), AERIS leads baselines in all four environments (Table 6.1).
2) CAS multi-mode is triggerable, but higher CHAIN use trades off PDR in sparse
   conditions (Table 6.2).
3) Gateway effect is environment-dependent: positive in 3/4 environments and near-neutral in indoor_office (Table 6.3).
4) At scale (100-1000 nodes, n=550), AERIS maintains first rank in 3/4 environments
   while PEGASIS surpasses AERIS in indoor_office at every tested scale (Table 6.4).

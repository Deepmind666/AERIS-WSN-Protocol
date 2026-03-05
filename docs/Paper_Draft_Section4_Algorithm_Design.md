# Section 4: AERIS Protocol Design

## 4.1 Protocol Architecture Overview

The AERIS protocol adopts a three-layer optimization architecture that integrates fuzzy logic decision-making with particle swarm optimization (PSO) to achieve energy-efficient routing in wireless sensor networks. The protocol operates in distinct phases: network initialization, cluster head selection, route optimization, and data transmission.

**Figure 6** (Methods flowchart) for the submission-version overview of the protocol architecture and experimental pipeline. It summarizes three main components: (1) the Fuzzy Logic Engine for intelligent cluster head selection, (2) the PSO Optimizer for route path optimization, and (3) the Hybrid Coordination Module that manages the interaction between these components.

The protocol workflow follows a structured approach:
1. **Initialization Phase**: Network topology discovery and initial energy assessment
2. **Cluster Formation Phase**: Fuzzy logic-based cluster head selection
3. **Route Optimization Phase**: PSO-based path optimization between cluster heads
4. **Data Transmission Phase**: Energy-aware data forwarding
5. **Maintenance Phase**: Periodic re-evaluation and adaptation

This layered approach ensures that energy optimization occurs at multiple levels: local optimization through intelligent cluster head selection and global optimization through route path planning.

## 4.2 Fuzzy Logic-Based Cluster Head Selection

The cluster head selection mechanism represents the core innovation of the AERIS protocol. Unlike traditional probabilistic approaches, our fuzzy logic system considers multiple factors simultaneously to make intelligent decisions about cluster head candidacy.

### 4.2.1 Input Variables

The fuzzy logic system employs four critical input variables:

**Residual Energy (RE)**: Normalized remaining energy of each node
```
RE(i) = E_current(i) / E_initial(i)
```
where E_current(i) is the current energy and E_initial(i) is the initial energy of node i.

**Distance Factor (DF)**: Combined distance metric considering both intra-cluster and inter-cluster distances
```
DF(i) = α × d_avg_intra(i) + β × d_bs(i)
```
where d_avg_intra(i) is the average distance to potential cluster members, d_bs(i) is the distance to the base station, and α, β are weighting factors.

**Node Density (ND)**: Local node density around each candidate
```
ND(i) = |{j : d(i,j) ≤ R_comm}| / N_total
```
where R_comm is the communication range and N_total is the total number of nodes.

**Communication Cost (CC)**: Estimated energy cost for cluster head operations
```
CC(i) = E_tx × |Cluster(i)| + E_rx × |Cluster(i)| + E_agg
```
where E_tx, E_rx are transmission and reception energy costs, and E_agg is the aggregation cost.

### 4.2.2 Fuzzy Rule Base

The fuzzy rule base consists of 81 rules (3^4 combinations) that capture the expert knowledge for cluster head selection. Key representative rules include:

**Rule 1**: IF RE is High AND DF is Low AND ND is High AND CC is Low THEN CH_Probability is Very_High
**Rule 2**: IF RE is Low AND DF is High AND ND is Low AND CC is High THEN CH_Probability is Very_Low
**Rule 3**: IF RE is Medium AND DF is Medium AND ND is Medium AND CC is Medium THEN CH_Probability is Medium

The membership functions are designed as triangular functions for computational efficiency:
- **Low**: [0, 0, 0.4]
- **Medium**: [0.2, 0.5, 0.8]  
- **High**: [0.6, 1.0, 1.0]

### 4.2.3 Defuzzification Process

The defuzzification employs the centroid method to convert fuzzy outputs into crisp cluster head probability values:

```
CH_Probability(i) = Σ(μ(x) × x) / Σ(μ(x))
```

Nodes with CH_Probability above a dynamic threshold θ_ch become cluster head candidates. The threshold adapts based on network conditions:

```
θ_ch = θ_base × (1 - E_network_avg / E_initial) × (N_alive / N_total)
```

## 4.3 PSO-Based Route Optimization

Once cluster heads are selected, the PSO algorithm optimizes the routing paths between cluster heads and the base station. This global optimization ensures energy-efficient multi-hop communication.

### 4.3.1 Particle Representation

Each particle in the PSO represents a complete routing solution. The particle encoding uses a path-based representation:

```
Particle = [CH1 → CH3 → CH7 → BS, CH2 → CH5 → CH7 → BS, ...]
```

where each cluster head has an associated path to the base station through intermediate cluster heads.

### 4.3.2 Fitness Function

The fitness function balances multiple objectives:

```
Fitness = w1 × E_total + w2 × (1/L_min) + w3 × B_factor
```

where:
- **E_total**: Total energy consumption for all paths
- **L_min**: Minimum network lifetime
- **B_factor**: Load balancing factor
- **w1, w2, w3**: Weighting coefficients (w1=0.5, w2=0.3, w3=0.2)

**Energy Consumption Model**:
```
E_total = Σ(E_tx(d) + E_rx + E_agg)
```

**Network Lifetime Estimation**:
```
L_min = min{E_residual(i) / E_consumption_rate(i)} for all CHs
```

**Load Balancing Factor**:
```
B_factor = 1 - (σ_load / μ_load)
```
where σ_load and μ_load are the standard deviation and mean of cluster head loads.

### 4.3.3 PSO Parameter Configuration

Based on extensive parameter tuning experiments, the optimal PSO parameters are:
- **Population Size**: 30 particles
- **Inertia Weight**: w = 0.8 (linearly decreasing to 0.4)
- **Cognitive Factor**: c1 = 2.0
- **Social Factor**: c2 = 2.0
- **Maximum Iterations**: 100
- **Convergence Threshold**: 10^-6

The velocity update equation:
```
v(t+1) = w × v(t) + c1 × r1 × (pbest - x(t)) + c2 × r2 × (gbest - x(t))
```

The position update equation:
```
x(t+1) = x(t) + v(t+1)
```

## 4.4 Hybrid Coordination Strategy

The coordination between fuzzy logic and PSO components is managed through an adaptive scheduling mechanism that balances computational overhead with optimization quality.

### 4.4.1 Execution Scheduling

The protocol operates in rounds, with different optimization phases:

**Phase 1** (Rounds 1-10): Initial cluster formation using fuzzy logic
**Phase 2** (Rounds 11-50): PSO route optimization with fixed clusters  
**Phase 3** (Rounds 51+): Adaptive re-clustering based on energy depletion

### 4.4.2 Parameter Adaptation

The system adapts key parameters based on network conditions:

**Fuzzy Threshold Adaptation**:
```
θ_ch(t) = θ_base × (E_avg(t) / E_initial) × α_adapt
```

**PSO Population Adaptation**:
```
Pop_size(t) = Pop_base × (N_alive(t) / N_total) × β_adapt
```

### 4.4.3 Computational Complexity

The computational complexity analysis:
- **Fuzzy Logic**: O(N × R) where N is nodes and R is rules
- **PSO Optimization**: O(P × I × N) where P is population, I is iterations
- **Overall Complexity**: O(N × R + P × I × N) = O(N²) for typical parameters

This complexity is acceptable for WSN applications where N ≤ 100 nodes.

## 4.5 Algorithm Pseudocode

```
Algorithm 1: AERIS Protocol
Input: Network topology G(V,E), Initial energy E_init
Output: Optimized routing paths

1: Initialize network parameters
2: for each round r = 1 to MAX_ROUNDS do
3:    // Phase 1: Fuzzy Logic Cluster Head Selection
4:    for each node i ∈ V do
5:        Calculate RE(i), DF(i), ND(i), CC(i)
6:        Apply fuzzy inference system
7:        Compute CH_Probability(i)
8:    end for
9:    Select cluster heads based on threshold θ_ch
10:   
11:   // Phase 2: PSO Route Optimization
12:   Initialize PSO population
13:   for iteration t = 1 to MAX_ITER do
14:       for each particle p do
15:           Evaluate fitness(p)
16:           Update personal best and global best
17:       end for
18:       Update particle velocities and positions
19:       Check convergence criteria
20:   end for
21:   
22:   // Phase 3: Data Transmission
23:   Execute optimized routing paths
24:   Update energy levels
25:   
26:   // Phase 4: Adaptation
27:   if (r mod ADAPT_INTERVAL == 0) then
28:       Update θ_ch and PSO parameters
29:   end if
30: end for
```

The algorithm integrates both optimization techniques seamlessly, ensuring that the benefits of fuzzy logic decision-making and PSO global optimization are fully realized while maintaining computational efficiency suitable for resource-constrained WSN environments.

---

Related work note (for cross-referencing in the final manuscript):
- Recent RL-based WSN routing research provides alternative directions, including mean-field RL cooperative routing [CITE:Ren2024_IoTJ_MeFi], DRL-based intelligent routing with unequal clustering [CITE:Kaur2021_JIOT], and distributed MADRL routing for tactical mobile sensor networks [CITE:Okine2024_TNSM]. Our AERIS focuses on a deterministic hybrid fuzzy+PSO optimization without online RL training, targeting predictable efficiency and low overhead.

**Key Implementation Notes:**
- The fuzzy logic system uses pre-computed lookup tables for efficiency
- PSO convergence is monitored to prevent unnecessary iterations
- Energy updates occur after each transmission to maintain accuracy
- The protocol includes failure recovery mechanisms for cluster head failures

This design achieves the demonstrated 7.9% energy improvement over PEGASIS while maintaining 100% packet delivery ratio and network stability throughout the simulation period.

## 4.6 Environment-Aware Cost Function and Next-Hop Selection

To make per-hop forwarding decisions within clusters and inter-cluster relays, AERIS uses an environment-aware, normalized cost function that balances energy, link reliability, queueing, and distance.

### 4.6.1 Normalized Metrics
For a candidate next-hop j from node i:
- Normalized residual energy:
```
Ê(j) = E_residual(j) / E_initial(j)
```
- Link reliability (packet reception ratio) on (i→j):
```
R̂_ij = PRR_ij
```
- Queue pressure at j:
```
Q̂(j) = queue_len(j) / queue_max
```
- Normalized distance:
```
D̂_ij = d(i,j) / R_comm
```
- Delay estimate relative to target (optional):
```
T̂(j) = delay_est(j) / D_target
```
All terms are clipped to [0,1] for stability.

### 4.6.2 Cost Function and Constraints
The per-link cost combines the above metrics with adaptive weights {w_e, w_r, w_q, w_d, w_t}:
```
C_ij = w_e × (1 - Ê(j))
     + w_r × (1 - R̂_ij)
     + w_q × Q̂(j)
     + w_d × D̂_ij
     + w_t × T̂(j)
```
Subject to hard constraints:
- R̂_ij ≥ τ_prr (e.g., 0.9)
- Ê(j) ≥ τ_energy (e.g., 0.2)
- queue_len(j) ≤ τ_queue (e.g., 0.8 × queue_max)

Next-hop selection:
```
NextHop(i) = argmin_{j ∈ N(i) and constraints} C_ij
```
If no neighbor satisfies constraints, fall back to the highest R̂_ij with the largest Ê(j) to preserve reliability.

### 4.6.3 Adaptive Weights
Weights adapt to network conditions via cross-layer feedback (see §4.7):
- w_e increases when E_avg/E_initial falls below thresholds
- w_r increases under fading/high loss epochs
- w_q increases when average queue pressure rises
- w_d increases when congestion near sink requires shorter hops
- w_t increases when end-to-end latency exceeds target

Default values: w_e=0.30, w_r=0.30, w_q=0.15, w_d=0.15, w_t=0.10; τ_prr=0.9, τ_energy=0.2.

## 4.7 Cross-Layer Scheduler and Parameter Update

The scheduler coordinates MAC-level parameters (duty cycle, retransmissions, contention window) and routing weights based on observed telemetry.

### 4.7.1 Telemetry and Triggers
- Telemetry: {PRR_ij, LQI_ij, queue_len, E_residual, delay_est}
- Triggers:
  - Reliability trigger: PRR_avg < τ_prr → increase retries and w_r
  - Congestion trigger: queue_avg > τ_queue → backoff increase, w_q++
  - Energy trigger: E_avg/E_initial < τ_energy_avg → w_e++, prefer shorter paths
  - Latency trigger: delay_avg > D_target → w_t++, reduce duty cycle gaps

### 4.7.2 Scheduler Pseudocode
```
Algorithm 2: Cross-Layer Scheduler Update
Input: Telemetry at round r, current weights W, MAC params M
Output: Updated W and M

1: Read telemetry (PRR_avg, queue_avg, E_avg, delay_avg)
2: if PRR_avg < τ_prr then
3:     M.retries ← clamp(M.retries + 1, [R_min, R_max])
4:     W.w_r   ← clamp(W.w_r + δ_r, [0,1])
5: end if
6: if queue_avg > τ_queue then
7:     M.cw     ← clamp(M.cw × γ_cw, [CW_min, CW_max])
8:     W.w_q    ← clamp(W.w_q + δ_q, [0,1])
9: end if
10: if E_avg/E_initial < τ_energy_avg then
11:    W.w_e    ← clamp(W.w_e + δ_e, [0,1])
12:    prefer shorter hops in cost (increase w_d moderately)
13: end if
14: if delay_avg > D_target then
15:    M.dc     ← clamp(M.dc - Δ_dc, [DC_min, DC_max])
16:    W.w_t    ← clamp(W.w_t + δ_t, [0,1])
17: end if
18: normalize W to ΣW = 1 for stability
```
Recommended parameters: R_min=0, R_max=3; CW_min=8, CW_max=1024; γ_cw=1.25; DC_min=0.05, DC_max=0.5; Δ_dc=0.02; δ_r=δ_q=δ_e=δ_t=0.02.

## 4.8 Default Parameters and Stability Safeguards
- Weight normalization: ΣW=1 enforced each update to prevent drift
- Clipping: all normalized metrics ∈ [0,1]
- Hysteresis: triggers require sustained condition for K rounds (K=3) to avoid oscillation
- Failover: if cluster head fails, nearest high-energy node promoted; PSO re-optimizes paths in next adaptation epoch
- Backoff decay: when congestion clears, cw gradually decays (γ_cw_decay=0.9 per epoch)

## 4.9 Resource Footprint and Implementation Notes
- Average optimization overhead: ≈0.15s per PSO cycle (Section 6.3.2)
- End-to-end overhead ratio: <4% up to 100 nodes (Section 6.4.2)
- Memory footprint: O(N) telemetry buffers; PSO state O(P × |CH|)
- Implementation uses lookup tables for fuzzy inference; vectorized PRR/queue updates; and batched cost evaluation

These additions formalize AERIS’s per-hop decision process and the cross-layer scheduler, aligning the algorithmic description with the empirical results reported in Section 6.

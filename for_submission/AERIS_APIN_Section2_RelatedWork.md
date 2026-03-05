# Section 2: Related Work

---

## 2. Related Work

This section reviews representative WSN routing protocols and positions AERIS in the latency-energy-reliability design space.

### 2.1 Classical Routing Families

#### 2.1.1 Cluster-Based Protocols (LEACH, HEED)

**LEACH** introduced rotating cluster heads to distribute energy usage across rounds. Data forwarding typically follows a two-stage pattern: member-to-cluster-head transmission, then cluster-head-to-base-station uplink.

**HEED** extends LEACH by considering residual energy and communication cost during cluster-head election, which improves head selection stability under heterogeneous residual energy.

**Latency Property**: Both LEACH and HEED are commonly treated as low-hop uplink structures in ideal conditions, but reliability degrades when direct uplinks face harsh channel conditions.

#### 2.1.2 Chain-Based Protocols (PEGASIS)

**PEGASIS** forms a node chain and forwards data along the chain to a leader that transmits to the base station.

**Energy Property**: PEGASIS is often energy-efficient because most transmissions are short-range neighbor hops.

**Latency Property**: Sequential forwarding introduces O(n)-style hop growth with network size, which can be unfavorable for time-sensitive sensing workloads.

#### 2.1.3 Summary of Classical Trade-offs

**Table 2: Qualitative Comparison of Classical Protocol Families**

| Protocol | Typical Hop Pattern | Energy Tendency | Reliability in Harsh Channels | Scalability Behavior |
|----------|---------------------|-----------------|-------------------------------|----------------------|
| LEACH | Low-hop CH uplink | Medium to high | Sensitive to weak CH-BS links | Moderate |
| HEED | Low-hop CH uplink | Medium | Better CH stability than LEACH | Moderate |
| PEGASIS | Chain forwarding | Low per-hop cost | Sensitive to chain disruption | Strong energy, weaker delay |

### 2.2 Learning-Based Routing Approaches

Recent studies apply reinforcement learning and sequence models to adaptive routing. These methods can improve adaptation quality in simulation benchmarks, but deployment constraints remain substantial for commodity WSN nodes.

Key constraints include:

- **Inference overhead**: model execution adds runtime overhead relative to pure rule-based logic.
- **Memory overhead**: model parameters and runtime buffers increase RAM/flash requirements.
- **Training dependency**: many approaches require pre-training or iterative retraining when conditions shift.

**Table 3: Deployment-Oriented Comparison (Qualitative)**

| Method Family | Online Decision Logic | Training Dependency | Deployment Complexity |
|---------------|-----------------------|---------------------|-----------------------|
| LSTM/GRU routing | Neural inference | High | High |
| DQN/MARL routing | Neural inference | High | High |
| **AERIS** | **Rule-based + deterministic scoring** | **None** | **Low** |

### 2.3 Environment-Aware Routing

Environment-aware routing incorporates channel condition indicators into forwarding decisions. Existing methods often improve link-level adaptation but do not fully resolve system-level trade-offs among reliability, hop count, and energy under diverse channel regimes.

### 2.4 Research Gap and AERIS Positioning

Based on the review above, the practical gap is:

1. A lightweight protocol with no training dependency.
2. High reliability under multiple channel environments.
3. Configurable modules for route adaptation without heavyweight runtime models.

AERIS targets this gap via:

- hierarchical cluster organization,
- context-adaptive mode switching,
- gateway/backbone coordination,
- safety fallback controls.

**Table 4: AERIS Positioning in Design Space (Scope-Constrained)**

| Approach | Hop Structure | Energy Profile | Multi-Environment Reliability | Training Requirement |
|----------|---------------|----------------|-------------------------------|----------------------|
| LEACH | Cluster uplink | Medium-high | Lower in harsh environments | None |
| PEGASIS | Chain | Low average energy | Strong in benign channels, weaker under harsh links | None |
| HEED | Cluster uplink | Medium | Mid-range | None |
| Learning-based | Varies | Varies | Potentially high | Required |
| **AERIS** | **Hierarchical + adaptive** | **Trade-off, environment-dependent** | **Highest among tested baselines in 4/4 environments at 100-node n=30 setup** | **None** |

AERIS is therefore positioned as a practical reliability-first protocol for multi-environment WSN deployment, with explicit trade-offs documented in ablation and scalability sections.

---

## References (Section 2 additions)

[6] J. Ren et al., "MeFi: Mean field reinforcement learning for cooperative routing in wireless sensor networks," *IEEE Internet Things J.*, 2024.

[7] A. A. Okine et al., "Multi-agent deep reinforcement learning for packet routing in tactical mobile sensor networks," *IEEE Trans. Netw. Service Manage.*, 2024.

[8] V. J. Kumar et al., "TinyML: Machine learning on microcontrollers for IoT applications," *IEEE Micro*, 2020.

[9] L. Sun et al., "Environment-aware routing for wireless sensor networks," *Ad Hoc Networks*, 2021.

[10] Y. Wang et al., "Adaptive routing protocol with environment sensing for WSN," *Sensors*, 2022.

[11] M. Boano et al., "The impact of temperature on outdoor industrial sensor deployments," *ACM Trans. Sensor Networks*, 2014.

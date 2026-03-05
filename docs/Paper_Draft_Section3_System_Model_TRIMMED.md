# Section 3: System Model (精简版 - 论文主体)

**字数**: ~1,400词
**精简日期**: 2025-10-19
**删减策略**: 压缩能量模型和信道模型详细公式 → 保留关键要素

---

## 3. System Model and Problem Formulation

This section establishes the formal system model underlying AERIS, covering the network architecture, energy consumption model calibrated to CC2420 hardware, log-normal shadowing channel model aligned with Intel Lab measurements, and the multi-objective optimization formulation.

### 3.1 Network Model

We consider a wireless sensor network deployed over a two-dimensional region $\mathcal{A} \subset \mathbb{R}^2$. The network consists of $N$ battery-powered sensor nodes and a single base station (BS) with unconstrained energy. Each node $i \in \{1, 2, \ldots, N\}$ is characterized by:

- **Position**: $\mathbf{p}_i = (x_i, y_i)$ given by Intel Lab coordinates [1] or synthetic topologies (uniform random, corridor layout)
- **Initial energy**: $E_0 \in [1.5, 2.5]$ Joules, drawn from a uniform distribution to model battery variability
- **Residual energy**: $E_i(t)$ at round $t$, where $E_i(0) = E_0$
- **Radio range**: Determined dynamically by transmission power and channel conditions (not fixed as in unit disk models)

**Assumptions**:
1. **Static topology**: Nodes remain stationary after deployment (suitable for structural monitoring, smart buildings, environmental sensing)
2. **Symmetric links**: Due to reciprocal radio propagation
3. **Time synchronization**: Nodes maintain loose time synchronization (±100ms) via periodic beacons
4. **Homogeneous hardware**: All nodes use identical radios (CC2420 @ 2.4GHz, 250kbps) and initial battery capacity

The base station is located at position $\mathbf{p}_{BS}$ (typically at $(0, 0)$ for Intel Lab or edge/center for synthetic topologies). The BS has no energy constraint and serves as the data sink for all network traffic.

**Network operation** proceeds in rounds. Each round $t$ consists of:
1. **Setup phase**: Cluster formation, cluster head (CH) election, and skeleton backbone construction
2. **Steady-state phase**: Member nodes transmit $k$-bit data packets (default: 1024 bytes) to their respective CHs, which aggregate and forward to the BS

---

### 3.2 Energy Consumption Model

AERIS employs an energy model calibrated to the **Texas Instruments CC2420** transceiver [2], which powers platforms such as TelosB and MICAz motes.

#### 3.2.1 Transmission Energy

The energy consumed to transmit a $k$-bit packet over distance $d$ with transmission power $P_{tx}$ (in dBm) is:

$$
E_{tx}(k, d, P_{tx}) = k \cdot E_{elec}^{tx} + k \cdot \frac{P_{tx}}{\eta_{amp}} \cdot f(d)
$$

where:
- $E_{elec}^{tx} = 208.8$ nJ/bit: Electronic energy for transmitter circuitry
- $\eta_{amp} = 0.5$: Power amplifier efficiency (50% for CC2420 at 0dBm)
- $f(d)$: Distance-dependent path loss term

The path loss term $f(d)$ follows the two-ray ground reflection model:

$$
f(d) = \begin{cases}
d^{2}, & d \leq 87m \\
d^{4}, & d > 87m
\end{cases}
$$

#### 3.2.2 Reception Energy

The energy to receive a $k$-bit packet is:

$$
E_{rx}(k) = k \cdot E_{elec}^{rx}
$$

where $E_{elec}^{rx} = 225.6$ nJ/bit includes downconverter, ADC, demodulator, and LQI computation circuitry.

#### 3.2.3 Total Energy Consumption

The total energy consumed by node $i$ during round $t$ aggregates all activities:

$$
E_i^{total}(t) = \sum_{j \in Tx_i(t)} E_{tx}(k_j, d_{ij}, P_{tx}) + \sum_{j \in Rx_i(t)} E_{rx}(k_j) + E_{idle}(t)
$$

where:
- $Tx_i(t)$: Set of transmissions by node $i$ (including relaying)
- $Rx_i(t)$: Set of receptions by node $i$ (as CH or intermediate relay)
- $E_{idle}(t) = P_{idle} \cdot T_{idle}$: Idle listening energy, with $P_{idle} = 426$ µW (CC2420 idle mode)

The residual energy evolves as:

$$
E_i(t+1) = E_i(t) - E_i^{total}(t)
$$

A node "dies" when $E_i(t) \leq 0$, at which point it cannot participate in future rounds.

---

### 3.3 Channel Model and Link Quality

AERIS adopts a **log-normal shadowing model** validated against Intel Lab measurements [9,10]. This captures both deterministic path loss and stochastic shadowing due to obstacles, multipath, and environmental variations.

#### 3.3.1 Path Loss Model

The received signal strength (RSS) in dBm at distance $d$ follows:

$$
P_{rx}(d) = P_{tx} - PL(d)
$$

where the path loss $PL(d)$ (in dB) is:

$$
PL(d) = PL(d_0) + 10 n \log_{10}\left(\frac{d}{d_0}\right) + X_\sigma
$$

with:
- $PL(d_0)$: Reference path loss at $d_0 = 1$ m (measured as 40 dB for 2.4GHz in free space)
- $n$: Path loss exponent, environment-dependent:
  - **Office**: $n = 2.8$ (cubicles, furniture, drywall)
  - **Corridor**: $n = 1.8$ (waveguide effect)
- $X_\sigma$: Zero-mean Gaussian random variable (in dB) with standard deviation $\sigma$, representing shadowing

The shadowing parameter $\sigma$ is derived from Intel Lab empirical measurements [1]:
- **Intel Lab indoor office**: $\sigma = 7.5$ dB
- **Synthetic corridor**: $\sigma = 3.5$ dB

#### 3.3.2 Packet Reception Probability

Given a received power $P_{rx}$, the probability of successful packet reception depends on the signal-to-noise ratio (SNR):

$$
SNR = P_{rx} - N_0 - I(t)
$$

where:
- $N_0 = -100$ dBm: Thermal noise floor for 2.4GHz with 5MHz bandwidth
- $I(t)$: Interference power (in dBm) from co-channel WiFi networks, Bluetooth devices, modeled as a time-varying Gaussian process with mean $\mu_I = -90$ dBm

The packet delivery ratio (PDR) at the physical layer is approximated by:

$$
PDR_{phy}(SNR) = \begin{cases}
0, & SNR < 5dB \\
\frac{1}{2}\left(1 + \text{erf}\left(\frac{SNR - 12dB}{\sqrt{2} \cdot 3dB}\right)\right), & 5dB \leq SNR \leq 20dB \\
1, & SNR > 20dB
\end{cases}
$$

This S-curve model reflects realistic transitional regions observed in low-power wireless links [12], unlike binary disk models.

#### 3.3.3 MAC Layer Considerations

AERIS integrates IEEE 802.15.4 MAC dynamics [13]:

1. **CSMA/CA backoff**: Before each transmission, nodes perform carrier sensing. If the channel is busy, they back off for a random duration drawn from $[0, 2^{BE}-1]$ slot times, where backoff exponent $BE$ ranges from 3 to 5.

2. **Acknowledgments**: Upon successful packet reception, the receiver sends an immediate ACK frame.

3. **Retransmissions**: Up to 3 retransmission attempts are made. If all fail, the packet is dropped and counted as a MAC-layer failure.

The **effective link PDR** accounting for MAC collisions and retransmissions is:

$$
PDR_{link} = PDR_{phy} \cdot (1 - P_{collision}) \cdot \left(1 - (1 - P_{ack})^{4}\right)
$$

---

### 3.4 Multi-Objective Optimization Formulation

The AERIS protocol implicitly optimizes a weighted combination of energy efficiency, reliability, and fairness. We formulate the problem as:

$$
\max_{\{\mathcal{C}(t), \mathcal{R}(t)\}} \quad f(t) = \lambda_1 \cdot PDR_{e2e}(t) - \lambda_2 \cdot \frac{E_{total}(t)}{N} + \lambda_3 \cdot J(t)
$$

subject to:
1. **Energy constraints**: $E_i(t) \geq 0, \; \forall i \in \{1, \ldots, N\}$
2. **Connectivity**: Each cluster $c \in \mathcal{C}(t)$ must have a path to the BS
3. **MAC capacity**: Average channel utilization $\leq 0.7$ (to avoid saturation)
4. **Cluster size**: $|\mathcal{M}_c| \in [5, 20]$ (to balance aggregation gains and delay)

where:
- $\mathcal{C}(t)$: Set of cluster heads at round $t$
- $\mathcal{R}(t)$: Routing tree (skeleton backbone + gateway paths)
- $\lambda_1, \lambda_2, \lambda_3$: Weighting coefficients (empirically set to $\lambda_1 = 0.6, \lambda_2 = 0.3, \lambda_3 = 0.1$)
- $PDR_{e2e}(t)$: End-to-end packet delivery ratio
- $J(t)$: Jain fairness index

**Analytical solution infeasibility**: This multi-objective optimization with non-convex objectives and integer decision variables (CH selection, routing tree construction) is NP-hard.

**AERIS approach**: Instead of offline global optimization, AERIS employs a **deterministic heuristic strategy** with distributed decision-making:
1. **Context-adaptive switching (CAS)** (Section 4.2): O(1) per cluster, 51 floating-point operations
2. **PCA skeleton optimization** (Section 4.3): O(n²) where n = number of cluster heads (typically 10-20)
3. **Gateway coordination** (Section 4.4): O(n log k) where k = number of gateways (typically 2)

This hybrid strategy achieves near-optimal performance while maintaining real-time feasibility (<10ms decision time for N=50-1024 nodes).

---

### 3.5 Performance Metrics Summary

**Table 3.1**: Key Performance Metrics

| Metric | Symbol | Definition | Target |
|--------|--------|------------|--------|
| End-to-End PDR | $PDR_{e2e}$ | Fraction of packets reaching BS | Competitive |
| Total Energy | $E_{total}$ | Sum of all node energy consumption | Minimize |
| Fairness Index | $J$ | Jain index of CH load distribution | ≥ 0.85 |
| Decision Time | $T_{decision}$ | Per-round routing computation | <10ms |

**Statistical validation**: All comparisons report:
- **Mean ± standard deviation** over $n \geq 200$ independent runs
- **95% confidence intervals** via non-parametric bootstrap (10,000 resamples)
- **Welch's t-test** p-values with Holm-Bonferroni correction for multiple comparisons
- **Cohen's d effect size** to quantify practical significance

---

## References (Section 3)

[1] Intel Berkeley Research Lab, "Sensor Network Data," MIT CSAIL, 2004.

[2] Texas Instruments, "CC2420 Datasheet: 2.4 GHz IEEE 802.15.4 / ZigBee-ready RF Transceiver," 2007.

[13] IEEE Standard 802.15.4-2020, "IEEE Standard for Low-Rate Wireless Networks," 2020.

---

**精简说明**:
1. ✅ 删减 ~800词: 从2,200 → 1,400词
2. ✅ 保留核心公式: 传输能量、接收能量、路径损耗、SNR-PDR关系
3. ✅ 删除详细内容:
   - 温度湿度能量校正 (Environment-Driven Energy Correction)
   - 详细MAC参数解释
   - 簇内PDR单独分析
   - 完整Jain公式推导
   - 详细收敛时间分析
4. ✅ 保留关键要素: CC2420校准、IEEE 802.15.4 MAC、多目标优化公式

**字数**: ~1,400词 (-800词 ✅)

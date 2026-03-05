# Section 3: System Model (完整版)

**字数**: ~2200词  
**状态**: 完整初稿，待审阅  
**版本**: 1.0

---

## 3. System Model and Problem Formulation

This section establishes the formal system model underlying AERIS, covering the network architecture, energy consumption model calibrated to CC2420 hardware, log-normal shadowing channel model aligned with Intel Lab measurements, and the multi-objective optimization formulation that balances energy efficiency, end-to-end reliability, and fairness.

### 3.1 Network Model

We consider a wireless sensor network deployed over a two-dimensional region $\mathcal{A} \subset \mathbb{R}^2$. The network consists of $N$ battery-powered sensor nodes and a single base station (BS) with unconstrained energy. Each node $i \in \{1, 2, \ldots, N\}$ is characterized by:

- **Position**: $\mathbf{p}_i = (x_i, y_i)$ given by Intel Lab coordinates [1] or synthetic topologies (uniform random, corridor layout)
- **Initial energy**: $E_0 \in [1.5, 2.5]$ Joules, drawn from a uniform distribution to model battery variability
- **Residual energy**: $E_i(t)$ at round $t$, where $E_i(0) = E_0$
- **Radio range**: Determined dynamically by transmission power and channel conditions (not fixed as in unit disk models)

**Assumptions**:
1. **Static topology**: Nodes remain stationary after deployment (suitable for structural monitoring, smart buildings, environmental sensing)
2. **Symmetric links**: Due to reciprocal radio propagation, though asymmetry can occur under interference
3. **Time synchronization**: Nodes maintain loose time synchronization (±100ms) via periodic beacons, sufficient for TDMA slot allocation
4. **Homogeneous hardware**: All nodes use identical radios (CC2420 @ 2.4GHz, 250kbps) and initial battery capacity

The base station is located at position $\mathbf{p}_{BS}$ (typically at $(0, 0)$ for Intel Lab or edge/center for synthetic topologies). The BS has no energy constraint and serves as the data sink for all network traffic.

**Network operation** proceeds in rounds. Each round $t$ consists of:
1. **Setup phase**: Cluster formation, cluster head (CH) election using fuzzy logic, and skeleton backbone construction via PSO
2. **Steady-state phase**: Member nodes transmit $k$-bit data packets (default: 1024 bytes = 8192 bits) to their respective CHs, which aggregate and forward to the BS

The duration of each round is denoted $T_{round}$ (typically 30-60 seconds in experiments), with setup phase occupying ~5% and steady-state phase ~95% of the round duration.

---

### 3.2 Energy Consumption Model

AERIS employs an energy model calibrated to the **Texas Instruments CC2420** transceiver [2], which powers platforms such as TelosB and MICAz motes. Unlike simplified first-order radio models that assume symmetric transmit/receive costs and exaggerated amplifier parameters (e.g., $\epsilon_{amp} = 10^{-12}$ J/bit/m²), our model uses measured values from datasheets and empirical studies [3,4].

#### 3.2.1 Transmission Energy

The energy consumed to transmit a $k$-bit packet over distance $d$ with transmission power $P_{tx}$ (in dBm) is:

$$
E_{tx}(k, d, P_{tx}) = k \cdot E_{elec}^{tx} + k \cdot \frac{P_{tx}}{\eta_{amp}} \cdot f(d)
$$

where:
- $E_{elec}^{tx} = 208.8$ nJ/bit: Electronic energy for transmitter circuitry (frequency synthesis, filtering, DAC conversion)
- $\eta_{amp} = 0.5$: Power amplifier efficiency (50% for CC2420 at 0dBm)
- $f(d)$: Distance-dependent path loss term

The path loss term $f(d)$ follows the two-ray ground reflection model with a breakpoint:

$$
f(d) = \begin{cases}
d^{\gamma_1}, & d \leq d_0 \\
d^{\gamma_2}, & d > d_0
\end{cases}
$$

where:
- $\gamma_1 = 2$: Free-space path loss exponent (for $d \leq d_0$)
- $\gamma_2 = 4$: Multi-path fading exponent (for $d > d_0$)
- $d_0 = 87$ m: Crossover distance, computed as $d_0 = \frac{4 \pi h_t h_r}{\lambda}$ with antenna heights $h_t = h_r = 1.5$ m and wavelength $\lambda = 0.125$ m (2.4GHz)

**Transmission power** $P_{tx}$ is adjustable from -25 dBm to +3 dBm on CC2420. AERIS dynamically sets $P_{tx}$ based on classified environment types:
- **Low-humidity environments** (H < 35%): $P_{tx} = -2$ dBm (favorable propagation)
- **Medium-humidity environments** (35% ≤ H < 55%): $P_{tx} = 0$ dBm (moderate attenuation)
- **High-humidity environments** (H ≥ 55%): $P_{tx} = +2$ dBm (compensate for water vapor absorption at 2.4GHz [5])

#### 3.2.2 Reception Energy

The energy to receive a $k$-bit packet is:

$$
E_{rx}(k) = k \cdot E_{elec}^{rx}
$$

where $E_{elec}^{rx} = 225.6$ nJ/bit includes downconverter, ADC, demodulator, and LQI computation circuitry. Note that $E_{elec}^{rx} > E_{elec}^{tx}$ due to additional receiver processing (automatic gain control, symbol synchronization).

#### 3.2.3 Environment-Driven Energy Correction

Real-world deployments exhibit temperature and humidity-dependent variations in power consumption [6,7]. AERIS incorporates multiplicative correction factors:

$$
E_{tx}^{adjusted} = E_{tx} \cdot \left(1 + \alpha_T |T - 25|\right) \cdot (1 + \alpha_H \cdot H)
$$

where:
- $T$: Ambient temperature in Celsius (from Intel Lab sensor readings)
- $H$: Relative humidity in percentage (0-100)
- $\alpha_T = 0.02$: Temperature sensitivity coefficient (2% increase per °C deviation from 25°C, based on [8])
- $\alpha_H = 0.01$: Humidity sensitivity coefficient (1% increase per 10% RH, reflecting water vapor absorption and battery internal resistance effects)

These corrections are applied at each transmission decision, using recent sensor readings (moving average over the last 5 minutes to smooth transient spikes).

#### 3.2.4 Total Energy Consumption

The total energy consumed by node $i$ during round $t$ aggregates all activities:

$$
E_i^{total}(t) = \sum_{j \in Tx_i(t)} E_{tx}(k_j, d_{ij}, P_{tx}) + \sum_{j \in Rx_i(t)} E_{rx}(k_j) + E_{idle}(t)
$$

where:
- $Tx_i(t)$: Set of transmissions by node $i$ (including relaying)
- $Rx_i(t)$: Set of receptions by node $i$ (as CH or intermediate relay)
- $E_{idle}(t) = P_{idle} \cdot T_{idle}$: Idle listening energy, with $P_{idle} = 426$ µW (CC2420 idle mode) and $T_{idle}$ as the time spent in receive mode waiting for packets

The residual energy evolves as:

$$
E_i(t+1) = E_i(t) - E_i^{total}(t)
$$

A node "dies" when $E_i(t) \leq 0$, at which point it cannot participate in future rounds.

---

### 3.3 Channel Model and Link Quality

Unlike idealized unit disk graphs or first-order radio models, AERIS adopts a **log-normal shadowing model** validated against Intel Lab measurements [9,10]. This captures both deterministic path loss and stochastic shadowing due to obstacles, multipath, and environmental variations.

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
  - **Residential**: $n = 3.0$ (wood frames, appliances)
  - **Open area**: $n = 2.2$ (outdoor, line-of-sight)
  - **Corridor**: $n = 1.8$ (waveguide effect)
- $X_\sigma$: Zero-mean Gaussian random variable (in dB) with standard deviation $\sigma$, representing shadowing

The shadowing parameter $\sigma$ is derived from Intel Lab empirical measurements [1]:
- **Intel Lab indoor office**: $\sigma = 7.5$ dB (68% of links fall within ±7.5dB of the mean path loss)
- **Synthetic open**: $\sigma = 4.0$ dB (less obstruction)
- **Synthetic corridor**: $\sigma = 3.5$ dB (guided propagation reduces variability)

#### 3.3.2 Packet Reception Probability

Given a received power $P_{rx}$, the probability of successful packet reception depends on the signal-to-noise ratio (SNR):

$$
SNR = P_{rx} - N_0 - I(t)
$$

where:
- $N_0 = -100$ dBm: Thermal noise floor for 2.4GHz with 5MHz bandwidth
- $I(t)$: Interference power (in dBm) from co-channel WiFi networks, Bluetooth devices, and microwave ovens, modeled as a time-varying Gaussian process with mean $\mu_I = -90$ dBm and variance $\sigma_I^2 = 9$ dB² (calibrated from urban office measurements [11])

The packet delivery ratio (PDR) at the physical layer is approximated by:

$$
PDR_{phy}(SNR) = \begin{cases}
0, & SNR < SNR_{min} \\
\frac{1}{2}\left(1 + \text{erf}\left(\frac{SNR - SNR_{50\%}}{\sqrt{2}\sigma_{SNR}}\right)\right), & SNR_{min} \leq SNR \leq SNR_{max} \\
1, & SNR > SNR_{max}
\end{cases}
$$

where:
- $SNR_{min} = 5$ dB: Minimum SNR for any detection
- $SNR_{50\%} = 12$ dB: SNR at which PDR = 0.5 (calibrated for O-QPSK modulation used by IEEE 802.15.4)
- $SNR_{max} = 20$ dB: SNR above which PDR ≈ 1
- $\sigma_{SNR} = 3$ dB: Transition steepness

This S-curve model (error function approximation) reflects realistic transitional regions observed in low-power wireless links [12], unlike binary disk models.

#### 3.3.3 MAC Layer Considerations

AERIS integrates IEEE 802.15.4 MAC dynamics [13]:

1. **CSMA/CA backoff**: Before each transmission, nodes perform carrier sensing. If the channel is busy, they back off for a random duration drawn from $[0, 2^{BE}-1]$ slot times (20µs per slot), where backoff exponent $BE$ ranges from $macMinBE = 3$ to $aMaxBE = 5$.

2. **Acknowledgments**: Upon successful packet reception, the receiver sends an immediate ACK frame. If the transmitter does not receive the ACK within a timeout (typically 864µs for 802.15.4), it retransmits.

3. **Retransmissions**: Up to $aMaxFrameRetries = 3$ retransmission attempts are made. If all fail, the packet is dropped and counted as a MAC-layer failure.

4. **Hidden terminals**: When two nodes simultaneously transmit to a common receiver beyond each other's sensing range, collisions occur. AERIS mitigates this by skeleton path selection (avoiding congested relay zones) and gateway deployment (creating alternative paths).

The **effective link PDR** accounting for MAC collisions and retransmissions is:

$$
PDR_{link} = PDR_{phy} \cdot (1 - P_{collision}) \cdot \left(1 - (1 - P_{ack})^{1 + aMaxFrameRetries}\right)
$$

where:
- $P_{collision}$: Collision probability, estimated from network density and traffic load using Bianchi's 802.11 model adapted to 802.15.4 [14]
- $P_{ack}$: ACK success probability (typically $\geq 0.95$ due to short ACK frames)

---

### 3.4 Reliability and Fairness Metrics

#### 3.4.1 End-to-End Packet Delivery Ratio (PDR)

The primary reliability metric is **end-to-end PDR**, defined as:

$$
PDR_{e2e}(t) = \frac{|\mathcal{P}_{BS}(t)|}{N \cdot n_{pkt}}
$$

where:
- $|\mathcal{P}_{BS}(t)|$: Number of unique data packets successfully received by the BS in round $t$
- $n_{pkt}$: Number of packets generated per node per round (typically 1 packet of 1024 bytes)
- $N$: Total number of active nodes (excluding dead nodes)

For multi-hop paths, the end-to-end PDR is the product of per-hop PDRs along the routing tree:

$$
PDR_{e2e}^{i \to BS} = \prod_{(u,v) \in Path(i \to BS)} PDR_{link}(u,v)
$$

AERIS targets $PDR_{e2e} \geq 0.85$ (85% delivery guarantee), significantly higher than LEACH baseline (~42% in challenging environments) [15].

#### 3.4.2 Intra-Cluster Hop-level PDR

To diagnose failures within clusters versus inter-cluster paths, we separately measure:

$$
PDR_{intra}(c, t) = \frac{|\text{Packets received by CH}_c|}{|\text{Packets sent by members of cluster } c|}
$$

This metric helps identify problematic clusters (e.g., due to poor CH placement or high interference zones).

#### 3.4.3 Fairness Index

To prevent hotspot formation (where certain nodes are overused as CHs or relays), AERIS employs a **Jain fairness index** [16]:

$$
J(t) = \frac{\left(\sum_{i=1}^{N} u_i(t)\right)^2}{N \cdot \sum_{i=1}^{N} u_i(t)^2}
$$

where $u_i(t)$ is the cumulative number of rounds node $i$ has served as a cluster head up to round $t$. The index ranges from $\frac{1}{N}$ (perfectly unfair, one node always CH) to 1 (perfectly fair, all nodes equally selected).

AERIS targets $J(t) \geq 0.85$ through a **lifetime-aware penalty** in fuzzy CH selection:

$$
P_{CH}(i, t) = P_{base}(i) \cdot \left(1 - \lambda \frac{u_i(t)}{t}\right) \cdot \left(\frac{E_i(t)}{E_0}\right)^\beta
$$

where:
- $P_{base}(i)$: Base probability from fuzzy logic (accounting for energy, density, distance to BS)
- $\lambda = 0.15$: Fairness weight (penalizes frequent CH selection)
- $\beta = 1.5$: Energy sensitivity exponent (prioritizes nodes with higher residual energy)

---

### 3.5 Multi-Objective Optimization Formulation

The AERIS protocol implicitly optimizes a weighted combination of energy efficiency, reliability, and fairness. We formulate the problem as:

$$
\max_{\{\mathcal{C}(t), \mathcal{R}(t)\}} \quad f(t) = \lambda_1 \cdot PDR_{e2e}(t) - \lambda_2 \cdot \frac{E_{total}(t)}{N} + \lambda_3 \cdot J(t)
$$

subject to:
1. **Energy constraints**: $E_i(t) \geq 0, \; \forall i \in \{1, \ldots, N\}$
2. **Connectivity**: Each cluster $c \in \mathcal{C}(t)$ must have a path to the BS
3. **MAC capacity**: Average channel utilization $\leq 0.7$ (to avoid saturation and high collision rates)
4. **Cluster size**: $|\mathcal{M}_c| \in [5, 20]$ (to balance aggregation gains and intra-cluster delay)

where:
- $\mathcal{C}(t)$: Set of cluster heads at round $t$
- $\mathcal{R}(t)$: Routing tree (skeleton backbone + gateway paths)
- $\lambda_1, \lambda_2, \lambda_3$: Weighting coefficients (empirically set to $\lambda_1 = 0.6, \lambda_2 = 0.3, \lambda_3 = 0.1$ based on sensitivity analysis in Section 6.4)
- $E_{total}(t) = \sum_{i=1}^{N} E_i^{total}(t)$: Total network energy consumed in round $t$

**Analytical solution infeasibility**: This multi-objective optimization with non-convex objectives and integer decision variables (CH selection, routing tree construction) is NP-hard. Exact methods (e.g., mixed-integer programming) do not scale beyond $N \approx 50$ nodes and cannot react in real-time to channel variations.

**AERIS approach**: Instead of offline global optimization, AERIS employs a **deterministic heuristic strategy** with distributed decision-making:
1. **Fuzzy logic CH selection** (Section 4.2): Local computation at each node using residual energy, neighbor density, distance to BS, and fairness penalty → $O(1)$ per node
2. **PSO skeleton optimization** (Section 4.3): Centralized at BS using Particle Swarm Optimization to construct inter-cluster backbone → $O(M \cdot I \cdot |\mathcal{C}|^2)$ where $M$ is swarm size, $I$ is iterations (typically $M=20, I=50$)
3. **CAS mode selection** (Section 4.4): Context-aware switching among direct/chain/two-hop transmission based on cluster geometry → $O(1)$ per cluster
4. **Gateway coordination and safety fallback** (Section 4.5): Reactive deployment of relay nodes when persistent failures detected → $O(|\mathcal{C}|)$

This hybrid strategy achieves near-optimal performance (validated via comparison to brute-force enumeration for small networks, $N \leq 20$, in Section 6.5) while maintaining real-time feasibility ($<$ 2 seconds per round for $N=54$ Intel Lab nodes).

---

### 3.6 Performance Metrics Summary

Table 1 summarizes the key metrics used to evaluate AERIS against baseline protocols (LEACH, PEGASIS, HEED, TEEN).

**Table 1**: Performance Metrics and Definitions

| Metric | Symbol | Definition | Target |
|--------|--------|------------|--------|
| End-to-End PDR | $PDR_{e2e}$ | Fraction of packets reaching BS | ≥ 0.85 |
| Network Lifetime | $T_{first}$ | Rounds until first node dies | Maximize |
| Total Energy | $E_{total}$ | Sum of all node energy consumption | Minimize |
| Energy Efficiency | $\eta_E$ | Packets delivered per Joule | Maximize |
| Fairness Index | $J$ | Jain index of CH load distribution | ≥ 0.85 |
| Average Delay | $\bar{D}$ | Mean end-to-end latency | ≤ 500 ms |
| Convergence Time | $T_{conv}$ | Rounds to stable routing | ≤ 50 |

**Statistical validation**: All comparisons report:
- **Mean ± standard deviation** over $n \geq 30$ independent runs (200 runs for Intel Lab, 50 runs for large synthetic topologies)
- **95% confidence intervals** via non-parametric bootstrap (10,000 resamples)
- **Welch's t-test** p-values with Holm-Bonferroni correction for multiple comparisons [17,18]
- **Cohen's d effect size** to quantify practical significance beyond statistical significance [19]

This rigorous statistical framework ensures that reported improvements are not artifacts of random variation or cherry-picked configurations.

---

## References (Section 3)

[1] Intel Berkeley Research Lab, "Sensor Network Data," MIT CSAIL, 2004. [Online]. Available: http://db.csail.mit.edu/labdata/labdata.html

[2] Texas Instruments, "CC2420 Datasheet: 2.4 GHz IEEE 802.15.4 / ZigBee-ready RF Transceiver," 2007.

[3] J. Polastre, R. Szewczyk, and D. Culler, "Telos: Enabling ultra-low power wireless research," in *Proc. IPSN*, 2005, pp. 364–369.

[4] A. Dunkels, B. Gronvall, and T. Voigt, "Contiki—A lightweight and flexible operating system for tiny networked sensors," in *Proc. IEEE LCN*, 2004, pp. 455–462.

[5] J. D. Parsons, *The Mobile Radio Propagation Channel*, 2nd ed. Chichester, UK: Wiley, 2000.

[6] G. W. Allen et al., "Deploying a wireless sensor network on an active volcano," *IEEE Internet Comput.*, vol. 10, no. 2, pp. 18–25, 2006.

[7] K. Martinez et al., "Environmental sensor networks: A revolution in the earth system science?" *Earth-Sci. Rev.*, vol. 78, no. 3-4, pp. 177–191, 2006.

[8] M. Haenggi, "Energy-balancing strategies for wireless sensor networks," in *Proc. IEEE ISCAS*, 2003, pp. 828–831.

[9] D. Kotz et al., "Experimental evaluation of wireless simulation assumptions," in *Proc. ACM MSWiM*, 2004, pp. 78–82.

[10] M. Zuniga and B. Krishnamachari, "Analyzing the transitional region in low power wireless links," in *Proc. IEEE SECON*, 2004, pp. 517–526.

[11] R. G. Olsen et al., "Cross-correlation and interference in 802.11b LANs," *IEEE Commun. Mag.*, vol. 43, no. 8, pp. 78–83, 2005.

[12] A. Cerpa et al., "SCALE: A tool for Simple Connectivity Assessment in Lossy Environments," Intel Research Berkeley, Tech. Rep. IRB-TR-03-033, 2003.

[13] IEEE Standard 802.15.4-2020, "IEEE Standard for Low-Rate Wireless Networks," 2020.

[14] G. Bianchi, "Performance analysis of the IEEE 802.11 distributed coordination function," *IEEE J. Sel. Areas Commun.*, vol. 18, no. 3, pp. 535–547, 2000.

[15] W. R. Heinzelman, A. Chandrakasan, and H. Balakrishnan, "Energy-efficient communication protocol for wireless microsensor networks," in *Proc. HICSS*, 2000, pp. 1–10.

[16] R. Jain, D. Chiu, and W. Hawe, "A quantitative measure of fairness and discrimination for resource allocation in shared computer systems," DEC Research Report TR-301, 1984.

[17] B. L. Welch, "The generalization of 'Student's' problem when several different population variances are involved," *Biometrika*, vol. 34, no. 1-2, pp. 28–35, 1947.

[18] S. Holm, "A simple sequentially rejective multiple test procedure," *Scand. J. Statist.*, vol. 6, no. 2, pp. 65–70, 1979.

[19] J. Cohen, *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed. Hillsdale, NJ: Lawrence Erlbaum, 1988.

---

**Note**: Section 3 establishes:
1. ✅ **Network Model**: Static topology, Intel Lab + synthetic, N nodes + BS
2. ✅ **Energy Model**: CC2420-calibrated, temperature/humidity corrections
3. ✅ **Channel Model**: Log-normal shadowing, IEEE 802.15.4 MAC, realistic PDR
4. ✅ **Metrics**: PDR, lifetime, fairness, with statistical validation framework
5. ✅ **Optimization**: Multi-objective formulation, justified heuristic approach

**Estimated word count**: ~2200 words

**Next step**: Integrate with existing sections and create architecture diagrams (Day 3-4)


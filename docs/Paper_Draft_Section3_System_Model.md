# Section 3. System Model and Problem Formulation

## 3.1 Network Model
We consider a static WSN deployed over a two-dimensional region \(\mathcal{A}\subset\mathbb{R}^2\) with \(N\) battery-powered sensor nodes and a single base station (BS). Node locations \(\mathbf{p}_i=(x_i,y_i)\) follow either the Intel Lab geometry (54 motes with surveyed coordinates) or synthetic layouts (uniform random, corridor chains). Each node starts with initial energy \(E_0\in[1.5,2.5]\,\mathrm{J}\) according to hardware calibration, and maintains residual energy \(E_i(t)\). Data collection proceeds in rounds; at round \(t\), every node senses a packet of \(k\) bits (default 1024 bytes) that must be delivered to the BS via multi-hop clustering.

Cluster heads (CHs) are elected per round. Let \(\mathcal{C}(t)\) denote the CH set and \(\mathcal{M}_c(t)\) the members of cluster \(c\). The skeleton backbone selects a subset of CHs to serve as relay hubs. The objective is to maximize network lifetime (time to first node death) and end-to-end delivery reliability while respecting energy budgets.

## 3.2 Energy Consumption Model
Transmission and reception energy follow the improved CC2420-based model:
\[
E_{tx}(k,d,P_{tx}) = k E_{elec}^{tx} + k \frac{P_{tx}}{\eta_{amp}} \times
\begin{cases}
 d^{\gamma_1}, & d\leq d_0 \
 d^{\gamma_2}, & d > d_0
\end{cases}
\]
where \(E_{elec}^{tx}=208.8\,\mathrm{nJ/bit}\), \(\eta_{amp}=0.5\), \(\gamma_1=2\), \(\gamma_2=4\), and \(d_0=87\,\mathrm{m}\) for TelosB-class radios. Reception energy is \(E_{rx}(k) = k E_{elec}^{rx}\) with \(E_{elec}^{rx}=225.6\,\mathrm{nJ/bit}\). Processing, idle, and sleep energies are included via hardware-specific coefficients. Environmental factors modulate these costs: temperature deviations and humidity ratios scale energy by \(1+\alpha_T|T-25|\) and \(1+\alpha_H H\) respectively (\(\alpha_T=0.02\), \(\alpha_H=0.01\)).

## 3.3 Channel and Link Model
The wireless channel obeys the log-normal shadowing model:
\[
PL(d) = PL(d_0) + 10 n \log_{10}\frac{d}{d_0} + X_\sigma,
\]
where \(n\) and \(\sigma\) depend on the environment class \(\mathcal{E}\in\{\text{indoor-office},\text{indoor-residential},\text{outdoor-open},\ldots\}\). For Intel Lab replays, environment classes derive from humidity/temperature clusters; synthetic topologies adopt scenario-specific parameters. Received power is \(P_{rx}=P_{tx}-PL(d)\). Link quality metrics include RSSI, IEEE 802.15.4 LQI, and empirical PDR estimated via node state histories. CAS operates on normalized energy, link, distance, radius, and fairness features calculated each round.

## 3.4 Reliability and Fairness Metrics
End-to-end PDR \(\mathrm{PDR}_{e2e}\) measures packets delivered to BS divided by source packets emitted within a round. Hop-level PDR captures intra-cluster reliability. Fairness is quantified with Jain's index over CH usage counts:
\[
J = \frac{(\sum_{i\in\mathcal{C}(t)} u_i)^2}{|
\mathcal{C}(t)| \sum_{i\in\mathcal{C}(t)} u_i^2},
\]
where \(u_i\) is the cumulative rounds node \(i\) serves as CH. Safety fallback triggers when consecutive rounds fall below a reliability threshold \(\theta\), enabling redundant uplinks or power boosts.

## 3.5 Optimization Goals
We formulate a multi-objective problem balancing reliability and efficiency:
\[
\max_{\{\mathcal{C}(t), \text{routing}\}} \quad \lambda_1 \mathrm{PDR}_{e2e} - \lambda_2 \frac{E_{tot}}{N} + \lambda_3 J,
\]
subject to residual energy \(E_i(t)\geq 0\), duty-cycle constraints, and per-round transmission budgets. Here \(E_{tot}\) is total consumed energy, and \(\lambda_1,\lambda_2,\lambda_3\) weigh reliability, efficiency, and fairness. AERIS implements this implicitly: skeleton selection minimizes path stretch, CAS chooses intra-cluster forwarding mode, gateway selection reduces long-haul losses, and safety/fairness modules enforce constraints. Analytical solutions are intractable; thus AERIS employs deterministic heuristics validated through empirical evaluation with statistical significance tests.

# Section 3: System Model and Design Rationale

---

## 3. System Model and Design Rationale

This section defines the network, energy, and channel models used in the evaluation pipeline, then explains how these assumptions map to AERIS design choices.

### 3.1 Network Model

We consider a static wireless sensor network with the following assumptions:

1. **Static nodes**: sensor nodes remain fixed after deployment.
2. **Homogeneous hardware**: nodes share the same transceiver/energy model parameters.
3. **Single base station**: the BS has unconstrained energy and acts as data sink.
4. **Periodic sensing**: each alive node generates one packet per round.
5. **Symmetric connectivity approximation**: if node i can communicate with node j, reverse communication is assumed feasible under the same channel state.

The network is represented as graph G = (V, E), where V is node set and E is communication edges under current channel conditions.

### 3.2 Energy Model

The simulator follows a first-order radio model with transmission, reception, and aggregation costs.

Transmission energy:

```
E_tx(L, d) = E_elec * L + E_amp(d) * L
```

Reception energy:

```
E_rx(L) = E_elec * L
```

Aggregation energy:

```
E_agg(L) = E_DA * L
```

Where:
- L is packet size in bits.
- E_elec is electronics cost per bit.
- E_amp(d) is distance-dependent amplifier cost.
- E_DA is aggregation cost per bit.

For publication-tier experiments in this manuscript, packet size is aligned to the benchmark framework default (`packet_size = 1024` bytes).

### 3.3 Channel Model

The channel layer uses environment-parameterized propagation settings. Different environments are represented by different path loss and shadowing characteristics, allowing controlled comparison across:

- indoor_office
- indoor_factory
- outdoor_urban
- outdoor_suburban

This model is designed for **comparative protocol evaluation** under heterogeneous channel quality, not for hardware-specific PHY calibration.

### 3.4 Design Rationale (Evidence-Scoped)

AERIS is designed to address reliability degradation under harsh channel conditions while keeping runtime logic lightweight.

The design is modular:

1. **Context-Adaptive Switching (CAS)** for local forwarding-mode selection.
2. **Gateway coordination** for robust uplink reinforcement.
3. **Skeleton backbone** for structured long-path stabilization.
4. **Safety fallback** for failure containment.

Rationale in this manuscript is scoped to publication-tier protocol evidence (multi-environment comparison, ablation, scalability, and NS-3 trend-level alignment). We intentionally avoid unsupported dataset-level predictive claims in this section.

### 3.5 Principle-to-Mechanism Mapping

| Principle | Operational Goal | AERIS Mechanism |
|-----------|------------------|-----------------|
| Reliability under harsh channels | Maintain delivery under weak links | Gateway-assisted uplink + safety fallback |
| Adaptive local forwarding | Match routing mode to local context | CAS mode selection (Direct/Chain/TwoHop) |
| Structured long-path routing | Avoid unstable long random paths | Skeleton/backbone routing when triggered |
| Practical deployability | Keep runtime lightweight | Rule-based scoring without model training |

These principles and mechanisms are evaluated empirically in Section 6 with explicit scope labels and statistical tests.

---

## References (Section 3 additions)

[12] Texas Instruments, "CC2420 2.4 GHz IEEE 802.15.4 RF Transceiver," Datasheet, 2007.

[13] S. Madden, "Intel Lab Data," MIT CSAIL, 2004.

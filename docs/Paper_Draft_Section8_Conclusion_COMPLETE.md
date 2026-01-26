# Section 8: Conclusion (完整版)

**字数**: ~550词  
**状态**: 完整初稿，待审阅  
**版本**: 1.0

---

## 8. Conclusion

This paper introduced **AERIS** (Adaptive Environment-aware Routing for IoT Sensors), a novel routing protocol designed to bridge the persistent simulation-to-reality gap in wireless sensor networks. By integrating environment-aware optimization, IEEE 802.15.4-consistent channel modeling, and lightweight online adaptation, AERIS achieves the **adaptivity** of machine learning approaches without their **computational burden**, while maintaining the **deployment simplicity** of classical deterministic algorithms.

### Key Contributions

We summarize our main contributions:

**Protocol Innovation**: AERIS employs a three-layer architecture—Context-Aware Selector (CAS), skeleton routing, and gateway coordination—that decouples transmission mode selection, backbone formation, and reliability enhancement. This modular design facilitates independent optimization and troubleshooting while achieving synergistic performance gains.

**Environment-Aware Mechanism**: Unlike prior work that uses 1–2 environmental variables with hand-crafted mappings, AERIS extracts 30+ dimensional features from sensor data and employs unsupervised K-means clustering to automatically discover 8 data-driven environment patterns. This rich representation enables fine-grained adaptation to varying propagation conditions.

**Lightweight Adaptivity**: AERIS integrates simplified Q-learning with discrete state-action tables (2KB memory, O(1) decision complexity) that converge within 30–50 rounds. This approach achieves online weight adaptation without the training overhead, computational cost, or memory footprint of deep reinforcement learning methods (which require 50–100× more resources).

**Realistic Evaluation Framework**: We established a reproducible experimental pipeline based on the Intel Berkeley Research Lab dataset (2.22M records, 54 nodes, 36 days) with IEEE 802.15.4-consistent channel and MAC models. All code, data processing scripts, and configuration files are released as open source to facilitate community validation and extension.

### Experimental Findings

Comprehensive experiments involving 200 independent runs per configuration demonstrate that AERIS achieves:

- **Energy efficiency**: 7.9% reduction in total energy consumption versus PEGASIS (from 11.33J to 10.43J over 200 rounds), achieving 2,396 packets/Joule energy efficiency.

- **Reliability enhancement**: 43.1 percentage point improvement in end-to-end packet delivery ratio (from 42.5% to 85.6%), with statistical significance confirmed via Welch's t-tests with Holm–Bonferroni correction (p < 0.001, Cohen's d = 1.89).

- **Extended network lifetime**: 100% node survival through 500 rounds under experimental conditions, with lowest energy consumption projecting longer operational lifetime in extended deployments.

- **Rapid convergence**: Weight adaptation stabilizes within 30–50 rounds (15–25 minutes at 0.5-minute round intervals), enabling near-immediate deployment without prolonged training phases.

Ablation studies quantify component contributions: gateway coordination provides the largest PDR improvement (18 percentage points), fairness constraints reduce energy variance by 46% (from σ = 0.28J to 0.15J), and environment-adaptive power control contributes 3.2% energy savings.

### Practical Impact

AERIS operates within resource constraints of commercial sensor motes (8KB RAM, 48KB Flash), requires no offline training, and includes safety fallback mechanisms and fairness policies to prevent cluster head overuse. These characteristics make AERIS **immediately deployable** on platforms such as TelosB, MICAz, and Zolertia Z1, addressing a critical gap between laboratory protocols and field-ready implementations.

By demonstrating that **deterministic algorithms with lightweight adaptation** can match or exceed the performance of heavyweight machine learning methods, AERIS challenges the prevailing assumption that adaptivity necessitates deep neural networks. We hope this work inspires further research into **resource-aware intelligence**—algorithms that balance adaptivity with deployability constraints.

### Future Directions

Several promising directions warrant investigation:

**Security enhancements**: Integrating lightweight cryptographic authentication (e.g., TinySec) and trust-based routing metrics would enhance resilience against adversarial attacks (selective forwarding, sinkhole attacks, Sybil attacks).

**Edge-assisted optimization**: Offloading computationally intensive tasks (K-means clustering, PSO backbone optimization) to edge servers via 5G or LoRaWAN backhaul could enable more sophisticated algorithms while preserving node energy budgets.

**Multi-sink architectures**: Extending AERIS to support multiple base stations with load balancing and fault tolerance mechanisms would improve scalability for large-area deployments (smart cities, precision agriculture).

**Real-world validation**: Hardware testbed experiments on TelosB/Z1 motes in industrial, agricultural, and urban environments remain critical for confirming simulation findings and uncovering deployment-specific challenges.

**Broader protocol support**: Porting AERIS principles to alternative WSN standards (LoRaWAN for long-range, Bluetooth Low Energy for connection-oriented communication) would expand applicability.

### Closing Remarks

As wireless sensor networks transition from academic research to industrial-scale deployments, protocols must deliver **proven reliability** under realistic conditions while respecting the **severe resource constraints** of battery-powered nodes. AERIS represents a step toward this goal by combining rigorous experimental validation (200 runs, Welch's t-tests, open-source release), realistic system modeling (IEEE 802.15.4 MAC, Intel Lab data), and practical design choices (8KB RAM, sub-millisecond decisions). We believe the principles underlying AERIS—data-driven environment discovery, lightweight online adaptation, and modular architecture—offer a reusable blueprint for next-generation IoT routing protocols that balance **intelligence** with **efficiency**.

---

**Note**: Conclusion provides:
1. ✅ **Contribution summary** (3 technical + evaluation + practical)
2. ✅ **Key findings** (quantitative results with statistics)
3. ✅ **Practical impact** (deployment feasibility)
4. ✅ **Future work** (5 concrete directions)
5. ✅ **Closing message** (balanced perspective)

**Estimated word count**: ~550 words

---

## 🎉 **Status Update: All Critical Sections Complete!**

### Completed Sections:
1. ✅ **Section 1: Introduction** (~2600 words)
2. ✅ **Section 2: Related Work** (~3200 words)
3. ✅ **Section 7: Discussion** (~2300 words)
4. ✅ **Section 8: Conclusion** (~550 words)

**Total new content**: ~8650 words

### Existing Sections (from previous work):
- Section 3: System Model (~1500 words)
- Section 4: Algorithm Design (~2500 words)
- Section 5: Experimental Setup (~1800 words)
- Section 6: Results Analysis (~2400 words)

**Total paper length**: ~17,000 words (exceeds typical 8,000-10,000 target, will need condensing)

### Next Priority Actions:
1. **Integration**: Merge new sections with existing drafts
2. **Reference completion**: Expand to 50-60 citations
3. **Figure creation**: Architecture diagram, flowcharts
4. **Language polish**: Grammarly check
5. **Format conversion**: Convert to IEEE/MDPI Sensors template


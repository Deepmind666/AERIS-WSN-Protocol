# Section 7: Conclusions and Future Work

This paper presented AERIS (Environment-Adaptive Skeleton Routing), a lightweight, deployment-oriented WSN routing protocol that integrates calibrated energy modeling, log-normal shadowing channels aligned with Intel Lab geometry, and a hybrid coordination layer (skeleton backbone, gateway relays, and CAS mode switching) for robust performance under realistic interference and environmental variability.

Conclusions and key findings:
- AERIS consistently reduces total energy consumption against strong classical baselines (LEACH, PEGASIS, HEED), achieving a 7.9% reduction vs. PEGASIS while maintaining 100% node survival in 500 rounds. See Figures 1–2 for energy and PDR panels and Figure 1 for the main energy comparison.
- Across repeated independent runs, improvements are statistically significant under Welch’s t-test with Holm–Bonferroni adjustment; bootstrap 95% confidence intervals and effect sizes (Cohen’s d) confirm both statistical and practical significance. Figure 3 summarizes adjusted p-values and effect sizes; Figure 7 provides Gardner–Altman estimation for end-to-end PDR.
- Ablation results (Figures 4–5) indicate that the coordination between the fuzzy-logic CH selection and PSO-based routing path optimization is the dominant source of gains, complemented by the safety/fairness mechanisms that stabilize performance when conditions degrade.
- The methodology and visualizations are fully aligned across Sections 4–6: Figure 6 contains the method flowchart; Figures 1–5 report core results; Figure 7 visualizes estimation statistics for PDR. All additional diagnostics are deferred to Supplementary Figures S1–S4 to keep the main narrative concise.

Limitations:
- Evaluation primarily focuses on static deployments and IEEE 802.15.4-class radios; while we replay realistic environmental traces (Intel Lab) and probe synthetic layouts (uniform and corridor), broader heterogeneous environments (industrial metal-rich, outdoor with high mobility) remain to be explored.
- AERIS’s coordination is deterministic and lightweight by design; while this brings predictability and low overhead, it may be slower than learning-based policies in rapidly time-varying or highly non-stationary interference regimes.
- The present study uses a single-sink topology and single radio; multi-sink, multi-radio, and duty-cycled MACs (with different contention behaviors) deserve systematic treatment.

Future work:
- Dynamic environments: integrate adaptive horizon selection and proactive interference sensing to accelerate responses to abrupt changes while preserving energy efficiency.
- Security and resilience: harden gateway screening and emergency fallbacks against adversarial jamming and selective forwarding; add anomaly detection at the skeleton layer.
- Edge-assisted intelligence: explore offline RL for parameter sweeping/tuning and online deterministic execution for predictability; develop safe-switch policies with verifiable performance bounds.
- Broader validation: extend to hardware-in-the-loop testbeds, multi-sink topologies, mobility, and mixed PHY/MAC stacks; expand dataset coverage beyond Intel Lab traces.
- Engineering usability: finalize a reproducible package with containerized pipelines, CI-validated figures, and parameterized templates to regenerate Figures 1–7 and Supplementary S1–S4 from raw logs.

Reproducibility statement:
All analysis scripts, simulation harnesses, and plotting code used for Figures 1–7 are available in the artifact repository; statistical evaluation follows a single pipeline (Welch’s t-test, Holm–Bonferroni adjustment, bootstrap 95% CI, effect sizes) consistently across Sections 5–6. This ensures that reported conclusions are transparent, auditable, and fully reproducible.

## MDPI Compliance Statements

### Author Contributions
Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Resources, Data curation, Writing—original draft preparation, Writing—review and editing, Visualization, Supervision, Project administration, Funding acquisition: [insert full author names per role]. All authors have read and agreed to the published version of the manuscript.

### Funding
This work was supported by [funding agency and grant numbers, if any]. If no external funding applies, replace this sentence with: "This research received no external funding."

### Institutional Review Board Statement
Not applicable. The study does not involve human participants or animals.

### Informed Consent Statement
Not applicable.

### Data Availability Statement
Code, simulation logs, and plotting scripts to reproduce Figures 1–7 and Supplementary S1–S4 are available in the artifact repository accompanying this paper; processed datasets are derived from the publicly available Intel Berkeley Research Lab traces. Additional configuration files and parameters are documented in Sections 4–6 and the reproducibility appendix.

### Acknowledgments
We thank [collaborators, institutions, or facilities] for their support and constructive feedback.

### Conflicts of Interest
The authors declare no conflict of interest.

### Sample Availability
Not applicable.
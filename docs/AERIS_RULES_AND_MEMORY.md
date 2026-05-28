# AERIS Rules and Memory

## Stable Paper Position

AERIS is a reliability-first, rule-based, auditable routing design for wireless
sensor networks under heterogeneous and unstable channels.

Do not frame it as a universal routing winner. The correct story is bounded:

- strong against classical WSN baselines in selected harsh regimes;
- especially useful when long CH-to-BS uplinks are fragile;
- bounded by stronger collection-tree/RPL-style baselines;
- reliability is bought with concentrated relay work and shorter lifetime.

## Evidence Roles

- Classical NS-3 audit: primary classical-baseline fairness anchor.
- Expanded seven-protocol boundary sweep: deployment-boundary evidence.
- NS-3 ablation: mechanism attribution.
- Strict-physics Python layer: stress evidence only.
- Mechanism study: explains Gateway, CAS, Skeleton, FND, lifetime, and energy.

## Mechanism Attribution

- Gateway-assisted uplink is the main gain carrier in the current evidence.
- CAS is conditionally useful when local detours remain viable.
- CH scoring is near-neutral in the current publication configuration.
- Skeleton is a reserve fallback and mostly dormant in the audited setting.

## Reviewer Risks

- Fixed coefficients need sensitivity analysis or explicit heuristic framing.
- Simplified CTP/RPL-MRHOF baselines are not full standards stacks.
- Idealized MAC/collision behavior limits external validity.
- TEEN PDR-expected denominator must be discussed with energy/lifetime.
- Pooled fixed-100-node table values must not be generalized to all scales.

## User Writing Preferences

The user expects:

- strict reviewer-style feedback;
- no flattery;
- compact and readable figures;
- no line overlap or unexplained heatmap clutter;
- nearby figure/table interpretation;
- exact template formatting;
- synchronized source, PDF, data, zip, and scripts before pushing.

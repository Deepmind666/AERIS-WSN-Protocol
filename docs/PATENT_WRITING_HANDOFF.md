# Patent Writing Handoff

This repository is prepared so the AERIS patent can be drafted on another
computer.

## Style Target

Use the user's local patent-writing skill. Match the language quality, structure,
and technical precision of the user's MoE patent work.

The patent draft should not read like a conference paper. It should read like a
technical patent document:

- technical field;
- background problem;
- invention purpose;
- technical solution;
- method/system/device/storage-medium claims;
- beneficial effects;
- embodiment steps;
- module descriptions;
- figure descriptions.

## AERIS Invention Core

AERIS is an environment-aware reliable routing method for wireless sensor
networks under heterogeneous channels. The patent angle should center on a
coordinated routing-control method that combines:

1. context-adaptive intra-cluster forwarding;
2. Gateway-assisted uplink reinforcement from cluster head to base station;
3. energy/link/utilization-aware scoring for role and path selection;
4. reserve fallback routing through a sparse Skeleton backbone;
5. deterministic fallback order for unreliable links.

## Suggested Claim Objects

Draft claims around:

- a routing method for wireless sensor networks;
- a routing control apparatus or network node;
- a wireless sensor network system;
- a computer-readable storage medium or program product, if appropriate.

## Suggested Independent Method Claim Skeleton

1. Obtain node state information including residual energy, link-quality
   estimate, node position/distance, neighborhood density, and forwarding load.
2. Select cluster heads or routing roles according to a composite score.
3. Assign member nodes to cluster heads according to link/energy/distance scores.
4. For each cluster, select an intra-cluster forwarding mode from direct, chain,
   and two-hop forwarding according to context features.
5. Select a Gateway relay for cluster-head uplink when direct uplink reliability
   is insufficient.
6. If direct and Gateway-assisted uplinks fail to satisfy the condition, use a
   reserve Skeleton fallback path.
7. Update routing state across rounds according to residual energy and link
   observations.

## Technical Effects to Emphasize

- Reduces fragile long-range uplink failure in harsh heterogeneous channels.
- Adapts local forwarding mode to link quality, energy, and load.
- Provides deterministic fallback order, which makes the protocol auditable.
- Improves delivery reliability in selected harsh-channel regimes.
- Trades reliability for concentrated relay burden; do not claim it always
  maximizes lifetime.

## Important Boundaries

- Do not copy paper-style phrases such as "not globally dominant" directly into
  claims. Use them only to avoid overclaiming.
- Do not state that AERIS is optimal.
- Do not state that Skeleton is the main gain source. It is a reserve fallback.
- Do not rely on exact experimental percentages in patent claims unless the user
  asks for an evidence section.
- Keep claims broad enough to cover variants of the scoring weights and
  thresholds; avoid locking the invention to one fixed coefficient set unless
  needed.

## Files to Read First

```text
paper/LCN26_AERIS_overleaf/aeris_lcn2026.pdf
paper/LCN26_AERIS_overleaf/aeris_lcn2026.tex
src/aeris_protocol.py
src/enhanced_aeris_protocol.py
src/cas_selector.py
src/gateway_selector.py
src/skeleton_selector.py
ns3_validation/aeris-validation-standalone.cc
agent_memory/AERIS_project_memory.md
```

## Existing Patent Seed

The folder below contains an earlier patent seed. Treat it as reference material,
not as final language:

```text
patent_seed_20260306/
```

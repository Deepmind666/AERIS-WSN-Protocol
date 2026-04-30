# 2026-04-30 LCN26 Protocol Technical Notes

Purpose:
- short explanation of AERIS, the classical WSN baselines, and the added CTP/RPL-MRHOF baselines
- use for paper discussion, slides, and reviewer-response preparation

## AERIS

AERIS is a rule-based hierarchical routing design. It is not a learned policy and not a full LLN stack. Its value is that the control path is auditable:
- `CAS` selects sensor-to-cluster-head mode among direct, chain, and two-hop.
- `Gateway` reinforces the cluster-head-to-base-station uplink when direct uplink quality is weak.
- `Skeleton` is a sparse fallback backbone, but the corrected mechanism audit shows it is dormant in the current publication configuration.

Current mechanism interpretation:
- Gateway is the dominant active reliability mechanism.
- CAS is environment dependent.
- Skeleton is reserve logic, not the source of the measured gain.
- The cost is concentrated relay burden and earlier first-node death.

## Classical WSN Baselines

`LEACH`:
- randomized/rotating cluster-head protocol
- low overhead and simple
- weak when cluster-head placement or CH-to-BS links are poor

`PEGASIS`:
- chain-based aggregation
- strong lifetime behavior in benign cells
- can incur very long paths and serialization
- remains a strong office/longevity baseline

`HEED`:
- residual-energy and communication-cost-aware cluster-head election
- more stable than pure random clustering
- still exposes the uplink when harsh channels dominate

`TEEN`:
- event/threshold-triggered reporting
- energy efficient when events are sparse
- PDR interpretation is sensitive to the expected-transmission denominator, so energy, lifetime, FND, and hops must be shown alongside PDR

## Added Collection / LLN Baselines

`CTP`:
- collection-tree style routing toward the sink
- chooses forwarding paths using link/path quality and progress constraints
- very strong in benign office-like settings where stable collection paths exist

`RPL-MRHOF`:
- RPL-style low-power and lossy-network routing with a minimum-rank/path-cost objective
- chooses parents by accumulated path cost
- strong in factory and urban cells in the expanded NS-3 audit

`ORPL / ORW`:
- opportunistic RPL/anycast-style families
- use candidate forwarder sets and local link diversity
- not implemented in the current evidence package
- important future baselines if the paper is expanded toward full LLN-stack comparison

## Current Paper Boundary

Best defensible statement:
- AERIS is strong against classical WSN baselines:
  - rank-1 in `21/28` dual-sweep cells when only classical baselines are considered
  - top-2 in `28/28`
- AERIS is not globally best after adding CTP/RPL-MRHOF:
  - rank-1 in `7/28`
  - top-2 in `8/28`
  - strongest mainly in outdoor suburban cells
  - CTP leads office
  - RPL-MRHOF leads most factory and urban cells

Interpretation:
- AERIS solves a classical-baseline weakness by reinforcing fragile uplinks.
- CTP/RPL-MRHOF solve part of the same weakness through richer collection-tree parent choice.
- The strongest final paper is therefore a boundary-mapping engineering paper, not a universal-winner paper.

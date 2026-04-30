# 2026-04-30 LCN26 Evidence Route Assessment

## Question

Should the paper keep the current five-protocol corrected NS-3 evidence route, or should it promote the new seven-protocol expanded NS-3 sweep to the main canonical result?

## Current Five-Protocol Route

Current draft anchor:
- `ns3_validation/results/lcn26_ns3_audit_20260420_012811/summary/`
- protocols: `AERIS`, `LEACH`, `PEGASIS`, `HEED`, `TEEN`
- environments: `indoor_office`, `indoor_factory`, `outdoor_suburban`, `outdoor_urban`
- nodes: `100`, `500`, `1000`
- replicates: `n=30` per environment-node-protocol cell

Main story:
- `PEGASIS` is strongest in benign `indoor_office`.
- `AERIS` is strongest in `indoor_factory`, `outdoor_suburban`, and `outdoor_urban`.
- The mechanism matrix supports a narrow attribution: Gateway support carries the reliability gain, CAS is environment dependent, and Skeleton is dormant in the audited publication configuration.

Assessment:
- Best short-term route for the current LCN26 submission.
- It is internally consistent with the current abstract, figures, captions, discussion, and limitations.
- It already avoids universal-superiority wording by treating office as a negative control.
- The main residual risk is that the canonical baseline family is classical rather than exhaustive.

## New Seven-Protocol Expanded Route

New evidence:
- `ns3_validation/results/lcn26_ns3_expanded_20260430_173108/summary/`
- `ns3_validation/results/lcn26_ns3_dual_combined_20260430_191527_191528/summary/`
- protocols: `AERIS`, `LEACH`, `HEED`, `PEGASIS`, `TEEN`, `RPL-MRHOF`, `CTP`

Observed winner pattern in the dual-machine sweep:
- `indoor_office`: `CTP`
- `indoor_factory`: mostly `RPL-MRHOF`, except `50` nodes where `AERIS` is slightly ahead
- `outdoor_suburban`: mostly `AERIS`, with a near-tie or `RPL-MRHOF` edge at `1000` nodes
- `outdoor_urban`: `RPL-MRHOF`

Assessment:
- Stronger as long-term scientific evidence because it tests richer low-power routing baselines.
- Riskier as the main route for the current LCN26 manuscript because it changes the paper's core claim.
- If promoted to canonical evidence, the paper must be rewritten around a narrower boundary claim: AERIS is not the harsh-environment winner in general; it is strongest mainly in the suburban regime and competitive in selected cells, while RPL-MRHOF/CTP become the stronger deployment baselines in several environments.

## Recommendation

For the current LCN26 submission, keep the five-protocol corrected NS-3 route as the main canonical evidence and treat the seven-protocol sweep as supplementary/follow-up evidence, not as a replacement.

Reason:
- The current paper is already coherent as a scoped engineering contribution: rule-based AERIS improves reliability against classical WSN baselines in harsher channels while exposing a real lifetime cost.
- Promoting the seven-protocol sweep now would require reworking the abstract, contribution framing, result figures, discussion, and conclusion. That route is more rigorous but not better for a near-term conference submission unless the target is changed to a negative-results or boundary-mapping paper.

Practical next step:
- Submit/polish the current five-protocol LCN26 package.
- Preserve the seven-protocol sweep as evidence for the next revision cycle or a stronger journal version where CTP/RPL-family baselines become part of the primary design boundary.

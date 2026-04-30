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

## Updated Recommendation After Rewrite Approval

The user accepted a full rewrite if it improves the final paper. Under that assumption, the best route is to promote the seven-protocol sweep into the main evidence story, but not as an `AERIS is globally best` claim.

New main story:
- Against classical WSN baselines (`LEACH`, `PEGASIS`, `HEED`, `TEEN`), AERIS is strong:
  - rank-1 in `21/28` dual-sweep environment-node cells
  - top-2 in `28/28`
- Against the expanded seven-protocol family, AERIS has a narrower boundary:
  - rank-1 in `7/28`
  - top-2 in `8/28`
  - strongest mainly in `outdoor_suburban`
  - beaten by `CTP` in `indoor_office`
  - beaten by `RPL-MRHOF` in most `indoor_factory` and `outdoor_urban` cells

Reason:
- This route is stronger scientifically because it directly addresses the reviewer attack that the comparison set lacks credible LLN/collection baselines.
- The paper becomes a boundary-mapping engineering paper rather than a simple winner paper.
- The AERIS claim is narrower but more defensible: simple Gateway-assisted rule control beats classical WSN baselines in many harsh cells, but it is not a universal replacement for collection-tree or RPL-style routing.

Practical next step:
- Rewrite the abstract, results, discussion, and conclusion around the seven-protocol boundary.
- Add a new LCN-style boundary figure:
  - `scripts/build_lcn26_expanded_boundary.py`
  - `_LCN26_AERIS/generated/fig_lcn26_ns3_expanded_boundary.pdf`
- Keep the corrected five-protocol evidence as the classical-baseline repair anchor and provenance layer.

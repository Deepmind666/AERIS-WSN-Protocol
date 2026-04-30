# 2026-04-20 Review Comment Mapping

Source review:
- `_LCN26_AERIS/review_comment.pdf`

Current draft:
- `_LCN26_AERIS/aeris_lcn2026.tex`

Status tags:
- `Done`: main attack surface materially reduced in the current draft
- `Partial`: improved, but not fully closed
- `Open`: still a real gap

## Critical weaknesses

### 1. Cross-platform closure insufficient
Status: `Done`

What changed:
- The main conference draft no longer relies on `AERIS vs LEACH` only.
- Canonical NS-3 now uses the corrected five-protocol rerun at:
  - `ns3_validation/results/lcn26_ns3_audit_20260420_012811/summary/`
- Main text now explicitly states the full environment-scoped five-protocol ranking.

Residual risk:
- Canonical NS-3 covers `100, 500, 1000`, not the old `50..1000` sweep.
- This is acceptable for a focused repair, but still smaller than a full independent matrix.

### 2. Baseline fairness
Status: `Partial`

What changed:
- The paper now separates:
  - canonical NS-3 evidence
  - adapted strict-physics evidence
- The draft explicitly says the strict matrix is not the sole fairness anchor.

Residual risk:
- The Python strict layer still depends on adapted baselines.
- That concern is mitigated, not eliminated.

### 3. Innovation depth limited / engineering combination
Status: `Open`

What changed:
- The draft now frames AERIS as a scoped protocol-engineering paper rather than a universal or theory-first protocol.

Residual risk:
- This is a method-class issue, not a wording issue.
- We can only position it honestly; we cannot convert a heuristic engineering paper into a mechanism-theory paper by revision alone.

### 4. Reliability-lifetime trade-off not resolved
Status: `Partial`

What changed:
- The trade-off is now framed as a deployment boundary, not hidden.
- The corrected mechanism panel explains where the cost materializes:
  - gateway bottleneck
  - very early first-node death in many `500/1000` harsh cells

Residual risk:
- The paper still does not propose a new mitigation mechanism.
- The trade-off is better explained, not solved.

### 5. Mechanism explanation too weak
Status: `Done`

What changed:
- Added corrected 400-replicate mechanism matrix.
- Current text now states:
  - `Gateway` is the dominant active mechanism
  - `CAS` is environment dependent
  - `Skeleton` is inactive in the audited publication configuration
- Supporting source:
  - `results/lcn26_targeted_20260420/mechanism_grid_fat/mechanism_summary.csv`

Residual risk:
- Still not a closed-form causal theory.
- But it is no longer only a shallow ablation report.

## Technical / clarity issues

### MAC realism gap
Status: `Partial`

What changed:
- The limitations section now explicitly states that the strict-physics layer omits full 802.15.4/TSCH/CSMA-CA details such as ACK loops, backoff dynamics, capture, and channel hopping.

Residual risk:
- This is still a real external-validity limit.

### Conditional independence concern
Status: `Done`

What changed:
- The current draft explicitly downgrades the success-path reasoning to an engineering approximation rather than a formal proof.

### Algorithmic details underspecified
Status: `Partial`

What changed:
- The draft now gives short clarifications for:
  - feature normalization
  - centrality as a local geometry surrogate
  - link quality as channel-estimated success probability
  - Skeleton refresh timing

Residual risk:
- Still concise rather than exhaustive.
- Could be expanded more if space allows.

### Flowchart vs strict no-retransmission conflict
Status: `Partial`

What changed:
- Text is already aligned to strict mode.
- Temporary workflow source has been switched to:
  - `_LCN26_AERIS/AERIS流程图.pdf`
  - copied into
  - `_LCN26_AERIS/generated/fig0_aeris_workflow_temp_20260420.pdf`

Residual risk:
- The flowchart itself still depends on the manual fix to fully align with text.

### TEEN bias from `PDR_expected`
Status: `Partial`

What changed:
- Evaluation design now explicitly says the paper also reports total energy, lifetime, FND, and hops, so the draft is not asking the reader to rely on `PDR_expected` alone.

Residual risk:
- We still do not add a TEEN-specific secondary metric.
- So the bias concern is reduced, not fully closed.

### Missing related standards / protocols
Status: `Partial`

What changed:
- Related work / limitations now explicitly mention:
  - `CTP`
  - `RPL/ORPL/ORW`
  - coding-based reliability
  - channel-hopping systems

Residual risk:
- They are acknowledged, not experimentally compared.

### Reviewer-side reproducibility not available
Status: `Open`

What changed:
- Provenance is much cleaner internally.

Residual risk:
- Reviewers still do not have public code during review.
- This is not fully solvable within the current submission cycle.

## Bottom line

Current judgment:
- Main review attack surfaces are now substantially reduced.
- The draft is materially stronger than the earlier version.
- But it is not true that every issue in `review_comment.pdf` has been completely solved.

Most improved:
- cross-platform closure
- fairness separation
- mechanism explanation
- trade-off explanation

Still fundamentally limited:
- method novelty ceiling
- adapted-baseline external validity
- lack of direct comparison to standards like RPL/CTP
- absence of reviewer-time code release

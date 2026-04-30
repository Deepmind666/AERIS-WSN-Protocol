---
name: aeris-lcn-revision
description: Use for LCN 2026 revision work under C:\AERIS-WSN-Protocol when updating the AERIS conference draft, auditing corrected rerun evidence, regenerating non-workflow figures, or tightening claims against review_comment.pdf. Trigger for requests about AERIS paper polishing, LCN resubmission, review_comment mapping, corrected NS-3/mechanism results, or final PDF rebuilds.
---

# AERIS LCN Revision

Use this skill for the AERIS LCN 2026 resubmission workflow. The paper is an engineering protocol paper, not a theory paper. Default to narrowing claims, improving evidence closure, and preserving provenance.

## Core positioning

- Present AERIS as a `reliability-first`, `rule-based`, `auditable` routing design.
- Do not present AERIS as a universal-best protocol.
- Use environment-scoped conclusions:
  - `PEGASIS` is strongest in benign `indoor_office`.
  - `AERIS` is strongest in harsher `indoor_factory`, `outdoor_suburban`, and `outdoor_urban`.
- Do not overstate `Skeleton`; the corrected mechanism matrix shows it is inactive in the current audited publication configuration.

## Evidence hierarchy

Always keep these layers separate in writing and captions:

1. `Canonical NS-3 evidence`
   - Main fairness/ranking anchor
   - Use only:
     - `ns3_validation/results/lcn26_ns3_audit_20260420_012811/summary/ns3_focused_descriptive.csv`
     - `ns3_validation/results/lcn26_ns3_audit_20260420_012811/summary/ns3_focused_significance.csv`
   - Never use the invalid rerun:
     - `ns3_validation/results/lcn26_ns3_audit_20260420_010122`

2. `Strict-physics Python evidence`
   - Stress-test layer only
   - Use:
     - `results/mega_experiments/scalability_4env_v50rigor_20260222_descriptive.csv`
   - Explicitly say this is not the sole fairness anchor because the strict layer includes adapted baselines.

3. `Corrected AERIS mechanism evidence`
   - Use:
     - `results/lcn26_targeted_20260420/mechanism_grid_fat/mechanism_summary.csv`
     - `results/lcn26_targeted_20260420/mechanism_grid_fat/mechanism_raw_merged.json`
   - Current active interpretation:
     - `Gateway` is the dominant active reliability mechanism
     - `CAS` is environment dependent
     - `Skeleton` is inactive in the audited publication configuration

4. `Frozen 100-node publication block`
   - Cross-protocol summary only
   - Use:
     - `results/mega_experiments/energy_lifetime_stats.csv`
     - `results/mega_experiments/latency_hop_v3_20260211_stats.csv`

## Workflow figure rule

- Until replaced by a better final version, use:
  - `_LCN26_AERIS/AERIS流程图.pdf`
- The conference draft compiles via the ASCII copy:
  - `_LCN26_AERIS/generated/fig0_aeris_workflow_temp_20260420.pdf`
- Do not hand-edit the temporary copy; refresh it from the source PDF.

## Rebuild chain

Use:
- `scripts/rebuild_lcn26_final_assets.ps1`

This wrapper:
1. refreshes the temporary workflow figure from `_LCN26_AERIS/AERIS流程图.pdf`
2. rebuilds the base figure set
3. refreshes the corrected canonical NS-3 figure
4. refreshes the corrected tradeoff/mechanism figure
5. recompiles `_LCN26_AERIS/aeris_lcn2026.tex`

## Figure generation environment

Use:
- `C:\Users\admin\anaconda3\python.exe`

Do not use:
- `C:\Users\admin\anaconda3\envs\aether-wsn\python.exe`

Reason:
- In this project session, the `aether-wsn` environment crashed at `matplotlib.savefig`.

## Writing constraints

- Prefer concise, conference-style prose.
- Avoid meta-narration about “evidence layers” unless it is needed to clarify fairness.
- Avoid inflated novelty language.
- Make limitations explicit instead of hiding them.
- For TEEN, do not rely on `PDR_expected` alone; mention energy/lifetime/FND/hops alongside it.
- If you mention MAC realism, explicitly note the current simulator does not model full 802.15.4/TSCH/CSMA-CA details.

## High-value local references

Read these before making important paper changes:

- `docs/20260420_review_comment_mapping.md`
- `docs/20260420_lcn26_claim_audit.md`
- `docs/20260420_lcn26_results_writing_card.md`
- `docs/20260420_lcn26_rerun_provenance.md`
- `docs/20260420_lcn26_remaining_work.md`

## When improving the draft

Prefer this order:
1. claim audit
2. prose tightening
3. figure/caption consistency
4. rebuild + PDF inspection

## Never do this

- Do not cite the invalid `010122` rerun as the corrected NS-3 evidence.
- Do not describe PEGASIS office behavior as a perfectly flat line anymore.
- Do not claim Skeleton is a major active contributor in the current publication configuration.
- Do not reintroduce universal-superiority wording.

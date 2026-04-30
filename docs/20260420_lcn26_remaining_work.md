# 2026-04-20 LCN26 Remaining Work

Current base:
- Draft: `_LCN26_AERIS/aeris_lcn2026.tex`
- PDF: `_LCN26_AERIS/aeris_lcn2026.pdf`
- Valid canonical NS-3 source: `ns3_validation/results/lcn26_ns3_audit_20260420_012811/summary/`
- Valid mechanism source: `results/lcn26_targeted_20260420/mechanism_grid_fat/`

## P0: Must finish before submission

1. Workflow figure status.
   - Current paper already uses the user-designated temporary workflow PDF:
     - `_LCN26_AERIS/AERIS流程图.pdf`
     - copied into `_LCN26_AERIS/generated/fig0_aeris_workflow_temp_20260420.pdf`
   - Remaining dependency:
     - only replace it if the user later provides a better final workflow figure

2. Final line-by-line claim audit.
   - Check that every hard number in abstract, results, discussion, and captions matches:
     - corrected NS-3 rerun
     - mechanism summary
     - frozen 100-node publication block
   - Remove any stale wording left from older figure/data generations.
   - Current support files already created:
     - `docs/20260420_lcn26_claim_audit.md`
     - `docs/20260420_review_comment_mapping.md`

3. Final LCN-style prose tightening.
   - Reduce remaining “explainer” tone.
   - Keep the conference paper focused on:
     - problem
     - design
     - corrected evidence
     - deployment boundary

4. Final figure/caption consistency pass.
   - Ensure caption wording matches actual evidence source.
   - Ensure top-level labels such as `canonical`, `strict`, `frozen 100-node`, and `corrected mechanism matrix` are used consistently.

5. Final PDF polish.
   - Recompile
   - check figure readability
   - check last-page balance
   - check no stale figure path remains

## P1: Strongly recommended

6. Add one tighter fairness note in the strict-physics results/discussion transition.
   - Goal:
     - make adapted-baseline limitations explicit without derailing the main story

7. Slightly compress protocol-design wording.
   - Goal:
     - reduce the risk that reviewers read the design as over-claiming three equally active modules

8. Small bibliography cleanup.
   - Check the most recent added related-work items are used where they matter most.

## P2: Optional if we want another quality bump

9. Add a secondary metric note for TEEN.
   - Not a new large experiment package.
   - Just enough text or a small appendix note to reduce the `PDR_expected` bias attack surface.

10. Prepare a one-page reviewer-defense memo.
   - Not for submission.
   - For us to use when checking whether the current version still exposes obvious attack surfaces.

## Not planned for the current minimal path

11. Direct implementation of `CTP / RPL / ORPL / ORW`.
   - Valuable, but not a small patch.
   - This is a separate expansion path, not current submission cleanup.

12. Full MAC realism upgrade to detailed 802.15.4 / TSCH / CSMA-CA.
   - Too large for the current revision cycle.

## Practical estimate

- Remaining to reach a solid non-flowchart-final submission candidate:
  - about `0.25 ~ 0.75` day of focused cleanup
- Remaining after the user provides the final workflow figure:
  - about `1 ~ 2` additional compile / polish passes
- If we also choose to add the optional TEEN/fairness bump:
  - add about `0.5` day

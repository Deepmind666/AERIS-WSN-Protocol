# 20260211 Claude Assignment (Post-Run, No New Heavy Experiments)

## Role Boundary

- Codex (local): manuscript integration and claim gate
- Claude (server): NS-3/provenance evidence closure
- No unassigned large experiment in this card

## Task A (P0): Fix Server Provenance Schema

Goal: make server fix550 provenance files schema-consistent with local sidecars.

Target files:

- C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_indoor_office_server_fix550_20260210_provenance.json
- C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_outdoor_suburban_server_fix550_20260210_provenance.json

Requirements:

1. `script_sha256` must be full 64-hex
2. keep `git_commit`, `git_dirty`, `git_diff_stat` unchanged from actual run context
3. do not modify raw experiment JSON values

Deliverable:

- `*_provenance_v2.json` files with patch note fields:
  - `patched_from`
  - `patch_note`

## Task B (P0): Restore NS-3 Alignment Evidence at Canonical Path

Goal: provide a single canonical NS-3 alignment evidence document.

Target path (must exist after task):

- C:\AERIS-WSN-Protocol\ns3_validation\NS3_ALIGNMENT_EVIDENCE.md

Must include:

1. parameter parity table (Python vs NS-3)
2. 4-environment result summary with file references
3. claim gate statement:
   - what is trend-level safe
   - what is not numeric-level safe

## Task C (P1): No-Run Consistency Audit

Goal: verify cross-file consistency without launching new heavy jobs.

Check list:

1. `ns3_multienv_significance.csv` rows/fields parse correctly
2. all referenced result files in evidence doc exist
3. no contradictory statement in evidence doc vs data table

## Output Template (strict)

1. file list (full paths)
2. what was completed
3. what still needs verification
4. no long narrative, no extra experiment


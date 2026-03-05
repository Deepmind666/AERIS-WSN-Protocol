# Claude Assignment During Local 8h Run (Project-Specific)

Date: 2026-02-11
Owner split:
- Codex: local long-run scalability experiment (resource-capped).
- Claude: server-side evidence-chain and NS-3 gate tasks only (no overlap with local run).

## Constraints (must follow)
- Do not run any local experiments on `C:\AERIS-WSN-Protocol`.
- Do not modify core protocol code unless explicitly approved.
- All outputs must include full paths and reproducible metadata fields.
- Follow claim gate: `docs/20260207_Claim_Gating_List.md`.

## Task C1 (P0): Normalize provenance schema for server_fix550 outputs
Input files:
- `results/mega_experiments/scalability_indoor_office_server_fix550_20260210.json`
- `results/mega_experiments/scalability_outdoor_suburban_server_fix550_20260210.json`
- existing sidecars with `_provenance.json`

Required fixes:
1. Keep original sidecars unchanged.
2. Generate v2 sidecars with full metadata:
   - `git_commit`, `git_dirty`, `git_diff_stat`
   - full-length `script_sha256` (64 hex)
   - `config_hash`
   - `run_tier`, `primary_metric`
   - `raw_results_count`, `error_runs`
3. Name format:
   - `*_provenance_v2_20260211.json`

Acceptance:
- 2 new sidecars created.
- `script_sha256` length is 64 in both files.
- No overwrite of existing sidecars.

## Task C2 (P0): Build merged 4-env scalability metadata table
Input files:
- `results/mega_experiments/scalability_indoor_office_server_fix550_20260210.json`
- `results/mega_experiments/overnight_scalability_20260209_163524/scalability_indoor_factory_20260209_163524.json`
- `results/mega_experiments/scalability_outdoor_urban_fix550_20260210_102734.json`
- `results/mega_experiments/scalability_outdoor_suburban_server_fix550_20260210.json`

Output:
- `results/mega_experiments/scalability_4env_metadata_audit_20260211.csv`
- `results/mega_experiments/scalability_4env_metadata_audit_20260211.md`

Table fields:
- environment
- git_commit
- run_tier
- primary_metric
- raw_results_count
- error_runs
- seeds_count
- node_counts
- rounds
- replicates_inferred (= raw_results_count / (len(node_counts)*5))

Acceptance:
- 4 rows complete.
- Explicitly flag commit mismatch across environments if present.

## Task C3 (P1): NS-3 publication gate execution plan refresh (no new experiment run)
Input docs:
- `docs/20260210_NS3_Publication_Gate_Checklist.md`
- `docs/20260211_NS3_Audit_and_Worksplit.md`

Output:
- `docs/20260211_NS3_Run_Plan_V2.md`

Must include:
- exact commands for 4-environment NS-3 runs
- expected output schema fields
- acceptance gate (trend-level vs numeric-level)
- ETA per stage

Do not run NS-3 in this task. Plan only.

## Response format (mandatory)
1. Files created (full paths)
2. What was completed
3. What still needs verification
4. ETA estimate for any pending step

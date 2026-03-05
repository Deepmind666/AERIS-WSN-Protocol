# 20260210 Claude NS-3 Execution Card (Project-Specific)

Project: C:\AERIS-WSN-Protocol
Owner split: Codex=local, Claude=server

## Hard Constraints
1. Do not modify Python core protocol logic in this round.
2. Every update must include ETA remaining.
3. Server resource cap: CPU <= 65%, MEM <= 65% unless explicitly approved.
4. Keep publication claim gate: before NS-3 gate pass, only trend-level wording is allowed.

## Task NS3-A (P0): Parameter Parity Lock
Goal: finalize a one-to-one Python vs NS-3 parameter table for publication run.

Output:
- C:\AERIS-WSN-Protocol\ns3_validation\results\NS3_PARAMETER_PARITY_20260210.md

Must include exact mapping:
- environment
- tx_power_dbm
- initial_energy
- rounds
- packet_size
- seed list
- path-loss/shadowing model params

Acceptance:
- each field marked MATCH / PARTIAL / MISMATCH
- no empty cells
- unresolved mismatches listed with concrete remediation

## Task NS3-B (P0): Smoke Run (Aligned)
Goal: run one fully aligned smoke case and produce reproducible outputs.

Output:
- ns3_validation/results/ns3_smoke_aligned_20260210.json
- ns3_validation/results/ns3_smoke_aligned_20260210.provenance.json

Metadata required:
- git_commit
- git_dirty
- script_sha256
- config_hash
- run_tier (diagnostic)
- primary_metric (pdr_expected)

Acceptance:
- run success, error_runs = 0
- output schema complete

## Task NS3-C (P1): Publication Matrix Plan (n>=30)
Goal: produce executable matrix and ETA for publication NS-3 run.

Output:
- ns3_validation/results/NS3_PUBLICATION_MATRIX_PLAN_20260210.md

Must include:
- scenario matrix
- seeds (n>=30)
- worker/resource settings
- per-stage ETA and total ETA
- expected JSON outputs and sidecar naming

## Mandatory Reply Template
1. Files produced (full paths)
2. Time spent and ETA remaining
3. Metadata (git_commit, git_dirty, script_sha256, config_hash, run_tier, primary_metric)
4. Pass/fail checks
5. Blockers

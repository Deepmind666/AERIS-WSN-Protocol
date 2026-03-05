# 20260210 Claude Server NS-3 Task Card (Project-Specific)

## Scope
Project root: C:\AERIS-WSN-Protocol
Role split: Codex local + manuscript; Claude server + NS-3 execution.
Hard constraints: CPU <= 65%, MEM <= 65%, no unapproved experiments.

## Task A (P0): NS-3 Parity Freeze
Goal: lock Python-vs-NS-3 parameter parity for publication claims.

Inputs:
- C:\AERIS-WSN-Protocol\docs\20260210_NS3_Publication_Gate_Checklist.md
- C:\AERIS-WSN-Protocol\results\mega_experiments\env_sensitivity_20260207_205317.json
- C:\AERIS-WSN-Protocol\scripts\run_fair_5protocol.py
- NS-3 config/source files currently used on server

Output:
- C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_parity_freeze_20260210.md

Must include:
1. exact mapping table: environment, tx_power_dbm, initial_energy, rounds, packet_size, seed list
2. list of any unresolved mismatches (if none, write "none")
3. run command set for reproducibility

## Task B (P0): NS-3 Publication Run (100 nodes, n=30)
Goal: generate publication-tier NS-3 evidence for 4 environments.

Run matrix:
- protocols: AERIS, LEACH, PEGASIS, HEED, TEEN
- environments: indoor_office, indoor_factory, outdoor_urban, outdoor_suburban
- seeds: 42001-42030
- nodes: 100
- rounds: 300

Outputs (required):
- C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_env100_publication_20260210.json
- C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_env100_publication_20260210.provenance.json

Mandatory metadata fields:
- git_commit, git_dirty, git_diff_stat, script_sha256, config_hash, run_tier=publication, primary_metric=pdr_expected

## Task C (P1): Python vs NS-3 Comparison Pack
Goal: produce direct comparison table for manuscript insertion.

Inputs:
- ns3_env100_publication_20260210.json
- env_sensitivity_20260207_205317.json

Outputs:
- C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_vs_python_env100_20260210.csv
- C:\AERIS-WSN-Protocol\ns3_validation\results\ns3_vs_python_env100_20260210.md

Required columns:
- environment, protocol, pdr_python_mean, pdr_ns3_mean, abs_diff, trend_match

## Reporting Template (Mandatory)
1. files produced (full paths)
2. time spent and ETA remaining
3. metadata values
4. quality checks pass/fail
5. blockers

## Forbidden
- do not modify manuscript files
- do not modify local-only scripts without approval
- do not run extra long jobs outside A/B/C

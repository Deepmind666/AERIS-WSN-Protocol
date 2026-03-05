# 20260210 Claude Task Card (Next Round)

## Scope
Project: C:\AERIS-WSN-Protocol
Owner split: Codex local, Claude server.
Do not modify core algorithm files unless explicitly approved.

## Task A (P0): Server-side Scalability Consistency Pack
Goal: ensure 4-env scalability evidence is complete and consistently packaged.

Inputs:
- C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_indoor_office_server_fix550_20260210.json
- C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_indoor_factory_20260209_163524.json
- C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_outdoor_urban_fix550_20260210_102734.json
- C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_outdoor_suburban_server_fix550_20260210.json

Required output:
- one summary CSV + one summary MD with per-env, per-node PDR ranks
- verify each file raw_results count == 16500 and error_runs == 0
- verify provenance sidecar exists for each file

## Task B (P0): NS-3 Publication Gate Preparation
Goal: produce executable NS-3 run plan with parameter parity table (Python vs NS-3).

Required output file:
- C:\AERIS-WSN-Protocol\ns3_validation\results\NS3_PUBLICATION_RUN_PLAN_20260210.md

Must include:
1. exact parameter mapping table (env, tx_power, initial_energy, rounds, packet_size, seeds)
2. run matrix (n >= 30 seeds) and expected runtime per case
3. output schema requirements matching project RULES
4. hard gate statement: before/after conditions for numeric-level NS-3 claim

## Task C (P1): NS-3 Smoke Execution (only if Task B complete)
Goal: run one aligned smoke case and return reproducible artifacts.

Required output:
- ns3_validation/results/ns3_smoke_*_20260210.json
- sidecar metadata (git_commit, run_tier, seeds, config hash)
- short comparison table against Python same-case result

## Response Template (Mandatory)
1. Files produced (full paths)
2. Time spent and ETA remaining
3. Metadata: git_commit, git_dirty, script_sha256, config_hash, run_tier, primary_metric
4. Quality checks: pass/fail items
5. Blockers (if any)

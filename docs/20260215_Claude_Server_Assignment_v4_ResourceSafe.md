# Claude Server Assignment v4 (Resource-Safe, No-Reboot Policy)

## Scope

You are assigned **server-side support only**.  
Do **not** launch new high-load long runs until this checklist is completed.

## Hard constraints

1. Chinese-only responses.
2. Do not edit `src/` unless explicitly approved.
3. Resource ceiling:
   - CPU <= 75%
   - Memory <= 80%
4. If memory exceeds 85% for 60s, auto-reduce workers and pause new task submission.
5. Every report must include ETA basis and remaining time.

## Tasks

### T1. Validate current S8 server evidence package (no rerun)
- Input files:
  - `results/mega_experiments/scalability_outdoor_urban_server_s8_20260213.json`
  - `results/mega_experiments/scalability_outdoor_suburban_server_s8_20260213.json`
  - `results/mega_experiments/scalability_indoor_office_server_s8_20260213.json`
  - `results/mega_experiments/scalability_indoor_factory_server_s8_20260215.json`
  - their corresponding `.provenance.json`
- Output:
  - `results/mega_experiments/s8_server_validation_report_20260215.md`
- Checkpoints:
  - each cell n=1000
  - error handling consistency
  - provenance completeness
  - commit consistency explanation

### T2. NS-3 boundary hardening (docs only)
- Input:
  - `ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md`
  - `ns3_validation/results/NS3_CLAIM_GATE.md`
- Output:
  - `ns3_validation/results/NS3_GATE_CONSISTENCY_CHECK_20260215.md`
- Goal:
  - ensure all claims remain trend-level and no numerical-equivalence language leaks into writable claims.

### T3. Resource-safe runner profile
- Create:
  - `scripts/server_resource_safe_profile.ps1`
- Behavior:
  - staged worker escalation (6 -> 8 -> 10)
  - hard stop at memory > 85%
  - heartbeat log every 5 minutes
  - no background mode that loses child processes silently

## Do not do

- Do not start new 4-env x n=1000 reruns.
- Do not start NS-3 new experiment batches.
- Do not change manuscript text.

## Return template

1. File list  
2. What was completed  
3. Remaining verification items  
4. Current server resource snapshot (CPU/MEM)  
5. If any command failed, include exact command + error tail


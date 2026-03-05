# AERIS Local-Server Experiment Workflow SOP (Project-Specific)

Version: 2026-02-10
Scope: This SOP applies only to `C:\AERIS-WSN-Protocol`.

## 1. Role Split (Mandatory)
- Codex (local):
  - Owns local code changes, manuscript updates, local experiment execution, and final gate decisions.
  - Enforces local resource guard: CPU <= 70%, MEM <= 70%.
- Claude (server):
  - Owns server environment execution and data return.
  - Must run only approved commands and approved commit baseline.

## 2. Pre-Run Checklist (Mandatory)
Before each experiment run, report:
1. Path + Plan + Impact scope.
2. Estimated remaining runtime (ETA) in hh:mm.
3. Resource cap settings (workers, max CPU, max MEM).
4. Git state (`git_commit`, `git_dirty`, `git_diff_stat`).

No run is allowed before this checklist is posted.

## 3. Runtime Reporting Rule (Mandatory)
- Every progress report must include:
  - Current completion (e.g., completed tasks / total tasks).
  - Estimated remaining runtime (ETA).
- If ETA confidence is low, state range (e.g., 01:20-01:50).

## 4. Resource Guard Rule (Mandatory)
- Local machine:
  - Hard cap CPU <= 70%, MEM <= 70%.
  - Prefer scripts with resource guards (`--max-cpu-percent`, `--max-mem-percent`).
- Server:
  - Default cap CPU <= 65%, MEM <= 65% unless explicitly approved.

## 5. Evidence and Metadata Rule (Mandatory)
Every publication-tier result must include:
- `timestamp`, `git_commit`, `git_dirty`, `git_diff_stat`
- `run_tier=publication`, `primary_metric=pdr_expected`
- `config.seeds`, `config.node_counts`, `config.round_counts`
- `raw_results` with `error` status

Provenance sidecar is required for overnight/scalability batches.

## 6. Manuscript Gate Rule (Mandatory)
- No claim is allowed without file evidence.
- Forbidden claims follow `docs/20260207_Claim_Gating_List.md`.
- If scale-level results show mixed ranking (e.g., AERIS leads 3/4 envs), manuscript text must remain scoped and conditional.

## 7. NS-3 Publication Gate (Mandatory)
- NS-3 is a hard gate for publication-level finalization in this project.
- Until NS-3 gate is passed, manuscript can only claim trend-level cross-validation.
- Required NS-3 artifacts:
  - Parameter alignment table
  - Publication-tier JSON results (n >= 30)
  - Welch + Hedges g + Holm statistics

## 8. Handoff Template (Mandatory)
Use this exact structure in each handoff:
1. Files produced (full paths)
2. Time spent and ETA remaining
3. Key metadata (`git_commit`, `git_dirty`, `script_sha256`, `config_hash`)
4. Quality checks passed/failed
5. Next action request

## 9. Stop Conditions
Stop and escalate immediately if:
- Resource guard is violated repeatedly.
- Output schema is inconsistent with RULES.
- New results conflict with manuscript claims.
- Commit baseline changes during long run without explicit approval.

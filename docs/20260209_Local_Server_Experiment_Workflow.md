# AERIS Local-Server Experiment Workflow (v20260209)

Last updated: 2026-02-09
Project: AERIS-WSN-Protocol

## 1) Roles and Scope

- Codex (local owner): runs local experiments, code reviews, paper consistency checks, final gate decisions.
- Claude (server owner): runs server experiments only, returns artifacts and metadata, does not change core algorithm unless explicitly assigned.
- Rule: no unassigned experiments, no scope creep.

## 2) Hard Resource Limits

- Local machine:
  - CPU usage must stay <= 70%
  - Memory usage must stay <= 70%
- Server:
  - CPU usage must stay <= 70%
  - Memory usage must stay <= 70%
- If either limit is exceeded for sustained periods, stop new launches and reduce workers.

## 3) Mandatory ETA Reporting (every status update)

Every experiment update must include:

1. Current stage
2. Elapsed time
3. Basis for estimation (log timestamp + completed/total count)
4. Remaining ETA window
5. Expected finish time

Template:

Current stage: <stage>
Elapsed: <time>
Progress basis: <done>/<total> from <log file + timestamp>
Remaining ETA: <xx-yy min>
Expected finish: <YYYY-MM-DD HH:MM:SS>

## 4) Pre-Run Gate

Before any code/doc change or long experiment launch, provide:

- Path
- Plan
- Impact scope

Only execute after explicit approval.

## 5) Standard Launch Commands

### 5.1 Local scalability run (Codex)

Use resource guards explicitly:

python scripts/run_scalability_experiment.py --nodes 100,200,300,500,800,1000 --replicates <N> --workers <W> --rounds 300 --env <ENV> --max-cpu-percent 65 --max-mem-percent 65 --run-tier publication --output results/mega_experiments/<FILE>.json

### 5.2 Server scalability run (Claude)

Use conda-run (do not rely on default python):

conda run -n aether-wsn python scripts/run_scalability_experiment.py --nodes 100,200,300,500,800,1000 --replicates <N> --workers <W> --rounds 300 --env <ENV> --max-cpu-percent 65 --max-mem-percent 65 --run-tier publication --output results/mega_experiments/<FILE>.json

Then generate provenance:

python scripts/generate_scalability_provenance.py --overnight-dir results/mega_experiments/<DIR>

## 6) Artifact Requirements

Each publication-tier output must include:

- timestamp
- git_commit
- git_dirty
- git_diff_stat
- script_sha256
- config_hash
- run_tier = publication
- primary_metric = pdr_expected

For overnight runs:

- manifest.json
- per-environment JSON outputs
- per-environment .provenance.json

## 7) Acceptance Checks

Minimum checks before claiming completion:

1. exit_code == 0 for each environment
2. raw_results count matches expected cardinality
3. no forbidden claims introduced in manuscript sections
4. stats files and evidence paths are reproducible

## 8) Forbidden Claim Gate (must stay green)

Follow:

C:\AERIS-WSN-Protocol\docs\20260207_Claim_Gating_List.md

Examples of banned claims without dedicated publication evidence:

- "200 independent runs"
- "100% PDR at 500 nodes"
- unverified TDA claims
- absolute latency claims like "<10ms", "2500ms"

## 9) Handoff Format (Codex <-> Claude)

Every handoff must include:

- File list (full paths)
- What was done
- What still needs verification
- Current ETA block

## 10) Immediate Assignment Rule

- Local ongoing experiments continue under Codex control.
- Server runs are launched by Claude only after Codex confirms:
  - current local load is stable
  - exact command set
  - target output directory naming

## 11) GitHub Sync Gate (project-only)

Before any server run:

1. Local side records target commit hash.
2. Commit must be pushed to remote.
3. Server side checks out that exact commit.
4. Experiment metadata must include `git_commit`, `git_dirty`, `git_diff_stat`, `script_sha256`.

If server commit != assigned commit, results are marked "invalid for publication gate".

## 12) NS-3 Publication Gate (project-only)

NS-3 can be cited as numerical validation only when all are true:

1. Parameters aligned with Python experiment:
   - environment
   - tx_power_dbm
   - initial_energy
   - rounds
   - packet_size
   - seeds
2. Sample size per core scenario: n >= 30.
3. Raw JSON + stats + alignment note all present:
   - ns3_validation/results/ns3_*_publication*.json
   - Welch/Hedges/Holm stats output
   - ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md
4. If not fully aligned, manuscript wording must be "trend-level validation".

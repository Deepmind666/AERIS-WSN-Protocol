# 20260211 Claude Assignment (NS-3 Scale Extension, Server)

## Scope

This card is server-only. Do not edit Python algorithm code.  
Goal is to strengthen NS-3 evidence at larger node scales while keeping resource control.

## Execution Guardrails (mandatory)

1. Chinese-only communication for all progress and review messages.
2. After compact/context recovery, first line must be:
   `【恢复确认】已恢复上下文，将全程中文输出，并按本规则执行。`
3. Do not high-frequency poll logs.
   - default polling interval >= 30 min
   - long run (>4h) polling interval >= 45 min
   - earlier check only when user explicitly asks or stage changes
4. Every progress update must include ETA (or ETA range).

## Task A (P0): Run NS-3 scale extension

### Target

- Environments: `indoor_office`, `indoor_factory`, `outdoor_urban`, `outdoor_suburban`
- Protocols: `AERIS`, `LEACH`
- Node counts: `300`, `500`
- Seeds: `42001..42030` (n=30)
- Rounds: `300`

### Resource limits (mandatory)

- CPU <= `70%`
- Memory <= `70%`
- If either exceeds limit for sustained > 3 min, reduce workers and continue.

### Output files (required)

- `C:\\AERIS-WSN-Protocol\\ns3_validation\\results\\ns3_multienv_scaleext_20260211.json`
- `C:\\AERIS-WSN-Protocol\\ns3_validation\\results\\ns3_multienv_scaleext_stats.csv`
- `C:\\AERIS-WSN-Protocol\\ns3_validation\\results\\ns3_multienv_scaleext_significance.csv`

## Task B (P0): Claim gate update for NS-3 evidence

Update:

- `C:\\AERIS-WSN-Protocol\\ns3_validation\\results\\NS3_ALIGNMENT_EVIDENCE.md`

Required updates:

1. Add 300/500 node findings by environment.
2. Keep wording strictly trend-level unless numerical gate is fully satisfied.
3. Add a short "Can claim / Cannot claim" block for 300/500.

## Task C (P1): Provenance sidecar for NS-3 extension

Generate:

- `C:\\AERIS-WSN-Protocol\\ns3_validation\\results\\ns3_multienv_scaleext_20260211.provenance.json`

Required fields:

- `git_commit`
- `git_dirty`
- `git_diff_stat`
- `script_sha256` (full 64 hex)
- `config_hash`
- `run_tier` = `publication`
- `primary_metric` = `pdr_expected`

## Report format (strict)

1. File list (full paths)
2. What completed
3. Remaining verification items
4. Runtime summary (total time + ETA method)

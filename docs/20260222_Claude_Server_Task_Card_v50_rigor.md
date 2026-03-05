# Claude Server Task Card (v50-rigor)

Date: 2026-02-22  
Owner: Claude (server), Codex (local)

## 1) Goal

Run the v50-rigor full scalability matrix on server for the remaining 3 environments under the corrected fairness settings.

- Branch: `v50-rigor`
- Required options: `--mac-collision --multihop-relay`
- Metric: `pdr_expected`
- Run tier: `publication`

## 2) Required experiment scope

Run these 3 jobs on server:

1. `indoor_office`
2. `outdoor_suburban`
3. `outdoor_urban`

Common parameters (all jobs):

- `--replicates 1000`
- `--seed 42001`
- `--nodes 100,200,300,500,800,1000`
- `--rounds 300`
- `--workers 20`
- `--tx-power 10.0`
- `--run-tier publication`
- `--max-cpu-percent 90`
- `--max-mem-percent 90`
- `--resource-check-sec 1`
- `--allow-partial`
- `--mac-collision`
- `--multihop-relay`

Output naming (keep this exact pattern):

- `results/mega_experiments/scalability_<env>_v50rigor_20260222.json`

## 3) Server usage method (must follow)

### 3.1 Connect and environment check

Use full Python path (do not rely on `conda activate` in remote one-liners):

```powershell
ssh FatMachine
cd C:\Users\sshuser\AERIS-WSN
C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe --version
```

### 3.2 Reliable long-run launch

Use `Start-Process` with separate stdout/stderr logs and full Python path:

```powershell
$py = "C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe"
$envName = "indoor_office"   # change per run
$out = "results/mega_experiments/scalability_${envName}_v50rigor_20260222.json"
$logOut = "results/mega_experiments/scalability_${envName}_v50rigor_20260222.out.log"
$logErr = "results/mega_experiments/scalability_${envName}_v50rigor_20260222.err.log"

$args = @(
  "scripts/run_scalability_experiment.py",
  "--env", $envName,
  "--replicates", "1000",
  "--seed", "42001",
  "--nodes", "100,200,300,500,800,1000",
  "--rounds", "300",
  "--workers", "20",
  "--run-tier", "publication",
  "--tx-power", "10.0",
  "--max-cpu-percent", "90",
  "--max-mem-percent", "90",
  "--resource-check-sec", "1",
  "--mac-collision",
  "--multihop-relay",
  "--allow-partial",
  "--output", $out
)

Start-Process -FilePath $py -ArgumentList $args -RedirectStandardOutput $logOut -RedirectStandardError $logErr -WindowStyle Hidden
```

If detached process is not stable in your current shell/session, fallback to WMI process creation (`Win32_Process.Create`) and keep the same full command line.

### 3.3 Monitoring commands

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -eq "python.exe" -and $_.CommandLine -like "*run_scalability_experiment.py*" } | Select-Object ProcessId,CommandLine
Get-Content results/mega_experiments/scalability_<env>_v50rigor_20260222.out.log -Tail 30
Get-Counter "\Processor(_Total)\% Processor Time"
Get-Counter "\Memory\% Committed Bytes In Use"
```

### 3.4 Resource policy

- Keep CPU near 90% but avoid hard saturation.
- Do not run multiple heavy `run_scalability_experiment.py` jobs in parallel on the same server.
- Run environments sequentially.

## 4) Acceptance criteria

For each environment JSON:

1. `raw_results == 30000`
2. `error_runs == 0` (or clearly documented if non-zero)
3. `run_tier == "publication"`
4. `primary_metric == "pdr_expected"`
5. `git_commit` and config metadata present

## 5) Deliverables to return

Return exactly:

1. 3 JSON paths
2. 3 provenance sidecar paths (`.provenance.json`)
3. Per-file summary: `raw_results`, `error_runs`, `run_tier`, `primary_metric`, `git_commit`
4. Per-environment AERIS PDR means at `n=100,500,1000`
5. Remaining-time estimate during execution based on current log throughput

## 6) Notes for rigor

- This run is part of the reviewer-fatal fix path (R1 physics realism, R2 fairness).
- Do not mix old S8/S9/S10 control claims with v50-rigor outputs.
- Do not modify manuscript claims until all 4 environments are merged and significance is regenerated.

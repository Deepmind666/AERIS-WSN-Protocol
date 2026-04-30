$ErrorActionPreference = 'Stop'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$projectWin = 'C:\AERIS-WSN-Protocol'
$projectWsl = '/mnt/c/AERIS-WSN-Protocol'
$tag = 'office_factory'
$outDirWin = Join-Path $projectWin "ns3_validation\results\lcn26_ns3_dual_local_${tag}_$stamp"
$outDirWsl = "$projectWsl/ns3_validation/results/lcn26_ns3_dual_local_${tag}_$stamp"
$logDirWin = Join-Path $outDirWin 'logs'
$stdoutLog = Join-Path $logDirWin 'dual_local.stdout.log'
$stderrLog = Join-Path $logDirWin 'dual_local.stderr.log'

$protocols = 'AERIS LEACH HEED PEGASIS TEEN RPL-MRHOF CTP'
$envs = 'indoor_office indoor_factory'
$nodes = '50,100,200,300,500,800,1000'
$parallel = 6

New-Item -ItemType Directory -Force $logDirWin | Out-Null

$bashCmd = @'
set -euo pipefail
cd '__PROJECT_WSL__'
export NS3_ROOT=/home/lkr/ns-allinone-3.40/ns-3.40
export BIN=/home/lkr/ns-allinone-3.40/ns-3.40/build/scratch/ns3.40-aeris-validation-standalone-default
export PROTOCOLS='__PROTOCOLS__'
export ENVS='__ENVS__'
export NODES='__NODES__'
bash ns3_validation/run_lcn26_focused_matrix.sh __PARALLEL__ '__OUTDIR__'
mkdir -p '__OUTDIR__/summary'
python3 ns3_validation/merge_lcn26_focused_results.py --input-dir '__OUTDIR__/raw' --output-dir '__OUTDIR__/summary'
echo "[DONE] local dual split complete"
'@

$bashCmd = $bashCmd.Replace('__PROJECT_WSL__', $projectWsl).
  Replace('__PROTOCOLS__', $protocols).
  Replace('__ENVS__', $envs).
  Replace('__NODES__', $nodes).
  Replace('__PARALLEL__', "$parallel").
  Replace('__OUTDIR__', $outDirWsl)

$runScriptWin = Join-Path $outDirWin 'run_dual_local.sh'
$runScriptWsl = "$outDirWsl/run_dual_local.sh"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($runScriptWin, $bashCmd, $utf8NoBom)

$proc = Start-Process -FilePath 'wsl.exe' -ArgumentList @('--', 'bash', $runScriptWsl) `
  -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput $stdoutLog `
  -RedirectStandardError $stderrLog

Write-Host "[STARTED] local dual NS-3 PID: $($proc.Id)"
Write-Host "[OUTDIR] $outDirWin"
Write-Host "[MATRIX] protocols=$protocols envs=$envs nodes=$nodes seeds=30"
Write-Host "[EXPECTED] shards=14 experiments=2940"
Write-Host "[LOG] stdout: $stdoutLog"
Write-Host "[LOG] stderr: $stderrLog"

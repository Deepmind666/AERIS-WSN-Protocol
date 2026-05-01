$ErrorActionPreference = 'Stop'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$projectWin = 'C:\AERIS-WSN-Protocol'
$projectWsl = '/mnt/c/AERIS-WSN-Protocol'
$tag = 'office_factory'
$outDirWin = Join-Path $projectWin "ns3_validation\results\lcn26_ns3_ablation_local_${tag}_$stamp"
$outDirWsl = "$projectWsl/ns3_validation/results/lcn26_ns3_ablation_local_${tag}_$stamp"
$logDirWin = Join-Path $outDirWin 'logs'
$stdoutLog = Join-Path $logDirWin 'ablation_local.stdout.log'
$stderrLog = Join-Path $logDirWin 'ablation_local.stderr.log'

$protocols = 'ABLATION'
$envs = 'indoor_office indoor_factory'
$nodes = '50,100,200,300,500,800,1000'
$parallel = 2

New-Item -ItemType Directory -Force $logDirWin | Out-Null

$bashCmd = @'
set -euo pipefail
cd '__PROJECT_WSL__'
cp ns3_validation/aeris-validation-standalone.cc /home/lkr/ns-allinone-3.40/ns-3.40/scratch/aeris-validation-standalone.cc
cd /home/lkr/ns-allinone-3.40/ns-3.40
./ns3 build
cd '__PROJECT_WSL__'
export NS3_ROOT=/home/lkr/ns-allinone-3.40/ns-3.40
export BIN=/home/lkr/ns-allinone-3.40/ns-3.40/build/scratch/ns3.40-aeris-validation-standalone-default
export PROTOCOLS='__PROTOCOLS__'
export ENVS='__ENVS__'
export NODES='__NODES__'
bash ns3_validation/run_lcn26_focused_matrix.sh __PARALLEL__ '__OUTDIR__'
mkdir -p '__OUTDIR__/summary'
python3 ns3_validation/merge_lcn26_focused_results.py --input-dir '__OUTDIR__/raw' --output-dir '__OUTDIR__/summary'
echo "[DONE] local ablation split complete"
'@

$bashCmd = $bashCmd.Replace('__PROJECT_WSL__', $projectWsl).
  Replace('__PROTOCOLS__', $protocols).
  Replace('__ENVS__', $envs).
  Replace('__NODES__', $nodes).
  Replace('__PARALLEL__', "$parallel").
  Replace('__OUTDIR__', $outDirWsl)

$runScriptWin = Join-Path $outDirWin 'run_ablation_local.sh'
$runScriptWsl = "$outDirWsl/run_ablation_local.sh"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($runScriptWin, $bashCmd, $utf8NoBom)

$proc = Start-Process -FilePath 'wsl.exe' -ArgumentList @('--', 'bash', $runScriptWsl) `
  -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput $stdoutLog `
  -RedirectStandardError $stderrLog

Write-Host "[STARTED] local ablation NS-3 PID: $($proc.Id)"
Write-Host "[OUTDIR] $outDirWin"
Write-Host "[MATRIX] protocols=$protocols envs=$envs nodes=$nodes seeds=30 variants=4"
Write-Host "[EXPECTED] shards=2 experiments=1680"
Write-Host "[LOG] stdout: $stdoutLog"
Write-Host "[LOG] stderr: $stderrLog"

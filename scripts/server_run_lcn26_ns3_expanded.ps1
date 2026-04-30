$ErrorActionPreference = 'Stop'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$outDir = "/mnt/c/Users/sshuser/AERIS-WSN-Protocol/ns3_validation/results/lcn26_ns3_expanded_$stamp"
$outDirWin = "C:\Users\sshuser\AERIS-WSN-Protocol\ns3_validation\results\lcn26_ns3_expanded_$stamp"
$logDirWin = Join-Path $outDirWin "logs"
$parallel = 8
$script = "/mnt/c/Users/sshuser/AERIS-WSN-Protocol/ns3_validation/run_lcn26_focused_matrix.sh"
$merge = "/mnt/c/Users/sshuser/AERIS-WSN-Protocol/ns3_validation/merge_lcn26_focused_results.py"
$protocols = "AERIS LEACH HEED PEGASIS TEEN RPL-MRHOF CTP"

New-Item -ItemType Directory -Force $logDirWin | Out-Null
$stdoutLog = Join-Path $logDirWin "ns3_expanded.stdout.log"
$stderrLog = Join-Path $logDirWin "ns3_expanded.stderr.log"

$bashCmd = @'
set -euo pipefail
export PROTOCOLS='__PROTOCOLS__'
bash '__SCRIPT__' __PARALLEL__ '__OUTDIR__'
mkdir -p '__OUTDIR__/summary'
python3 '__MERGE__' --input-dir '__OUTDIR__/raw' --output-dir '__OUTDIR__/summary'
echo "[DONE] expanded matrix and merge complete"
'@

$bashCmd = $bashCmd.Replace('__PROTOCOLS__', $protocols).Replace('__SCRIPT__', $script).Replace('__PARALLEL__', "$parallel").Replace('__OUTDIR__', $outDir).Replace('__MERGE__', $merge)
$runScriptWin = Join-Path $outDirWin 'run_expanded.sh'
$runScriptWsl = "$outDir/run_expanded.sh"
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($runScriptWin, $bashCmd, $utf8NoBom)

$args = @(
  "-u", "ns3user", "--",
  "bash", $runScriptWsl
)

$proc = Start-Process -FilePath "wsl.exe" -ArgumentList $args `
  -WindowStyle Hidden -PassThru `
  -RedirectStandardOutput $stdoutLog `
  -RedirectStandardError $stderrLog

Write-Host "[STARTED] server expanded ns3 PID: $($proc.Id)"
Write-Host "[OUTDIR] $outDir"
Write-Host "[PROTOCOLS] $protocols"
Write-Host "[SCRIPT] $runScriptWsl"
Write-Host "[LOG] stdout: $stdoutLog"
Write-Host "[LOG] stderr: $stderrLog"

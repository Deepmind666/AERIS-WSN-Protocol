$ErrorActionPreference = 'Stop'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$outDir = "/mnt/c/Users/sshuser/AERIS-WSN-Protocol/ns3_validation/results/lcn26_ns3_audit_$stamp"
$logDirWin = "C:\Users\sshuser\AERIS-WSN-Protocol\ns3_validation\results\lcn26_ns3_audit_$stamp\logs"
$parallel = 8
$script = "/mnt/c/Users/sshuser/AERIS-WSN-Protocol/ns3_validation/run_lcn26_focused_matrix.sh"

New-Item -ItemType Directory -Force $logDirWin | Out-Null
$stdoutLog = Join-Path $logDirWin "ns3_audit.stdout.log"
$stderrLog = Join-Path $logDirWin "ns3_audit.stderr.log"

$args = @(
  "-u", "ns3user", "--",
  "bash", $script, $parallel, $outDir
)

$proc = Start-Process -FilePath "wsl.exe" -ArgumentList $args `
  -NoNewWindow -PassThru `
  -RedirectStandardOutput $stdoutLog `
  -RedirectStandardError $stderrLog

Write-Host "[STARTED] server ns3 audit PID: $($proc.Id)"
Write-Host "[OUTDIR] $outDir"
Write-Host "[LOG] stdout: $stdoutLog"
Write-Host "[LOG] stderr: $stderrLog"

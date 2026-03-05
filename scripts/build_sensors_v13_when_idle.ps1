param(
    [double]$MaxCpuPercent = 65.0,
    [double]$MaxMemPercent = 80.0,
    [int]$CheckIntervalSec = 15
)

$ErrorActionPreference = "Stop"

function Wait-ForIdle {
    param(
        [double]$CpuLimit,
        [double]$MemLimit,
        [int]$SleepSec
    )
    while ($true) {
        $cpu = (Get-Counter '\Processor(_Total)\% Processor Time').CounterSamples[0].CookedValue
        $mem = (Get-Counter '\Memory\% Committed Bytes In Use').CounterSamples[0].CookedValue
        if ($cpu -le $CpuLimit -and $mem -le $MemLimit) {
            Write-Host "[IdleGuard] Ready: CPU=$([math]::Round($cpu,1))%, MEM=$([math]::Round($mem,1))%"
            break
        }
        Write-Host "[IdleGuard] Waiting: CPU=$([math]::Round($cpu,1))% (<=${CpuLimit}), MEM=$([math]::Round($mem,1))% (<=${MemLimit})"
        Start-Sleep -Seconds $SleepSec
    }
}

Set-Location "C:\AERIS-WSN-Protocol\for_submission"
Wait-ForIdle -CpuLimit $MaxCpuPercent -MemLimit $MaxMemPercent -SleepSec $CheckIntervalSec

pdflatex -interaction=nonstopmode -halt-on-error AERIS_Sensors_MDPI_Submission_Draft_20260213_v13.tex
bibtex AERIS_Sensors_MDPI_Submission_Draft_20260213_v13
pdflatex -interaction=nonstopmode -halt-on-error AERIS_Sensors_MDPI_Submission_Draft_20260213_v13.tex
pdflatex -interaction=nonstopmode -halt-on-error AERIS_Sensors_MDPI_Submission_Draft_20260213_v13.tex

Set-Location "C:\AERIS-WSN-Protocol"
python scripts\check_sensors_draft_gate.py --draft for_submission\AERIS_Sensors_MDPI_Submission_Draft_20260213_v13.tex
python scripts\audit_sensors_v3_consistency.py --draft for_submission\AERIS_Sensors_MDPI_Submission_Draft_20260213_v13.tex --out-prefix 20260213_Sensors_v13_Data_Consistency_Audit

Write-Host "[DONE] v13 compile + gate + audit completed."

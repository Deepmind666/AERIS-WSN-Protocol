$ErrorActionPreference = "SilentlyContinue"

while ($true) {
    $target = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq "python.exe" -and
        $_.CommandLine -like "*run_scalability_experiment.py*" -and
        $_.CommandLine -like "*--env outdoor_urban*" -and
        $_.CommandLine -like "*v50rigor*"
    }

    if ($target) {
        foreach ($p in $target) {
            Stop-Process -Id $p.ProcessId -Force
        }
        break
    }

    Start-Sleep -Seconds 5
}

$py = "C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe"
$wd = "C:\Users\sshuser\AERIS-WSN"
$out = "results\mega_experiments\scalability_indoor_office_v50rigor_20260222.json"
$logOut = "results\mega_experiments\scalability_indoor_office_v50rigor_20260222.out.log"
$logErr = "results\mega_experiments\scalability_indoor_office_v50rigor_20260222.err.log"

$args = @(
  "scripts/run_scalability_experiment.py",
  "--env", "indoor_office",
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

Set-Location $wd
Start-Process -FilePath $py -ArgumentList $args -RedirectStandardOutput $logOut -RedirectStandardError $logErr -WindowStyle Hidden

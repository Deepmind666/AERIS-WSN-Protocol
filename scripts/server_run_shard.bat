@echo off
setlocal

set PYTHON_CMD=C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe
set PROJECT=C:\Users\sshuser\AERIS-WSN-Protocol
set OUTDIR=%PROJECT%\results\mega_experiments\overnight_scalability_20260209_233631
set TIMESTAMP=20260209_233631

cd /d %PROJECT%

echo [%date% %time%] Starting server shard experiment >> %OUTDIR%\run.log
echo [%date% %time%] Environments: indoor_office, outdoor_suburban >> %OUTDIR%\run.log
echo [%date% %time%] Commit: >> %OUTDIR%\run.log
git rev-parse --short=8 HEAD >> %OUTDIR%\run.log

echo [%date% %time%] === ENV 1/2: indoor_office === >> %OUTDIR%\run.log
%PYTHON_CMD% scripts/run_scalability_experiment.py ^
  --nodes 100,200,300,500,800,1000 ^
  --replicates 550 ^
  --workers 12 ^
  --rounds 300 ^
  --env indoor_office ^
  --max-cpu-percent 65 ^
  --max-mem-percent 65 ^
  --run-tier publication ^
  --output %OUTDIR%\scalability_indoor_office_%TIMESTAMP%.json ^
  >> %OUTDIR%\stdout_indoor_office.log 2>> %OUTDIR%\stderr_indoor_office.log

echo [%date% %time%] indoor_office exit code: %ERRORLEVEL% >> %OUTDIR%\run.log

echo [%date% %time%] === ENV 2/2: outdoor_suburban === >> %OUTDIR%\run.log
%PYTHON_CMD% scripts/run_scalability_experiment.py ^
  --nodes 100,200,300,500,800,1000 ^
  --replicates 550 ^
  --workers 12 ^
  --rounds 300 ^
  --env outdoor_suburban ^
  --max-cpu-percent 65 ^
  --max-mem-percent 65 ^
  --run-tier publication ^
  --output %OUTDIR%\scalability_outdoor_suburban_%TIMESTAMP%.json ^
  >> %OUTDIR%\stdout_outdoor_suburban.log 2>> %OUTDIR%\stderr_outdoor_suburban.log

echo [%date% %time%] outdoor_suburban exit code: %ERRORLEVEL% >> %OUTDIR%\run.log
echo [%date% %time%] Server shard experiment finished. >> %OUTDIR%\run.log

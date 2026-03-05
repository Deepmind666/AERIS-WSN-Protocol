@echo off
setlocal
set PYTHON=C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe
set WORKDIR=C:\Users\sshuser\AERIS-WSN-Protocol
set SCRIPT=%WORKDIR%\scripts\run_scalability_experiment.py
set OUTDIR=%WORKDIR%\results\mega_experiments
set LOG=%WORKDIR%\results\mega_experiments\server_s9_bundle_20260216.log

cd /d %WORKDIR%

echo [%date% %time%] S9 BUNDLE START >> %LOG%

echo [%date% %time%] S9-A-1: indoor_office PATCH start >> %LOG%
%PYTHON% %SCRIPT% --env indoor_office --replicates 1000 --seed 62001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 10.0 --max-cpu-percent 88 --max-mem-percent 82 --mac-collision --multihop-relay --output %OUTDIR%\scalability_indoor_office_server_s9_patch_20260216.json >> %LOG% 2>&1
echo [%date% %time%] S9-A-1: indoor_office PATCH done (exit=%errorlevel%) >> %LOG%

echo [%date% %time%] S9-A-2: outdoor_suburban PATCH start >> %LOG%
%PYTHON% %SCRIPT% --env outdoor_suburban --replicates 1000 --seed 72001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 10.0 --max-cpu-percent 88 --max-mem-percent 82 --mac-collision --multihop-relay --output %OUTDIR%\scalability_outdoor_suburban_server_s9_patch_20260216.json >> %LOG% 2>&1
echo [%date% %time%] S9-A-2: outdoor_suburban PATCH done (exit=%errorlevel%) >> %LOG%

echo [%date% %time%] S9-B-1: indoor_office CONTROL start >> %LOG%
%PYTHON% %SCRIPT% --env indoor_office --replicates 600 --seed 82001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 10.0 --max-cpu-percent 88 --max-mem-percent 82 --output %OUTDIR%\scalability_indoor_office_server_s9_control_20260216.json >> %LOG% 2>&1
echo [%date% %time%] S9-B-1: indoor_office CONTROL done (exit=%errorlevel%) >> %LOG%

echo [%date% %time%] S9-B-2: outdoor_suburban CONTROL start >> %LOG%
%PYTHON% %SCRIPT% --env outdoor_suburban --replicates 600 --seed 92001 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 14 --run-tier publication --tx-power 10.0 --max-cpu-percent 88 --max-mem-percent 82 --output %OUTDIR%\scalability_outdoor_suburban_server_s9_control_20260216.json >> %LOG% 2>&1
echo [%date% %time%] S9-B-2: outdoor_suburban CONTROL done (exit=%errorlevel%) >> %LOG%

echo [%date% %time%] S9 BUNDLE ALL DONE >> %LOG%

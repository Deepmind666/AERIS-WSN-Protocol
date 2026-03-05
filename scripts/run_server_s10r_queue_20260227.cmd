@echo off
REM S10R Server Queue - outdoor_urban + outdoor_suburban (6 tasks serial)
REM Launched by Claude session, 2026-02-27
REM CPU budget: 90%%, 9h window

set PYTHON=C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe
set WORKDIR=C:\Users\sshuser\AERIS-WSN-Protocol
set OUTDIR=%WORKDIR%\results\mega_experiments
set LOGFILE=%WORKDIR%\logs\s10r_server_cmd_queue_20260227.log

cd /d %WORKDIR%
if not exist logs mkdir logs
if not exist %OUTDIR% mkdir %OUTDIR%

echo [%date% %time%] === S10R Server CMD Queue START === >> %LOGFILE%

REM Task 1: outdoor_urban tx5
set OUT1=%OUTDIR%\scalability_outdoor_urban_server_s10r_tx5_20260227.json
if exist %OUT1% (
    echo [%date% %time%] SKIP task1 outdoor_urban tx5 - already exists >> %LOGFILE%
) else (
    echo [%date% %time%] START task1 outdoor_urban tx5 >> %LOGFILE%
    %PYTHON% scripts/run_scalability_experiment.py --env outdoor_urban --tx-power 5 --replicates 1000 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 20 --run-tier publication --mac-collision --multihop-relay --max-cpu-percent 90 --max-mem-percent 96 --allow-partial --output %OUT1% >> %LOGFILE% 2>&1
    echo [%date% %time%] DONE task1 outdoor_urban tx5 exit=%ERRORLEVEL% >> %LOGFILE%
)

REM Task 2: outdoor_urban tx10
set OUT2=%OUTDIR%\scalability_outdoor_urban_server_s10r_tx10_20260227.json
if exist %OUT2% (
    echo [%date% %time%] SKIP task2 outdoor_urban tx10 - already exists >> %LOGFILE%
) else (
    echo [%date% %time%] START task2 outdoor_urban tx10 >> %LOGFILE%
    %PYTHON% scripts/run_scalability_experiment.py --env outdoor_urban --tx-power 10 --replicates 1000 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 20 --run-tier publication --mac-collision --multihop-relay --max-cpu-percent 90 --max-mem-percent 96 --allow-partial --output %OUT2% >> %LOGFILE% 2>&1
    echo [%date% %time%] DONE task2 outdoor_urban tx10 exit=%ERRORLEVEL% >> %LOGFILE%
)

REM Task 3: outdoor_urban tx15
set OUT3=%OUTDIR%\scalability_outdoor_urban_server_s10r_tx15_20260227.json
if exist %OUT3% (
    echo [%date% %time%] SKIP task3 outdoor_urban tx15 - already exists >> %LOGFILE%
) else (
    echo [%date% %time%] START task3 outdoor_urban tx15 >> %LOGFILE%
    %PYTHON% scripts/run_scalability_experiment.py --env outdoor_urban --tx-power 15 --replicates 1000 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 20 --run-tier publication --mac-collision --multihop-relay --max-cpu-percent 90 --max-mem-percent 96 --allow-partial --output %OUT3% >> %LOGFILE% 2>&1
    echo [%date% %time%] DONE task3 outdoor_urban tx15 exit=%ERRORLEVEL% >> %LOGFILE%
)

REM Task 4: outdoor_suburban tx5
set OUT4=%OUTDIR%\scalability_outdoor_suburban_server_s10r_tx5_20260227.json
if exist %OUT4% (
    echo [%date% %time%] SKIP task4 outdoor_suburban tx5 - already exists >> %LOGFILE%
) else (
    echo [%date% %time%] START task4 outdoor_suburban tx5 >> %LOGFILE%
    %PYTHON% scripts/run_scalability_experiment.py --env outdoor_suburban --tx-power 5 --replicates 1000 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 20 --run-tier publication --mac-collision --multihop-relay --max-cpu-percent 90 --max-mem-percent 96 --allow-partial --output %OUT4% >> %LOGFILE% 2>&1
    echo [%date% %time%] DONE task4 outdoor_suburban tx5 exit=%ERRORLEVEL% >> %LOGFILE%
)

REM Task 5: outdoor_suburban tx10
set OUT5=%OUTDIR%\scalability_outdoor_suburban_server_s10r_tx10_20260227.json
if exist %OUT5% (
    echo [%date% %time%] SKIP task5 outdoor_suburban tx10 - already exists >> %LOGFILE%
) else (
    echo [%date% %time%] START task5 outdoor_suburban tx10 >> %LOGFILE%
    %PYTHON% scripts/run_scalability_experiment.py --env outdoor_suburban --tx-power 10 --replicates 1000 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 20 --run-tier publication --mac-collision --multihop-relay --max-cpu-percent 90 --max-mem-percent 96 --allow-partial --output %OUT5% >> %LOGFILE% 2>&1
    echo [%date% %time%] DONE task5 outdoor_suburban tx10 exit=%ERRORLEVEL% >> %LOGFILE%
)

REM Task 6: outdoor_suburban tx15
set OUT6=%OUTDIR%\scalability_outdoor_suburban_server_s10r_tx15_20260227.json
if exist %OUT6% (
    echo [%date% %time%] SKIP task6 outdoor_suburban tx15 - already exists >> %LOGFILE%
) else (
    echo [%date% %time%] START task6 outdoor_suburban tx15 >> %LOGFILE%
    %PYTHON% scripts/run_scalability_experiment.py --env outdoor_suburban --tx-power 15 --replicates 1000 --nodes 100,200,300,500,800,1000 --rounds 300 --workers 20 --run-tier publication --mac-collision --multihop-relay --max-cpu-percent 90 --max-mem-percent 96 --allow-partial --output %OUT6% >> %LOGFILE% 2>&1
    echo [%date% %time%] DONE task6 outdoor_suburban tx15 exit=%ERRORLEVEL% >> %LOGFILE%
)

echo [%date% %time%] === S10R Server CMD Queue END === >> %LOGFILE%

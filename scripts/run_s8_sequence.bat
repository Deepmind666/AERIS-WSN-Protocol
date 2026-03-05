@echo off
cd /d C:\Users\sshuser\AERIS-WSN-Protocol

echo [S8] Starting outdoor_urban (workers=20, map-fix) at %date% %time% >> C:\Users\sshuser\s8_sequence.log

C:\Python314\python.exe -u scripts\run_scalability_experiment.py --replicates 1000 --workers 20 --seed 42001 --nodes 100,200,300,500,800,1000 --rounds 300 --env outdoor_urban --tx-power 10.0 --run-tier publication --max-cpu-percent 100 --max-mem-percent 99 --resource-check-sec 2 --output results\mega_experiments\scalability_outdoor_urban_server_s8_20260213.json > C:\Users\sshuser\s8_urban_stdout.log 2>&1

echo [S8] outdoor_urban finished at %date% %time%, exit code %errorlevel% >> C:\Users\sshuser\s8_sequence.log

echo [S8] Starting outdoor_suburban (workers=20, map-fix) at %date% %time% >> C:\Users\sshuser\s8_sequence.log

C:\Python314\python.exe -u scripts\run_scalability_experiment.py --replicates 1000 --workers 20 --seed 42001 --nodes 100,200,300,500,800,1000 --rounds 300 --env outdoor_suburban --tx-power 10.0 --run-tier publication --max-cpu-percent 100 --max-mem-percent 99 --resource-check-sec 2 --output results\mega_experiments\scalability_outdoor_suburban_server_s8_20260213.json > C:\Users\sshuser\s8_suburban_stdout.log 2>&1

echo [S8] outdoor_suburban finished at %date% %time%, exit code %errorlevel% >> C:\Users\sshuser\s8_sequence.log
echo [S8] ALL DONE at %date% %time% >> C:\Users\sshuser\s8_sequence.log

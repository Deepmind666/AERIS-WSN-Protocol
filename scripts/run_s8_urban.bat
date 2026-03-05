@echo off
cd /d C:\Users\sshuser\AERIS-WSN-Protocol
C:\Python314\python.exe -u scripts\run_scalability_experiment.py --replicates 1000 --workers 12 --seed 42001 --nodes 100,200,300,500,800,1000 --rounds 300 --env outdoor_urban --tx-power 10.0 --run-tier publication --max-cpu-percent 90 --max-mem-percent 85 --resource-check-sec 5 --output results\mega_experiments\scalability_outdoor_urban_server_s8_20260213.json > C:\Users\sshuser\s8_urban_stdout.log 2>&1

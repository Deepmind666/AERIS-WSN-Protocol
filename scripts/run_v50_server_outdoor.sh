#!/bin/bash
# v50-rigor full experiment: outdoor_urban + outdoor_suburban on FatMachine
# 21 workers, 90% CPU/MEM, 1000 replicates per config

PYTHON="C:/Users/sshuser/miniconda3/envs/aether-wsn/python.exe"
SCRIPT="C:/Users/sshuser/AERIS-WSN/scripts/run_scalability_experiment.py"
OUTDIR="C:/Users/sshuser/AERIS-WSN/results/mega_experiments"

echo "[$(date)] Starting outdoor_urban..."
"$PYTHON" "$SCRIPT" \
  --env outdoor_urban --replicates 1000 --seed 42001 --rounds 300 \
  --tx-power 10.0 --workers 21 --max-cpu-percent 90 --max-mem-percent 90 \
  --mac-collision --multihop-relay --run-tier publication \
  --output "$OUTDIR/v50_rigor_outdoor_urban.json" --allow-partial \
  2>&1 | tee "$OUTDIR/v50_rigor_outdoor_urban.log"
echo "[$(date)] outdoor_urban done, exit=$?"

echo "[$(date)] Starting outdoor_suburban..."
"$PYTHON" "$SCRIPT" \
  --env outdoor_suburban --replicates 1000 --seed 42001 --rounds 300 \
  --tx-power 10.0 --workers 21 --max-cpu-percent 90 --max-mem-percent 90 \
  --mac-collision --multihop-relay --run-tier publication \
  --output "$OUTDIR/v50_rigor_outdoor_suburban.json" --allow-partial \
  2>&1 | tee "$OUTDIR/v50_rigor_outdoor_suburban.log"
echo "[$(date)] outdoor_suburban done, exit=$?"

echo "[$(date)] ALL DONE"

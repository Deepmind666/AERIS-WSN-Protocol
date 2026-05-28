#!/bin/bash
cd /home/ns3user/ns-allinone-3.40/ns-3.40
OUTFILE="/home/ns3user/ns3_5proto_multienv_$(date +%Y%m%d).json"
LOGFILE="/home/ns3user/ns3_5proto_run.log"
echo "=== NS-3 5-Protocol Full Matrix Experiment ===" | tee "$LOGFILE"
echo "Start: $(date)" | tee -a "$LOGFILE"
echo "Output: $OUTFILE" | tee -a "$LOGFILE"
echo "Expected: 5 protocols x 7 node counts x 30 seeds x 4 envs = 4200 main + 480 ablation = 4680 total" | tee -a "$LOGFILE"
ulimit -v 16000000
./ns3 run "scratch/aeris-validation-standalone --runMultiEnv --output=$OUTFILE" 2>&1 | tee -a "$LOGFILE"
echo "End: $(date)" | tee -a "$LOGFILE"
echo "=== Done ===" | tee -a "$LOGFILE"

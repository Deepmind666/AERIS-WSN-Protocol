#!/bin/bash
cd /home/ns3user/ns-allinone-3.40/ns-3.40
OUTFILE="/home/ns3user/ns3_5proto_multienv_20260215.json"
LOGFILE="/home/ns3user/ns3_5proto_run.log"
echo "=== NS-3 5-Protocol Full Matrix ===" > "$LOGFILE"
echo "Start: $(date)" >> "$LOGFILE"
echo "Output: $OUTFILE" >> "$LOGFILE"
./ns3 run "scratch/aeris-validation-standalone --runMultiEnv --output=$OUTFILE" >> "$LOGFILE" 2>&1
echo "End: $(date)" >> "$LOGFILE"
echo "=== Done ===" >> "$LOGFILE"

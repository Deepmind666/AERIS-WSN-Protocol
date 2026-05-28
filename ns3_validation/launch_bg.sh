#!/bin/bash
# Launcher: starts the full experiment in background via nohup
sed -i 's/\r$//' /mnt/c/tmp/run_5proto_full.sh
nohup bash /mnt/c/tmp/run_5proto_full.sh > /home/ns3user/ns3_5proto_nohup.log 2>&1 &
echo "PID=$!"
echo "Log: /home/ns3user/ns3_5proto_nohup.log"
echo "Run log: /home/ns3user/ns3_5proto_run.log"
sleep 2
head -5 /home/ns3user/ns3_5proto_nohup.log 2>/dev/null || echo "(log not yet written)"

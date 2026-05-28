#!/bin/bash
# Daemonize: use setsid to fully detach from terminal
sed -i 's/\r$//' /mnt/c/tmp/run_5proto_bg.sh
setsid bash /mnt/c/tmp/run_5proto_bg.sh &
disown
echo "Launched with setsid, PID=$!"
sleep 3
wc -l /home/ns3user/ns3_5proto_run.log
tail -5 /home/ns3user/ns3_5proto_run.log

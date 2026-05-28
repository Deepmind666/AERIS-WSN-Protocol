#!/bin/bash
echo "=== Process check ==="
ps aux | grep -E "ns3|aeris" | grep -v grep
echo "=== Log lines ==="
wc -l /home/ns3user/ns3_5proto_run.log
echo "=== Log tail ==="
tail -10 /home/ns3user/ns3_5proto_run.log
echo "=== Memory ==="
free -h | head -3

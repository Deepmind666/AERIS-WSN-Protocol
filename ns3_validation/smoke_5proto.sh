#!/bin/bash
cd /home/ns3user/ns-allinone-3.40/ns-3.40
echo "=== NS-3 5-protocol smoke test ==="
./ns3 run "scratch/aeris-validation-standalone --numNodes=50 --numRounds=30 --seed=42001 --output=/home/ns3user/ns3_5proto_smoke.json"
echo "=== Smoke test done ==="
cat /home/ns3user/ns3_5proto_smoke.json

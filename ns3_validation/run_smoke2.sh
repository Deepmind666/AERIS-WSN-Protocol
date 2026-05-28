#!/bin/bash
cd /home/ns3user/ns-allinone-3.40/ns-3.40
echo "=== Smoke Test: single seed, default 100 nodes ==="
./ns3 run "aeris-validation-standalone --numNodes=100 --numRounds=300 --seed=42001" 2>&1

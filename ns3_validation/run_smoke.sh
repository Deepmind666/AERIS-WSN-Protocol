#!/bin/bash
cd /home/ns3user/ns-allinone-3.40/ns-3.40
echo "=== Smoke Test: indoor_office (INDOOR_LOS) ==="
./ns3 run "aeris-validation-standalone --seed=42001 --nodes=100 --rounds=300 --env=INDOOR_LOS" 2>&1
echo ""
echo "=== Smoke Test: indoor_factory (INDUSTRIAL) ==="
./ns3 run "aeris-validation-standalone --seed=42001 --nodes=100 --rounds=300 --env=INDUSTRIAL" 2>&1

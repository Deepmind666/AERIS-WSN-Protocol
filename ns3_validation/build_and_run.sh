#!/bin/bash
set -e
cd /home/ns3user/ns-allinone-3.40/ns-3.40
echo "=== Building NS-3 5-protocol validation ==="
./ns3 build scratch/aeris-validation-standalone
echo "=== Build complete ==="

#!/bin/bash
set -euo pipefail

SRC="/mnt/c/Users/sshuser/AERIS-WSN-Protocol/ns3_validation/aeris-validation-standalone.cc"
DST="/home/ns3user/ns-allinone-3.40/ns-3.40/scratch/aeris-validation-standalone.cc"
NS3_ROOT="/home/ns3user/ns-allinone-3.40/ns-3.40"

cp "$SRC" "$DST"
echo "[SERVER-BUILD] synced scratch source:"
md5sum "$DST"

cd "$NS3_ROOT"
./ns3 build

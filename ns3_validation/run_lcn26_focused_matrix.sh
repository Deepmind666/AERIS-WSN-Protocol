#!/bin/bash
set -euo pipefail

# Focused LCN26 NS-3 audit matrix.
# Defaults to 5 protocols x 4 envs x 3 node counts (100/500/1000) x 30 seeds.
# Override PROTOCOLS, ENVS, or NODES from the caller for split or expanded runs.

NS3_ROOT="${NS3_ROOT:-/home/ns3user/ns-allinone-3.40/ns-3.40}"
BIN="${BIN:-$NS3_ROOT/build/scratch/ns3.40-aeris-validation-standalone-default}"
PARALLEL="${1:-6}"
OUT_DIR="${2:-/mnt/c/Users/sshuser/AERIS-WSN-Protocol/ns3_validation/results/lcn26_ns3_audit_$(date +%Y%m%d_%H%M%S)}"
NODES="${NODES:-100,500,1000}"

mkdir -p "$OUT_DIR/raw" "$OUT_DIR/logs"

if [ ! -f "$BIN" ]; then
  echo "ERROR: NS-3 binary not found at $BIN"
  exit 1
fi

export LD_LIBRARY_PATH="$NS3_ROOT/build/lib:${LD_LIBRARY_PATH:-}"
if [ -n "${PROTOCOLS:-}" ]; then
  read -r -a protocols <<< "$PROTOCOLS"
else
  protocols=(AERIS LEACH HEED PEGASIS TEEN)
fi
if [ -n "${ENVS:-}" ]; then
  read -r -a envs <<< "$ENVS"
else
  envs=(indoor_office indoor_factory outdoor_urban outdoor_suburban)
fi

MASTER_LOG="$OUT_DIR/logs/master.log"
{
  echo "=== LCN26 focused NS-3 audit matrix ==="
  echo "Start: $(date)"
  echo "NS3_ROOT: $NS3_ROOT"
  echo "BIN: $BIN"
  echo "OUT_DIR: $OUT_DIR"
  echo "PARALLEL: $PARALLEL"
  echo "NODES: $NODES"
  echo "Protocols: ${protocols[*]}"
  echo "Environments: ${envs[*]}"
  echo "Expected shards: $((${#protocols[@]} * ${#envs[@]}))"
  node_count="$(awk -F',' '{print NF}' <<< "$NODES")"
  echo "Expected experiments: $((${#protocols[@]} * ${#envs[@]} * node_count * 30))"
} | tee "$MASTER_LOG"

run_one() {
  local proto="$1"
  local env="$2"
  local out="$OUT_DIR/raw/shard_${proto}_${env}.json"
  local log="$OUT_DIR/logs/${proto}_${env}.log"
  echo "[START] $(date '+%F %T') proto=$proto env=$env out=$out" | tee -a "$MASTER_LOG"
  "$BIN" --runShard --protocol="$proto" --env="$env" --nodes="$NODES" --output="$out" > "$log" 2>&1
  echo "[DONE ] $(date '+%F %T') proto=$proto env=$env" | tee -a "$MASTER_LOG"
}

active=0
for env in "${envs[@]}"; do
  for proto in "${protocols[@]}"; do
    run_one "$proto" "$env" &
    active=$((active + 1))
    if [ "$active" -ge "$PARALLEL" ]; then
      wait -n
      active=$((active - 1))
    fi
  done
done
wait

{
  echo "End: $(date)"
  echo "=== Focused NS-3 audit complete ==="
} | tee -a "$MASTER_LOG"

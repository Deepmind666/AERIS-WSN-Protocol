#!/bin/bash
# Self-contained: run NS-3 experiment directly (no ./ns3 wrapper)
# This avoids the Python ns3 CLI and runs the compiled binary directly
cd /home/ns3user/ns-allinone-3.40/ns-3.40
BINARY="./build/scratch/ns3.40-aeris-validation-standalone-default"
OUTFILE="/home/ns3user/ns3_5proto_multienv_20260215.json"
LOGFILE="/home/ns3user/ns3_5proto_run.log"

if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found at $BINARY" > "$LOGFILE"
    exit 1
fi

echo "=== NS-3 5-Protocol Full Matrix ===" > "$LOGFILE"
echo "Start: $(date)" >> "$LOGFILE"
echo "Binary: $BINARY" >> "$LOGFILE"
echo "Output: $OUTFILE" >> "$LOGFILE"

# Run the binary directly (no Python ns3 wrapper needed)
export LD_LIBRARY_PATH="/home/ns3user/ns-allinone-3.40/ns-3.40/build/lib:$LD_LIBRARY_PATH"
"$BINARY" --runMultiEnv --output="$OUTFILE" >> "$LOGFILE" 2>&1

echo "End: $(date)" >> "$LOGFILE"
echo "=== Done ===" >> "$LOGFILE"

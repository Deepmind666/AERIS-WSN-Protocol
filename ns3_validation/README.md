# NS-3 Validation Module for AERIS Protocol
## Cross-Platform Verification using Industry-Standard Simulator

---

## Overview

This module provides NS-3 implementation of the AERIS (Adaptive Environment-aware Routing for IoT Sensors) protocol for cross-validation against the Python simulation results.

### Why NS-3?

NS-3 is a discrete-event network simulator used extensively in academic research:
- **IEEE/ACM endorsed** simulation platform
- **Widely accepted** in top-tier conferences (INFOCOM, MobiCom, etc.)
- **Validated models** for IEEE 802.15.4, wireless channels, energy consumption
- **Reproducible** by other researchers

---

## Prerequisites

### 1. Install NS-3

**Windows (WSL2 recommended):**
```bash
# In WSL2 Ubuntu
sudo apt update
sudo apt install -y g++ python3 python3-dev python3-pip cmake ninja-build
sudo apt install -y libgsl-dev libsqlite3-dev

# Download NS-3
cd ~
wget https://www.nsnam.org/releases/ns-allinone-3.40.tar.bz2
tar xjf ns-allinone-3.40.tar.bz2
cd ns-allinone-3.40/ns-3.40

# Build NS-3
./ns3 configure --enable-examples --enable-tests
./ns3 build
```

**Linux:**
```bash
sudo apt install -y g++ python3 python3-dev cmake ninja-build
sudo apt install -y libgsl-dev libsqlite3-dev

cd ~
git clone https://gitlab.com/nsnam/ns-3-dev.git ns-3
cd ns-3
./ns3 configure --enable-examples
./ns3 build
```

### 2. Install AERIS Module

```bash
# Copy AERIS module to NS-3 contrib directory
cp -r ns3_validation/src/aeris ~/ns-3/contrib/

# Rebuild NS-3
cd ~/ns-3
./ns3 configure
./ns3 build
```

---

## Module Structure

```
ns3_validation/
├── README.md                    # This file
├── src/
│   └── aeris/                   # NS-3 module
│       ├── model/
│       │   ├── aeris-protocol.h
│       │   ├── aeris-protocol.cc
│       │   ├── aeris-helper.h
│       │   └── aeris-helper.cc
│       ├── helper/
│       ├── examples/
│       │   └── aeris-example.cc
│       └── wscript              # NS-3 build script
├── scripts/
│   ├── run_ns3_experiments.py   # Experiment runner
│   └── compare_results.py       # Cross-validation analysis
└── results/
    └── ...                      # NS-3 experiment results
```

---

## Running Validation Experiments

### Step 1: Run NS-3 Simulations

```bash
cd ~/ns-3
./ns3 run "aeris-example --numNodes=100 --simTime=200 --seed=42001"
```

### Step 2: Compare with Python Results

```bash
cd /path/to/AERIS-WSN-Protocol
python ns3_validation/scripts/compare_results.py
```

---

## Validation Metrics

The cross-validation compares:

| Metric | Description |
|--------|-------------|
| PDR | Packet Delivery Ratio (end-to-end) |
| Energy | Total energy consumption |
| Lifetime | Network lifetime (rounds until first death) |
| Latency | Average packet latency |

### Acceptance Criteria

- PDR difference < 5%
- Energy difference < 10%
- Trend consistency across scales

---

## Expected Results

Based on our Python simulations:

| Protocol | Python PDR | Expected NS-3 PDR |
|----------|------------|-------------------|
| AERIS | 100% | 98-100% |
| LEACH | 86.9% | 85-88% |
| PEGASIS | 97.9% | 96-99% |
| HEED | 87.1% | 85-88% |

---

## Citation

If you use this NS-3 module, please cite:

```bibtex
@article{aeris2026,
  title={AERIS: Adaptive Environment-aware Routing for IoT Sensors},
  author={AERIS Research Team},
  journal={MDPI Sensors},
  year={2026}
}
```

---

## Contact

For questions about the NS-3 implementation, please open an issue on GitHub.

#!/usr/bin/env python3
"""Launch indoor_factory 800-node experiment as a detached local process."""
import subprocess, sys, os

PYTHON = r"C:\Users\admin\anaconda3\envs\aether-wsn\python.exe"
CWD = r"C:\AERIS-WSN-Protocol"
STDOUT = r"C:\AERIS-WSN-Protocol\results\mega_experiments\local_800_stdout.txt"
STDERR = r"C:\AERIS-WSN-Protocol\results\mega_experiments\local_800_stderr.txt"

args = [
    PYTHON, "scripts/run_scalability_experiment.py",
    "--env", "indoor_factory",
    "--replicates", "1000",
    "--seed", "42001",
    "--nodes", "800",
    "--rounds", "300",
    "--workers", "6",
    "--run-tier", "publication",
    "--tx-power", "10.0",
    "--max-cpu-percent", "80",
    "--max-mem-percent", "75",
    "--allow-partial",
    "--output", "results/mega_experiments/scalability_indoor_factory_local_800_s8.json",
]

flags = 0x00000008 | 0x00000200 | 0x01000000

with open(STDOUT, "w") as fout, open(STDERR, "w") as ferr:
    proc = subprocess.Popen(
        args,
        cwd=CWD,
        stdout=fout,
        stderr=ferr,
        creationflags=flags,
        close_fds=True,
    )

print(f"PID={proc.pid}")
print("Process launched in detached mode.")

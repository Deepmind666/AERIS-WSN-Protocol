#!/usr/bin/env python3
"""Launch a detached subprocess on Windows that survives SSH disconnect."""
import subprocess, sys, os

PYTHON = r"C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe"
CWD = r"C:\Users\sshuser\AERIS-WSN-Protocol"
STDOUT = r"C:\Users\sshuser\factory_stdout.txt"
STDERR = r"C:\Users\sshuser\factory_stderr.txt"

args = [
    PYTHON, "scripts/run_scalability_experiment.py",
    "--env", "indoor_factory",
    "--replicates", "1000",
    "--seed", "42001",
    "--nodes", "800,1000",
    "--rounds", "300",
    "--workers", "8",
    "--run-tier", "publication",
    "--tx-power", "10.0",
    "--max-cpu-percent", "95",
    "--max-mem-percent", "85",
    "--allow-partial",
    "--output", "results/mega_experiments/scalability_indoor_factory_server_large_s8.json",
]

# DETACHED_PROCESS=0x8 + CREATE_NEW_PROCESS_GROUP=0x200 + CREATE_BREAKAWAY_FROM_JOB=0x01000000
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

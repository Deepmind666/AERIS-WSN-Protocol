#!/usr/bin/env python3
"""
Build missing provenance sidecars for local v50-rigor authoritative JSON files.

This script mirrors the sidecar schema used by server files so all four
environment files can be reconciled with one consistent metadata format.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_SCRIPT = PROJECT_ROOT / "scripts" / "run_scalability_experiment.py"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out
    except Exception:
        return "unknown"


def build_sidecar(source_file: Path, env_name: str, workers: int) -> dict:
    data = json.loads(source_file.read_text(encoding="utf-8"))
    data_sha = sha256_file(source_file)
    script_sha = sha256_file(RUN_SCRIPT)

    config = {
        "env": env_name,
        "replicates": 3200,
        "seed": 42001,
        "nodes": [100, 200, 300, 500, 800, 1000],
        "rounds": 300,
        "workers": workers,
        "tx_power": 10.0,
        "mac_collision": True,
        "multihop_relay": True,
        "allow_partial": True,
        "max_cpu_percent": 88,
        "max_mem_percent": 88,
    }

    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    sidecar = {
        "source_file": str(source_file).replace("\\", "/"),
        "data_sha256": data_sha,
        "script": "scripts/run_scalability_experiment.py",
        "script_sha256": script_sha,
        "git_commit": data.get("git_commit", "unknown"),
        "git_branch": get_git_branch(),
        "run_tier": data.get("run_tier", "publication"),
        "primary_metric": data.get("primary_metric", "pdr_expected"),
        "config": config,
        "config_hash": config_hash,
        "execution": {
            "host": platform.node() or "local",
            "python": os.path.realpath(os.sys.executable),
            "total_runs": len(data.get("raw_results", [])),
            "error_runs": data.get("error_runs", 0),
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "note": "Generated locally for v50-rigor queue outputs (workers=12, cpu/mem caps=88).",
    }
    return sidecar


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local v50-rigor sidecars.")
    parser.add_argument("--office", required=True, type=Path)
    parser.add_argument("--suburban", required=True, type=Path)
    args = parser.parse_args()

    targets = [
        (args.office, "indoor_office", 12),
        (args.suburban, "outdoor_suburban", 12),
    ]

    for src, env_name, workers in targets:
        if not src.exists():
            raise FileNotFoundError(f"Missing source JSON: {src}")
        sidecar = build_sidecar(src, env_name, workers)
        out = src.with_suffix(".provenance.json")
        out.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()

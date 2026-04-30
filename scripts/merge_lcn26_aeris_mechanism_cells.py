#!/usr/bin/env python3
"""Merge per-cell AERIS mechanism runs into one publication block."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from run_lcn26_aeris_mechanism_matrix import (
    OUTPUT_VERSION,
    aggregate,
    get_git_commit,
    get_git_diff_stat,
    get_git_dirty,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(input_root.glob("**/mechanism_raw.json"))
    if not raw_files:
        raise SystemExit(f"No mechanism_raw.json found under {input_root}")

    all_runs = []
    configs = []
    error_runs = 0
    for raw_path in raw_files:
        data = json.loads(raw_path.read_text(encoding="utf-8"))
        configs.append(data.get("config", {}))
        error_runs += int(data.get("error_runs", 0))
        all_runs.extend(data.get("raw_results", []))

    first_cfg = configs[0] if configs else {}
    merged = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "git_diff_stat": get_git_diff_stat(),
        "experiment_type": "lcn26_aeris_mechanism_grid_merged",
        "run_tier": "publication",
        "primary_metric": "pdr_expected",
        "output_version": OUTPUT_VERSION,
        "environment": "multiple",
        "error_runs": error_runs,
        "incomplete_runs": 0,
        "config": {
            "seed_base": first_cfg.get("seed_base"),
            "replicates": first_cfg.get("replicates"),
            "environments": sorted({run["environment"] for run in all_runs}),
            "node_counts": sorted({int(run["num_nodes"]) for run in all_runs}),
            "round_counts": first_cfg.get("round_counts", []),
            "area_size": first_cfg.get("area_size"),
            "base_station": first_cfg.get("base_station"),
            "mac_collision": bool(first_cfg.get("mac_collision", False)),
            "source_files": [str(p) for p in raw_files],
        },
        "raw_results": all_runs,
        "summary": aggregate(all_runs),
    }

    out_json = output_dir / "mechanism_raw_merged.json"
    out_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"[LCN26-MECH] wrote {out_json}")


if __name__ == "__main__":
    main()

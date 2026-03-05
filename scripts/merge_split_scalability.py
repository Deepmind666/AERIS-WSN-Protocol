#!/usr/bin/env python3
"""Merge two split scalability JSON files (small nodes + large nodes) into one."""

import json
import sys
from datetime import datetime
from pathlib import Path

def main():
    if len(sys.argv) != 4:
        print("Usage: merge_split_scalability.py <small.json> <large.json> <output.json>")
        sys.exit(1)

    small_path, large_path, out_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    small = json.loads(small_path.read_text(encoding="utf-8"))
    large = json.loads(large_path.read_text(encoding="utf-8"))

    # Validate compatible config
    for key in ("run_tier", "primary_metric", "environment", "tx_power_dbm"):
        assert small[key] == large[key], f"Mismatch on {key}: {small[key]} vs {large[key]}"
    for key in ("round_counts", "dropout_rates", "protocols", "area_size",
                "base_station", "packet_size", "initial_energy"):
        assert small["config"][key] == large["config"][key], f"Config mismatch on {key}"

    merged_raw = small["raw_results"] + large["raw_results"]
    merged_node_counts = sorted(set(small["config"]["node_counts"] + large["config"]["node_counts"]))
    merged_seeds = sorted(set(small["config"]["seeds"] + large["config"]["seeds"]))
    failed = sum(1 for r in merged_raw if not r.get("success", True))

    out = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "git_commit": small["git_commit"],
        "git_dirty": small.get("git_dirty", True) or large.get("git_dirty", True),
        "git_diff_stat": small.get("git_diff_stat", {}),
        "experiment_type": "scalability",
        "run_tier": small["run_tier"],
        "primary_metric": small["primary_metric"],
        "environment": small["environment"],
        "tx_power_dbm": small["tx_power_dbm"],
        "max_cpu_percent": max(small.get("max_cpu_percent", 0), large.get("max_cpu_percent", 0)),
        "max_mem_percent": max(small.get("max_mem_percent", 0), large.get("max_mem_percent", 0)),
        "workers_requested": f"local={small.get('workers_requested')},server={large.get('workers_requested')}",
        "workers_effective": f"local={small.get('workers_effective')},server={large.get('workers_effective')}",
        "error_runs": failed,
        "incomplete_runs": small.get("incomplete_runs", 0) + large.get("incomplete_runs", 0),
        "merge_note": f"Merged from {small_path.name} (nodes {small['config']['node_counts']}) + {large_path.name} (nodes {large['config']['node_counts']})",
        "config": {
            "seeds": merged_seeds,
            "node_counts": merged_node_counts,
            "round_counts": small["config"]["round_counts"],
            "dropout_rates": small["config"]["dropout_rates"],
            "force_ctp_reliable": small["config"].get("force_ctp_reliable", False),
            "protocols": small["config"]["protocols"],
            "area_size": small["config"]["area_size"],
            "base_station": small["config"]["base_station"],
            "packet_size": small["config"]["packet_size"],
            "initial_energy": small["config"]["initial_energy"],
            "aeris_profile": small["config"].get("aeris_profile", "energy"),
            "output_version": small["config"].get("output_version"),
        },
        "raw_results": merged_raw,
    }

    # Cell count validation
    from collections import Counter
    cells = Counter((r["num_nodes"], r["protocol"]) for r in merged_raw if not r.get("error"))
    print(f"Merged: {len(merged_raw)} raw_results, error_runs={failed}")
    print(f"Node counts: {merged_node_counts}")
    for k in sorted(cells.keys()):
        print(f"  {k}: n={cells[k]}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

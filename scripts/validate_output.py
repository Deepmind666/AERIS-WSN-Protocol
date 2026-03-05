#!/usr/bin/env python3
"""Validate scalability experiment output JSON."""
import json, sys

path = sys.argv[1]
d = json.load(open(path, encoding="utf-8"))

print(f"raw_results: {len(d.get('raw_results', []))}")
print(f"error_runs: {d.get('error_runs')}")
print(f"wall_time_s: {d.get('wall_time_s')}")
print(f"git_commit: {d.get('git_commit')}")
print(f"git_dirty: {d.get('git_dirty')}")
print(f"run_tier: {d.get('run_tier')}")
print(f"primary_metric: {d.get('primary_metric')}")
print(f"environment: {d.get('environment')}")
print(f"summary_keys: {list(d.get('summary', {}).keys())}")

# Check summary detail
summary = d.get("summary", {})
for nk in sorted(summary.keys(), key=lambda x: int(x)):
    protos = summary[nk]
    for p, v in protos.items():
        print(f"  {nk}/{p}: pdr={v['pdr_mean']:.4f}+/-{v['pdr_std']:.4f} "
              f"energy={v['energy_mean']:.2f} n={v['n']}")

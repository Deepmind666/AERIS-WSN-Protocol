#!/usr/bin/env python3
"""S8 wrapper: bypass execute_tasks deadlock by using simple Pool.map."""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime

# Import everything from the original script
from scripts.run_scalability_experiment import (
    run_task, aggregate, get_git_commit, get_git_dirty, get_git_diff_stat,
    DEFAULT_PROTOCOLS, DEFAULT_AREA_SIZE, DEFAULT_BASE_STATION,
    DEFAULT_PACKET_SIZE, DEFAULT_INITIAL_ENERGY, OUTPUT_VERSION,
)

def run_s8(env, workers, output_path):
    replicates = 1000
    seed = 42001
    node_counts = (100, 200, 300, 500, 800, 1000)
    rounds = 300
    tx_power = 10.0
    protocols = DEFAULT_PROTOCOLS
    area_size = DEFAULT_AREA_SIZE
    base_station = DEFAULT_BASE_STATION

    tasks = []
    for num_nodes in node_counts:
        for rep in range(replicates):
            for protocol in protocols:
                tasks.append((
                    num_nodes, rep, protocol, seed,
                    area_size, base_station, env, tx_power, rounds, False,
                ))

    total = len(tasks)
    print(f"[S8] {env}: {total} tasks, workers={workers}", flush=True)

    runs = []
    failed = 0
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for i, result in enumerate(executor.map(run_task, tasks, chunksize=10), 1):
            runs.append(result)
            if not result.get("success", True):
                failed += 1
            if i % 100 == 0 or i == total:
                elapsed = time.time() - t0
                rate = i / elapsed if elapsed > 0 else 0
                eta = (total - i) / rate if rate > 0 else 0
                print(f"[Scalability] {i}/{total} completed, failed={failed}, "
                      f"rate={rate:.1f}/s, ETA={eta/60:.0f}min", flush=True)

    seeds_used = sorted(set(r["seed"] for r in runs))
    failed_runs = sum(1 for r in runs if not r.get("success", True))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    out = {
        "timestamp": timestamp,
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "git_diff_stat": get_git_diff_stat(),
        "experiment_type": "scalability",
        "run_tier": "publication",
        "primary_metric": "pdr_expected",
        "environment": env,
        "tx_power_dbm": tx_power,
        "max_cpu_percent": 100.0,
        "max_mem_percent": 99.0,
        "workers_requested": workers,
        "workers_effective": workers,
        "error_runs": failed_runs,
        "incomplete_runs": total - len(runs),
        "config": {
            "seeds": seeds_used,
            "node_counts": list(node_counts),
            "round_counts": [rounds],
            "dropout_rates": [0.0],
            "protocols": list(protocols),
            "area_size": area_size,
            "base_station": list(base_station),
            "packet_size": DEFAULT_PACKET_SIZE,
            "initial_energy": DEFAULT_INITIAL_ENERGY,
            "output_version": OUTPUT_VERSION,
        },
        "raw_results": runs,
        "summary": aggregate(runs, node_counts, protocols),
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Wrote {output_path} ({len(runs)} results, failed={failed_runs})", flush=True)


if __name__ == "__main__":
    env = sys.argv[1]
    workers = int(sys.argv[2])
    output = sys.argv[3]
    run_s8(env, workers, output)

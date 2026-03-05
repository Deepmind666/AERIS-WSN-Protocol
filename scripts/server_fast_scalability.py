#!/usr/bin/env python3
"""Fast scalability runner for server shard - no per-task resource gating."""
import argparse, hashlib, json, os, random, subprocess, sys, time
from datetime import datetime
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from benchmark_protocols import (
    NetworkConfig, LEACHProtocol, PEGASISProtocol,
    HEEDProtocolWrapper, TEENProtocolWrapper,
)
from improved_energy_model import ImprovedEnergyModel, HardwarePlatform
from aeris_protocol import AerisProtocol

OUTPUT_VERSION = "v2_1"
PROTOCOLS = ("AERIS", "LEACH", "PEGASIS", "HEED", "TEEN")
NODE_COUNTS = (100, 200, 300, 500, 800, 1000)
AREA = 200.0
BS = (100.0, 200.0)
ENERGY = 2.0
PKT = 1024
ROUNDS = 300
TX = 10.0


def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % (10**9)


def get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL,
        ).decode().strip() or "unknown"
    except Exception:
        return "unknown"


def get_git_dirty() -> bool:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL,
        ).decode()
        return bool(out.strip())
    except Exception:
        return True


def get_git_diff_stat():
    def _s(cmd):
        try:
            return subprocess.check_output(
                cmd, cwd=os.path.dirname(__file__), stderr=subprocess.DEVNULL,
            ).decode(errors="ignore").strip() or "clean"
        except Exception:
            return "unknown"
    return {"unstaged": _s(["git","diff","--shortstat"]),
            "staged": _s(["git","diff","--cached","--shortstat"])}


def gen_pos(seed, n):
    rng = random.Random(seed)
    return [(rng.uniform(5, AREA-5), rng.uniform(5, AREA-5)) for _ in range(n)]


def build_cfg(n, seed, env, tx):
    cfg = NetworkConfig(
        num_nodes=n, area_width=AREA, area_height=AREA,
        base_station_x=BS[0], base_station_y=BS[1],
        initial_energy=ENERGY, packet_size=PKT,
        temperature_c=25.0, humidity_ratio=0.5,
        enable_channel=True, channel_env=env,
        tx_power_dbm=tx, link_retx=1, link_retx_power_step=1.0,
    )
    cfg.positions = gen_pos(seed, n)
    cfg.gateway_k = max(2, int(n / 25))
    cfg.gateway_retry_limit = 1
    cfg.gateway_rescue_direct = True
    cfg.intra_link_retx = 2
    cfg.intra_link_power_step = 1.5
    return cfg


def compute_pdr(res):
    if "pdr_expected" in res and res["pdr_expected"] is not None:
        v = float(res["pdr_expected"])
        if v >= 0: return v
    if "packet_delivery_ratio_end2end" in res and res["packet_delivery_ratio_end2end"] is not None:
        v = float(res["packet_delivery_ratio_end2end"])
        if v >= 0: return v
    add_m = res.get("additional_metrics", {})
    bs = add_m.get("bs_delivered_total", 0)
    src = add_m.get("source_packets_total", 0)
    if src > 0: return float(bs) / float(src)
    bs = res.get("bs_delivered", 0)
    se = res.get("source_packets_expected", 0)
    if se > 0: return float(bs) / float(se)
    return -1.0


def run_proto(proto, cfg, seed, rounds):
    import contextlib
    random.seed(seed)
    np.random.seed(seed)
    c = deepcopy(cfg)
    em = ImprovedEnergyModel(HardwarePlatform.CC2420_TELOSB)
    with open(os.devnull, "w") as dn:
        with contextlib.redirect_stdout(dn), contextlib.redirect_stderr(dn):
            if proto == "LEACH": res = LEACHProtocol(c, em).run_simulation(rounds)
            elif proto == "PEGASIS": res = PEGASISProtocol(c, em).run_simulation(rounds)
            elif proto == "HEED": res = HEEDProtocolWrapper(c, em).run_simulation(rounds)
            elif proto == "TEEN": res = TEENProtocolWrapper(c, em).run_simulation(rounds)
            elif proto == "AERIS":
                res = AerisProtocol(c, enable_cas=True, enable_fairness=True,
                    enable_gateway=True, enable_skeleton=True,
                    profile="energy", verbose=False, seed=seed,
                ).run_simulation(rounds)
            else: raise ValueError(proto)
    return {
        "pdr_expected": compute_pdr(res),
        "energy": float(res.get("total_energy_consumed", 0.0)),
        "lifetime": int(res.get("network_lifetime", 0)),
        "alive_nodes": int(res.get("final_alive_nodes", res.get("alive_nodes", 0))),
        "total_rounds": int(res.get("total_rounds", rounds)),
    }


def run_task(args):
    """Top-level wrapper for ProcessPoolExecutor (must be picklable)."""
    proto, n, seed, env, tx, rounds, rep_idx = args
    t0 = time.time()
    try:
        cfg = build_cfg(n, seed, env, tx)
        metrics = run_proto(proto, cfg, seed, rounds)
        return {
            "protocol": proto, "num_nodes": n, "seed": seed,
            "replicate": rep_idx, "env": env,
            "metrics": metrics, "error": None,
            "elapsed_s": round(time.time() - t0, 2),
        }
    except Exception as e:
        return {
            "protocol": proto, "num_nodes": n, "seed": seed,
            "replicate": rep_idx, "env": env,
            "metrics": None, "error": str(e),
            "elapsed_s": round(time.time() - t0, 2),
        }


def aggregate(runs, node_counts, protocols):
    """Aggregate summary by node_count and protocol using valid pdr_expected only."""
    summary = {}
    for n in node_counts:
        summary[n] = {}
        for proto in protocols:
            filtered = [r for r in runs
                        if r["num_nodes"] == n and r["protocol"] == proto]
            valid = [r for r in filtered
                     if r["metrics"] and r["metrics"]["pdr_expected"] >= 0]
            pdrs = [r["metrics"]["pdr_expected"] for r in valid]
            energies = [r["metrics"]["energy"] for r in valid]
            if not pdrs:
                continue
            summary[n][proto] = {
                "pdr_mean": float(np.mean(pdrs)),
                "pdr_std": float(np.std(pdrs, ddof=1)) if len(pdrs) > 1 else 0.0,
                "energy_mean": float(np.mean(energies)),
                "energy_std": float(np.std(energies, ddof=1)) if len(energies) > 1 else 0.0,
                "n": len(pdrs),
            }
    return summary


def main():
    ap = argparse.ArgumentParser(description="Fast scalability runner (no per-task gating)")
    ap.add_argument("--env", required=True, help="Channel environment")
    ap.add_argument("--workers", type=int, default=22)
    ap.add_argument("--replicates", type=int, default=550)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--base-seed", type=int, default=42001)
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument("--run-tier", default="publication")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    git_commit = get_git_commit()
    git_dirty = get_git_dirty()
    git_diff = get_git_diff_stat()

    # Build task list: 5 protocols x 6 node_counts x N replicates
    tasks = []
    seeds_used = set()
    for rep in range(args.replicates):
        for n in NODE_COUNTS:
            for proto in PROTOCOLS:
                seed = args.base_seed + rep * 997 + stable_hash(proto) % 997
                seeds_used.add(seed)
                tasks.append((proto, n, seed, args.env, TX, args.rounds, rep))

    total = len(tasks)
    print(f"[{args.env}] {total} tasks, {args.workers} workers, seed range "
          f"{min(seeds_used)}-{max(seeds_used)}")
    print(f"[{args.env}] git={git_commit} dirty={git_dirty} tier={args.run_tier}")

    # Execute all tasks with direct ProcessPoolExecutor - no per-task gating
    t_start = time.time()
    results = []
    failed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_task, t): i for i, t in enumerate(tasks)}
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done = len(results)
            if r["error"]:
                failed += 1
            if done % 500 == 0 or done == total:
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                print(f"  [{args.env}] {done}/{total} "
                      f"({100*done/total:.1f}%) "
                      f"elapsed={elapsed:.0f}s "
                      f"rate={rate:.1f}/s "
                      f"ETA={eta:.0f}s "
                      f"fail={failed}")

    wall_s = round(time.time() - t_start, 1)
    print(f"[{args.env}] Done: {len(results)} results, "
          f"{failed} errors, {wall_s}s wall time")

    # Aggregate
    summary = aggregate(results, NODE_COUNTS, PROTOCOLS)

    # Build output dict matching run_scalability_experiment.py format
    out = {
        "timestamp": timestamp,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_diff_stat": git_diff,
        "experiment_type": "scalability",
        "run_tier": args.run_tier,
        "primary_metric": "pdr_expected",
        "environment": args.env,
        "tx_power_dbm": TX,
        "workers_requested": args.workers,
        "workers_effective": args.workers,
        "error_runs": failed,
        "wall_time_s": wall_s,
    }
    out["config"] = {
        "seeds": sorted(seeds_used),
        "node_counts": list(NODE_COUNTS),
        "round_counts": [args.rounds],
        "dropout_rates": [0.0],
        "protocols": list(PROTOCOLS),
        "area_size": AREA,
        "base_station": list(BS),
        "initial_energy": ENERGY,
        "packet_size": PKT,
        "replicates": args.replicates,
        "base_seed": args.base_seed,
    }
    out["raw_results"] = results
    out["summary"] = {str(k): v for k, v in summary.items()}

    # Write JSON output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"[{args.env}] Saved: {args.output}")
    print(f"[{args.env}] raw_results={len(results)} summary_keys={list(summary.keys())}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

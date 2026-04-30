#!/usr/bin/env python3
"""Focused AERIS-only mechanism matrix for the LCN 2026 revision."""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from multiprocessing import Pool, freeze_support
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aeris_protocol import AerisProtocol  # noqa: E402
from benchmark_protocols import NetworkConfig  # noqa: E402
from mac_collision_model import MACCollisionConfig, MACCollisionModel  # noqa: E402
from realistic_channel_model import EnvironmentType, RealisticChannelModel  # noqa: E402


DEFAULT_ENVS = ("indoor_office", "indoor_factory", "outdoor_urban", "outdoor_suburban")
DEFAULT_NODES = (100, 500, 1000)
DEFAULT_AREA = 200.0
DEFAULT_BASE_STATION = (100.0, 200.0)
DEFAULT_INITIAL_ENERGY = 2.0
DEFAULT_PACKET_SIZE = 1024
DEFAULT_TX_POWER = 10.0
DEFAULT_ROUNDS = 300
DEFAULT_MAX_CPU = 80.0
DEFAULT_MAX_MEM = 80.0
DEFAULT_CHECK_SEC = 2.0
OUTPUT_VERSION = "lcn26_mech_v1"


def stable_hash(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) % (10**9)


def get_git_commit() -> str:
    try:
        res = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return res.stdout.strip() if res.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def get_git_dirty() -> bool:
    try:
        res = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, timeout=5)
        return bool(res.stdout.strip()) if res.returncode == 0 else True
    except Exception:
        return True


def get_git_diff_stat() -> Dict[str, str]:
    def _short(args: List[str]) -> str:
        try:
            res = subprocess.run(args, capture_output=True, text=True, timeout=5)
            out = res.stdout.strip()
            return out if res.returncode == 0 and out else "clean"
        except Exception:
            return "unknown"

    return {
        "unstaged": _short(["git", "diff", "--shortstat"]),
        "staged": _short(["git", "diff", "--cached", "--shortstat"]),
    }


def get_safe_worker_limit(requested_workers: int, max_cpu_percent: float) -> int:
    cpu_total = max(1, os.cpu_count() or 1)
    cpu_budget_workers = max(1, math.floor(cpu_total * (max_cpu_percent / 100.0)) - 1)
    return max(1, min(requested_workers, cpu_budget_workers))


def has_resource_headroom(max_cpu_percent: float, max_mem_percent: float) -> bool:
    if psutil is None:
        return True
    cpu_now = psutil.cpu_percent(interval=0.25)
    mem_now = psutil.virtual_memory().percent
    return cpu_now <= max_cpu_percent and mem_now <= max_mem_percent


def wait_for_resource_headroom(max_cpu_percent: float, max_mem_percent: float, check_sec: float) -> None:
    if psutil is None:
        return
    attempts = 0
    while True:
        cpu_now = psutil.cpu_percent(interval=0.25)
        mem_now = psutil.virtual_memory().percent
        if cpu_now <= max_cpu_percent and mem_now <= max_mem_percent:
            return
        attempts += 1
        if attempts == 1 or attempts % 10 == 0:
            print(f"[ResourceGuard] Waiting: cpu={cpu_now:.1f}% mem={mem_now:.1f}%")
        time.sleep(max(check_sec, 0.5))


def generate_positions(seed: int, num_nodes: int, area: float) -> List[Tuple[float, float]]:
    rng = random.Random(seed)
    return [(rng.uniform(5.0, area - 5.0), rng.uniform(5.0, area - 5.0)) for _ in range(num_nodes)]


def build_config(num_nodes: int, seed: int, env: str, tx_power: float) -> NetworkConfig:
    cfg = NetworkConfig(
        num_nodes=num_nodes,
        area_width=DEFAULT_AREA,
        area_height=DEFAULT_AREA,
        base_station_x=DEFAULT_BASE_STATION[0],
        base_station_y=DEFAULT_BASE_STATION[1],
        initial_energy=DEFAULT_INITIAL_ENERGY,
        packet_size=DEFAULT_PACKET_SIZE,
        temperature_c=25.0,
        humidity_ratio=0.5,
        enable_channel=True,
        channel_env=env,
        tx_power_dbm=tx_power,
        link_retx=1,
        link_retx_power_step=1.0,
    )
    cfg.force_ctp_reliable = False
    cfg.positions = generate_positions(seed, num_nodes, DEFAULT_AREA)
    cfg.gateway_k = max(2, int(num_nodes / 25))
    cfg.gateway_retry_limit = 1
    cfg.gateway_rescue_direct = True
    cfg.intra_link_retx = 2
    cfg.intra_link_power_step = 1.5
    return cfg


def make_channel(env_name: str, seed: int) -> RealisticChannelModel:
    env_map = {
        "indoor_office": EnvironmentType.INDOOR_OFFICE,
        "indoor_factory": EnvironmentType.INDOOR_FACTORY,
        "outdoor_urban": EnvironmentType.OUTDOOR_URBAN,
        "outdoor_suburban": EnvironmentType.OUTDOOR_SUBURBAN,
    }
    channel = RealisticChannelModel(env_map.get(env_name, EnvironmentType.INDOOR_OFFICE))
    channel.reset_rng(seed)
    return channel


def derive_failure_rounds(round_stats: List[Dict], num_nodes: int) -> Tuple[int, int]:
    fnd = 0
    hnd = 0
    for rs in round_stats:
        alive = int(rs.get("alive_nodes", num_nodes))
        rnd = int(rs.get("round", 0))
        if fnd == 0 and alive < num_nodes:
            fnd = rnd
        if hnd == 0 and alive <= num_nodes / 2:
            hnd = rnd
        if fnd and hnd:
            break
    return fnd, hnd


def run_single(task: Tuple) -> Dict:
    num_nodes, replicate, env, base_seed, rounds, tx_power, enable_mac_collision = task
    seed = base_seed + replicate * 997 + stable_hash(f"{env}:{num_nodes}") % 997

    try:
        cfg = build_config(num_nodes, seed, env, tx_power)
        cfg.external_channel_model = make_channel(env, seed + 17)
        mac_model = MACCollisionModel(MACCollisionConfig(enabled=True)) if enable_mac_collision else None

        proto = AerisProtocol(
            deepcopy(cfg),
            enable_cas=True,
            enable_fairness=True,
            enable_gateway=True,
            enable_skeleton=True,
            profile="energy",
            verbose=False,
            seed=seed + 31,
            mac_collision_model=mac_model,
        )
        if getattr(proto, "force_ctp_reliable", False):
            raise RuntimeError("force_ctp_reliable must be False in publication experiments")

        with open(os.devnull, "w", encoding="utf-8", errors="ignore") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                result = proto.run_simulation(rounds)

        additional = result.get("additional_metrics", {})
        round_stats = result.get("round_statistics", [])
        fnd_round, hnd_round = derive_failure_rounds(round_stats, num_nodes)

        skeleton_backbone_size = 0
        skeleton_assignments = 0
        if hasattr(proto, "skeleton_selector"):
            skeleton_backbone_size = int(getattr(proto.skeleton_selector, "backbone_size", 0))
            skeleton_assignments = int(getattr(proto.skeleton_selector, "total_assignments", 0))

        cas_stats = additional.get("cas_mode_usage_stats", {})
        cas_total = int(sum(int(cas_stats.get(k, 0)) for k in ("DIRECT", "CHAIN", "TWO_HOP")))

        return {
            "environment": env,
            "num_nodes": num_nodes,
            "replicate": replicate,
            "seed": seed,
            "success": True,
            "error": None,
            "metrics": {
                "pdr_expected": float(proto.bs_delivered_total) / float(proto.source_packets_expected) if proto.source_packets_expected > 0 else -1.0,
                "energy": float(result.get("total_energy_consumed", 0.0)),
                "lifetime": int(result.get("network_lifetime", 0)),
                "alive_nodes": int(result.get("final_alive_nodes", 0)),
                "total_rounds": int(len(round_stats) or rounds),
                "first_node_death_round": fnd_round,
                "half_nodes_death_round": hnd_round,
                "avg_hop_count": float(additional.get("avg_hop_count", 0.0)),
                "average_cluster_heads": float(additional.get("average_cluster_heads", 0.0)),
                "cluster_to_ch_pdr_total": float(additional.get("cluster_to_ch_pdr_total", 0.0)),
                "ch_to_bs_pdr_total": float(additional.get("ch_to_bs_pdr_total", 0.0)),
                "cluster_radius_mean_total": float(additional.get("cluster_radius_mean_total", 0.0)),
                "ch_to_bs_distance_mean_total": float(additional.get("ch_to_bs_distance_mean_total", 0.0)),
                "gateway_link_pdr_total": float(additional.get("gateway_link_pdr_total", 0.0)),
                "gateway_uplink_attempts_total": int(additional.get("gateway_uplink_attempts_total", 0)),
                "gateway_uplink_success_total": int(additional.get("gateway_uplink_success_total", 0)),
                "gateway_uplink_pdr_total": float(additional.get("gateway_uplink_pdr_total", 0.0)),
                "gateway_uplink_suppressed_total": int(additional.get("gateway_uplink_suppressed_total", 0)),
                "gateway_concurrency_usage_avg": float(additional.get("gateway_concurrency_usage_avg", 0.0)),
                "gateway_limit_current": int(additional.get("gateway_limit_current", 0) or 0),
                "skeleton_backbone_size": skeleton_backbone_size,
                "skeleton_assignments": skeleton_assignments,
                "cas_total_decisions": cas_total,
                "cas_mode_usage_stats": {
                    "DIRECT": int(cas_stats.get("DIRECT", 0)),
                    "CHAIN": int(cas_stats.get("CHAIN", 0)),
                    "TWO_HOP": int(cas_stats.get("TWO_HOP", 0)),
                    "safety_override": int(cas_stats.get("safety_override", 0)),
                },
                "cas_rule_trigger_counts": additional.get("cas_rule_trigger_counts", {}),
                "cas_score_winner_counts": additional.get("cas_score_winner_counts", {}),
            },
        }
    except Exception as exc:
        return {
            "environment": env,
            "num_nodes": num_nodes,
            "replicate": replicate,
            "seed": seed,
            "success": False,
            "error": str(exc),
            "metrics": {
                "pdr_expected": -1.0,
                "energy": 0.0,
                "lifetime": 0,
                "alive_nodes": 0,
                "total_rounds": 0,
                "first_node_death_round": 0,
                "half_nodes_death_round": 0,
            },
        }


def aggregate(runs: List[Dict]) -> Dict:
    buckets: Dict[Tuple[str, int], List[Dict]] = {}
    for run in runs:
        if not run.get("success"):
            continue
        key = (run["environment"], run["num_nodes"])
        buckets.setdefault(key, []).append(run["metrics"])

    summary: Dict[str, Dict] = {}
    for (env, nodes), metrics_list in sorted(buckets.items()):
        key = f"{env}|{nodes}"
        summary[key] = {"environment": env, "num_nodes": nodes, "n": len(metrics_list)}
        scalar_fields = [
            "pdr_expected", "energy", "lifetime", "alive_nodes", "total_rounds",
            "first_node_death_round", "half_nodes_death_round", "avg_hop_count",
            "average_cluster_heads", "cluster_to_ch_pdr_total", "ch_to_bs_pdr_total",
            "cluster_radius_mean_total", "ch_to_bs_distance_mean_total", "gateway_link_pdr_total",
            "gateway_uplink_attempts_total", "gateway_uplink_success_total", "gateway_uplink_pdr_total",
            "gateway_uplink_suppressed_total", "gateway_concurrency_usage_avg", "gateway_limit_current",
            "skeleton_backbone_size", "skeleton_assignments", "cas_total_decisions",
        ]
        for field in scalar_fields:
            vals = [float(m.get(field, 0.0)) for m in metrics_list]
            summary[key][field] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            }

        cas_keys = ("DIRECT", "CHAIN", "TWO_HOP", "safety_override")
        cas_means = {}
        for ck in cas_keys:
            vals = [int(m.get("cas_mode_usage_stats", {}).get(ck, 0)) for m in metrics_list]
            cas_means[ck] = {"mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0}
        summary[key]["cas_mode_usage_stats"] = cas_means

    return summary


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser(description="Run focused AERIS-only mechanism matrix.")
    parser.add_argument("--envs", type=str, default=",".join(DEFAULT_ENVS))
    parser.add_argument("--nodes", type=str, default=",".join(str(n) for n in DEFAULT_NODES))
    parser.add_argument("--replicates", type=int, default=400)
    parser.add_argument("--seed", type=int, default=52001)
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--tx-power", type=float, default=DEFAULT_TX_POWER)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=240)
    parser.add_argument("--max-cpu-percent", type=float, default=DEFAULT_MAX_CPU)
    parser.add_argument("--max-mem-percent", type=float, default=DEFAULT_MAX_MEM)
    parser.add_argument("--resource-check-sec", type=float, default=DEFAULT_CHECK_SEC)
    parser.add_argument("--mac-collision", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output-root", type=str, default=str(ROOT / "results" / "mega_experiments"))
    args = parser.parse_args()

    envs = tuple(e.strip() for e in args.envs.split(",") if e.strip())
    nodes = tuple(int(n) for n in args.nodes.split(",") if n.strip())
    replicates = min(args.replicates, 8) if args.smoke else args.replicates
    run_tier = "diagnostic" if args.smoke else "publication"

    output_root = Path(args.output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir = output_root / f"lcn26_aeris_mechanism_{timestamp}"
    batch_dir.mkdir(parents=True, exist_ok=True)

    workers = get_safe_worker_limit(args.workers, args.max_cpu_percent)
    tasks = []
    for env in envs:
        for node_count in nodes:
            for rep in range(replicates):
                tasks.append((node_count, rep, env, args.seed, args.rounds, args.tx_power, args.mac_collision))

    print(f"[LCN26-MECH] envs={envs} nodes={nodes} replicates={replicates} tasks={len(tasks)}")
    print(f"[LCN26-MECH] workers_requested={args.workers} workers_effective={workers}")

    wait_for_resource_headroom(args.max_cpu_percent, args.max_mem_percent, args.resource_check_sec)

    t0 = time.time()
    runs: List[Dict] = []
    failed = 0
    completed = 0
    batch_size = max(1, int(args.batch_size))
    for batch_start in range(0, len(tasks), batch_size):
        batch = tasks[batch_start: batch_start + batch_size]
        batch_id = batch_start // batch_size + 1
        batch_total = math.ceil(len(tasks) / batch_size)
        batch_failed = 0
        print(f"[LCN26-MECH] batch {batch_id}/{batch_total} start size={len(batch)}")
        if workers == 1:
            iterator = (run_single(task) for task in batch)
            for result in iterator:
                runs.append(result)
                completed += 1
                if not result.get("success", True):
                    failed += 1
                    batch_failed += 1
                if completed % 100 == 0 or completed == len(tasks):
                    elapsed = time.time() - t0
                    rate = completed / elapsed if elapsed > 0 else 0.0
                    eta_min = ((len(tasks) - completed) / rate / 60.0) if rate > 0 else 0.0
                    print(f"[LCN26-MECH] {completed}/{len(tasks)} done, failed={failed}, rate={rate:.2f}/s, ETA={eta_min:.1f} min")
        else:
            with Pool(processes=workers, maxtasksperchild=12) as pool:
                for result in pool.imap_unordered(run_single, batch, chunksize=4):
                    runs.append(result)
                    completed += 1
                    if not result.get("success", True):
                        failed += 1
                        batch_failed += 1
                    if completed % 100 == 0 or completed == len(tasks):
                        elapsed = time.time() - t0
                        rate = completed / elapsed if elapsed > 0 else 0.0
                        eta_min = ((len(tasks) - completed) / rate / 60.0) if rate > 0 else 0.0
                        print(f"[LCN26-MECH] {completed}/{len(tasks)} done, failed={failed}, rate={rate:.2f}/s, ETA={eta_min:.1f} min")
        print(f"[LCN26-MECH] batch {batch_id}/{batch_total} done failed={batch_failed}")
        gc.collect()

    summary = aggregate(runs)
    output = {
        "timestamp": timestamp,
        "git_commit": get_git_commit(),
        "git_dirty": get_git_dirty(),
        "git_diff_stat": get_git_diff_stat(),
        "experiment_type": "lcn26_aeris_mechanism_matrix",
        "run_tier": run_tier,
        "primary_metric": "pdr_expected",
        "output_version": OUTPUT_VERSION,
        "environment": "multiple",
        "tx_power_dbm": args.tx_power,
        "workers_requested": args.workers,
        "workers_effective": workers,
        "error_runs": failed,
        "incomplete_runs": max(0, len(tasks) - len(runs)),
        "config": {
            "seed_base": args.seed,
            "replicates": replicates,
            "environments": list(envs),
            "node_counts": list(nodes),
            "round_counts": [args.rounds],
            "area_size": DEFAULT_AREA,
            "base_station": list(DEFAULT_BASE_STATION),
            "mac_collision": bool(args.mac_collision),
        },
        "raw_results": runs,
        "summary": summary,
    }

    out_json = batch_dir / "mechanism_raw.json"
    out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"[LCN26-MECH] wrote {out_json}")


if __name__ == "__main__":
    freeze_support()
    main()

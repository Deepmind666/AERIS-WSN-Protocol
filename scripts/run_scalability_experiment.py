#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scalability experiment across network sizes with multiple protocols.

Outputs:
  results/scalability_experiment.json
"""

import argparse
import json
import os
import random
import sys
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from benchmark_protocols import (
    NetworkConfig,
    LEACHProtocol,
    PEGASISProtocol,
    HEEDProtocolWrapper,
    TEENProtocolWrapper,
)
from improved_energy_model import ImprovedEnergyModel, HardwarePlatform
from aeris_protocol import AerisProtocol


PROTOCOLS = ("LEACH", "PEGASIS", "HEED", "TEEN", "AERIS_energy", "AERIS_robust")
NODE_COUNTS = (30, 50, 70, 100)


def generate_positions(seed: int, num_nodes: int, width: float, height: float) -> List[Tuple[float, float]]:
    rng = random.Random(seed)
    return [
        (rng.uniform(5.0, width - 5.0), rng.uniform(5.0, height - 5.0))
        for _ in range(num_nodes)
    ]


def build_config(num_nodes: int, seed: int) -> NetworkConfig:
    # Keep density roughly constant by scaling area with sqrt(n/50)
    scale = (num_nodes / 50.0) ** 0.5
    width = 100.0 * scale
    height = 100.0 * scale
    base_station = (width * 0.5, height * 1.2)

    cfg = NetworkConfig(
        num_nodes=num_nodes,
        area_width=width,
        area_height=height,
        base_station_x=base_station[0],
        base_station_y=base_station[1],
        initial_energy=2.0,
        packet_size=1024,
        temperature_c=25.0,
        humidity_ratio=0.5,
        enable_channel=True,
        channel_env="indoor_office",
        tx_power_dbm=0.0,
        link_retx=1,
        link_retx_power_step=1.0,
    )
    cfg.positions = generate_positions(seed, num_nodes, width, height)
    cfg.gateway_k = max(2, int(num_nodes / 25))
    cfg.gateway_retry_limit = 1
    cfg.gateway_rescue_direct = True
    cfg.intra_link_retx = 2
    cfg.intra_link_power_step = 1.5
    return cfg


def run_protocol(protocol: str, cfg: NetworkConfig, seed: int) -> Dict:
    random.seed(seed)
    np.random.seed(seed)
    cfg_local = deepcopy(cfg)
    em = ImprovedEnergyModel(HardwarePlatform.CC2420_TELOSB)

    if protocol == "LEACH":
        res = LEACHProtocol(cfg_local, em).run_simulation(200)
    elif protocol == "PEGASIS":
        res = PEGASISProtocol(cfg_local, em).run_simulation(200)
    elif protocol == "HEED":
        res = HEEDProtocolWrapper(cfg_local, em).run_simulation(200)
    elif protocol == "TEEN":
        res = TEENProtocolWrapper(cfg_local, em).run_simulation(200)
    elif protocol == "AERIS_energy":
        res = AerisProtocol(
            cfg_local,
            enable_cas=True,
            enable_fairness=True,
            enable_gateway=True,
            enable_skeleton=True,
            profile="energy",
            verbose=False,
            seed=seed,
        ).run_simulation(200)
    elif protocol == "AERIS_robust":
        res = AerisProtocol(
            cfg_local,
            enable_cas=True,
            enable_fairness=True,
            enable_gateway=True,
            enable_skeleton=True,
            profile="robust",
            verbose=False,
            seed=seed,
        ).run_simulation(200)
    else:
        raise ValueError(f"Unknown protocol {protocol}")

    return {
        "pdr_end2end": float(res.get("packet_delivery_ratio_end2end", res.get("packet_delivery_ratio", 0.0))),
        "energy": float(res.get("total_energy_consumed", 0.0)),
        "lifetime": int(res.get("network_lifetime", 0)),
        "alive_nodes": int(res.get("final_alive_nodes", res.get("alive_nodes", 0))),
    }


def run_task(args: Tuple[int, int, str, int]) -> Dict:
    num_nodes, replicate, protocol, base_seed = args
    seed = base_seed + replicate * 997 + hash(protocol) % 997
    cfg = build_config(num_nodes, seed)
    metrics = run_protocol(protocol, cfg, seed + 17)
    return {
        "num_nodes": num_nodes,
        "replicate": replicate,
        "protocol": protocol,
        "seed": seed,
        "metrics": metrics,
    }


def aggregate(runs: List[Dict]) -> Dict:
    summary: Dict = {}
    for num_nodes in NODE_COUNTS:
        summary[num_nodes] = {}
        for protocol in PROTOCOLS:
            filtered = [r for r in runs if r["num_nodes"] == num_nodes and r["protocol"] == protocol]
            pdrs = [r["metrics"]["pdr_end2end"] for r in filtered]
            energies = [r["metrics"]["energy"] for r in filtered]
            if not pdrs:
                continue
            summary[num_nodes][protocol] = {
                "pdr_mean": float(np.mean(pdrs)),
                "pdr_std": float(np.std(pdrs)),
                "energy_mean": float(np.mean(energies)),
                "energy_std": float(np.std(energies)),
                "n": len(pdrs),
            }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Run scalability experiment.")
    parser.add_argument("--replicates", type=int, default=30, help="Replicates per node count")
    parser.add_argument("--workers", type=int, default=6, help="Parallel workers")
    parser.add_argument("--seed", type=int, default=13579, help="Base seed")
    parser.add_argument("--output", default=None, help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    tasks: List[Tuple[int, int, str, int]] = []
    for num_nodes in NODE_COUNTS:
        for rep in range(args.replicates):
            for protocol in PROTOCOLS:
                tasks.append((num_nodes, rep, protocol, args.seed))

    runs: List[Dict] = []
    total = len(tasks)
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_task, t): t for t in tasks}
        for future in as_completed(futures):
            result = future.result()
            runs.append(result)
            completed += 1
            if completed % 10 == 0:
                print(f"[Scalability] {completed}/{total} completed")

    out = {
        "config": {
            "replicates": args.replicates,
            "rounds": 200,
            "node_counts": list(NODE_COUNTS),
            "protocols": list(PROTOCOLS),
            "channel_env": "indoor_office",
        },
        "runs": runs,
        "summary": aggregate(runs),
    }

    out_path = args.output or os.path.join(os.path.dirname(__file__), "..", "results", "scalability_experiment.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()

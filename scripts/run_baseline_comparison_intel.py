#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline comparison on Intel Lab geometry with sampled environment conditions.

Outputs:
  results/baseline_comparison.json
"""

import argparse
import json
import os
import random
import sys
from copy import deepcopy
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
from intel_dataset_loader import IntelLabDataLoader


PROTOCOLS = ("LEACH", "PEGASIS", "HEED", "TEEN", "AERIS_energy", "AERIS_robust")


def load_intel_geometry(loader: IntelLabDataLoader) -> Tuple[List[Tuple[float, float]], float, float, Tuple[float, float]]:
    locs = loader.locations_data.sort_values("node_id")
    xs = locs["x"].to_list()
    ys = locs["y"].to_list()
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    width = maxx - minx if maxx > minx else 50.0
    height = maxy - miny if maxy > miny else 50.0
    positions = [(float(x - minx), float(y - miny)) for x, y in zip(xs, ys)]
    base_station = (width * 1.05, height * 0.5)
    return positions, width, height, base_station


def sample_environment(sensor_data, rng: random.Random) -> Tuple[float, float]:
    row = sensor_data.iloc[rng.randrange(len(sensor_data))]
    temperature_c = float(row["temperature"])
    humidity_ratio = max(0.0, min(1.0, float(row["humidity"]) / 100.0))
    return temperature_c, humidity_ratio


def build_config(
    positions: List[Tuple[float, float]],
    width: float,
    height: float,
    base_station: Tuple[float, float],
    temperature_c: float,
    humidity_ratio: float,
) -> NetworkConfig:
    cfg = NetworkConfig(
        num_nodes=len(positions),
        area_width=width,
        area_height=height,
        base_station_x=base_station[0],
        base_station_y=base_station[1],
        initial_energy=2.0,
        packet_size=1024,
        temperature_c=temperature_c,
        humidity_ratio=humidity_ratio,
        enable_channel=True,
        channel_env="indoor_office",
        tx_power_dbm=0.0,
        link_retx=1,
        link_retx_power_step=1.0,
    )
    cfg.positions = positions
    cfg.gateway_k = max(2, int(len(positions) / 25))
    cfg.gateway_retry_limit = 1
    cfg.gateway_rescue_direct = True
    cfg.intra_link_retx = 2
    cfg.intra_link_power_step = 1.5
    return cfg


def run_protocols(cfg: NetworkConfig, seed: int) -> Dict[str, Dict]:
    em = ImprovedEnergyModel(HardwarePlatform.CC2420_TELOSB)
    results: Dict[str, Dict] = {}

    proto_seeds = {
        "LEACH": seed + 11,
        "PEGASIS": seed + 23,
        "HEED": seed + 37,
        "TEEN": seed + 51,
        "AERIS_energy": seed + 71,
        "AERIS_robust": seed + 79,
    }

    for name, offset_seed in proto_seeds.items():
        random.seed(offset_seed)
        np.random.seed(offset_seed)
        cfg_local = deepcopy(cfg)

        if name == "LEACH":
            res = LEACHProtocol(cfg_local, em).run_simulation(200)
        elif name == "PEGASIS":
            res = PEGASISProtocol(cfg_local, em).run_simulation(200)
        elif name == "HEED":
            res = HEEDProtocolWrapper(cfg_local, em).run_simulation(200)
        elif name == "TEEN":
            res = TEENProtocolWrapper(cfg_local, em).run_simulation(200)
        elif name == "AERIS_energy":
            res = AerisProtocol(
                cfg_local,
                enable_cas=True,
                enable_fairness=True,
                enable_gateway=True,
                enable_skeleton=True,
                profile="energy",
                verbose=False,
                seed=offset_seed,
            ).run_simulation(200)
        elif name == "AERIS_robust":
            res = AerisProtocol(
                cfg_local,
                enable_cas=True,
                enable_fairness=True,
                enable_gateway=True,
                enable_skeleton=True,
                profile="robust",
                verbose=False,
                seed=offset_seed,
            ).run_simulation(200)
        else:
            raise ValueError(f"Unknown protocol {name}")

        results[name] = {
            "pdr_end2end": float(res.get("packet_delivery_ratio_end2end", res.get("packet_delivery_ratio", 0.0))),
            "energy": float(res.get("total_energy_consumed", 0.0)),
            "lifetime": int(res.get("network_lifetime", 0)),
            "alive_nodes": int(res.get("final_alive_nodes", res.get("alive_nodes", 0))),
        }
    return results


def summarize(runs: List[Dict]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for name in PROTOCOLS:
        pdrs = [r["protocols"][name]["pdr_end2end"] for r in runs]
        energies = [r["protocols"][name]["energy"] for r in runs]
        summary[name] = {
            "pdr_mean": float(np.mean(pdrs)),
            "pdr_std": float(np.std(pdrs)),
            "energy_mean": float(np.mean(energies)),
            "energy_std": float(np.std(energies)),
            "n": len(pdrs),
        }
    return summary


def parse_args():
    parser = argparse.ArgumentParser(description="Baseline comparison on Intel Lab geometry.")
    parser.add_argument("--replicates", type=int, default=50, help="Number of replicates")
    parser.add_argument("--seed", type=int, default=24680, help="Base seed for sampling")
    parser.add_argument("--output", default=None, help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    loader = IntelLabDataLoader(data_dir=data_dir, use_synthetic=False)
    if loader.sensor_data is None or loader.sensor_data.empty:
        raise RuntimeError("Intel dataset missing; please provide data.txt.gz under data/Intel_Lab_Data")
    if loader.locations_data is None or loader.locations_data.empty:
        raise RuntimeError("Intel mote_locs.txt missing; please provide under data/Intel_Lab_Data")

    sensor_data = loader.sensor_data.dropna(subset=["humidity", "temperature"]).reset_index(drop=True)
    if sensor_data.empty:
        raise RuntimeError("Intel sensor data missing humidity/temperature columns.")

    positions, width, height, base_station = load_intel_geometry(loader)
    rng = random.Random(args.seed)

    runs: List[Dict] = []
    for rep in range(args.replicates):
        temperature_c, humidity_ratio = sample_environment(sensor_data, rng)
        cfg = build_config(positions, width, height, base_station, temperature_c, humidity_ratio)
        rep_seed = args.seed + rep * 997
        results = run_protocols(cfg, rep_seed)
        runs.append(
            {
                "replicate": rep,
                "seed": rep_seed,
                "temperature_c": temperature_c,
                "humidity_ratio": humidity_ratio,
                "protocols": results,
            }
        )
        print(
            f"[Baseline] rep={rep} T={temperature_c:.2f}C RH={humidity_ratio:.2f} "
            f"AERIS_R PDR={results['AERIS_robust']['pdr_end2end']:.3f}"
        )

    out = {
        "config": {
            "dataset": "Intel Lab",
            "replicates": args.replicates,
            "rounds": 200,
            "positions": len(positions),
            "channel_env": "indoor_office",
        },
        "runs": runs,
        "summary": summarize(runs),
    }

    out_path = args.output or os.path.join(os.path.dirname(__file__), "..", "results", "baseline_comparison.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[DONE] Wrote {out_path}")


if __name__ == "__main__":
    main()

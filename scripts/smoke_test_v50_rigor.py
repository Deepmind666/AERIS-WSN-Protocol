#!/usr/bin/env python3
"""Smoke test: verify MAC collision + multihop relay integration across all 5 protocols."""
import sys, os, random, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from benchmark_protocols import NetworkConfig
from baseline_protocols import LEACHProtocol, PEGASISProtocol, HEEDProtocol
from baseline_protocols.leach_protocol import LEACHNode
from baseline_protocols.pegasis_protocol import PEGASISNode
from baseline_protocols.heed_protocol import HEEDNode
from teen_protocol import TEENConfig, TEENProtocol
from aeris_protocol import AerisProtocol
from mac_collision_model import MACCollisionModel, MACCollisionConfig
from realistic_channel_model import RealisticChannelModel, EnvironmentType

SEED = 42
N = 50
ROUNDS = 30
AREA = 150.0
TX_POWER = 10.0

def make_positions(n, seed):
    rng = random.Random(seed)
    return [(rng.uniform(10, AREA-10), rng.uniform(10, AREA-10)) for _ in range(n)]

def make_cfg(positions):
    return NetworkConfig(
        area_width=AREA, area_height=AREA,
        base_station_x=AREA/2, base_station_y=AREA+50,
        num_nodes=N, initial_energy=2.0, packet_size=1024,
        enable_channel=True, channel_env='indoor_office',
        tx_power_dbm=TX_POWER, link_retx=1, link_retx_power_step=1.0,
        positions=positions,
    )

def run_leach(cfg, positions, channel, mac_model):
    nodes = [LEACHNode(i, positions[i][0], positions[i][1], cfg.initial_energy) for i in range(N)]
    proto = LEACHProtocol(nodes, (cfg.base_station_x, cfg.base_station_y),
                          tx_power_dbm=TX_POWER, channel_model=channel,
                          mac_collision_model=mac_model, enable_multihop_relay=True,
                          link_retx=1, link_retx_power_step=1.0)
    proto.run_simulation(ROUNDS)
    pdr = float(proto.total_bs_delivered) / max(1, proto.source_packets_expected)
    return pdr

def run_pegasis(cfg, positions, channel, mac_model):
    nodes = [PEGASISNode(i, positions[i][0], positions[i][1], cfg.initial_energy) for i in range(N)]
    proto = PEGASISProtocol(nodes, (cfg.base_station_x, cfg.base_station_y),
                            tx_power_dbm=TX_POWER, channel_model=channel,
                            mac_collision_model=mac_model,
                            link_retx=1, link_retx_power_step=1.0)
    proto.run_simulation(ROUNDS)
    pdr = float(proto.total_bs_delivered) / max(1, proto.source_packets_expected)
    return pdr

def run_heed(cfg, positions, channel, mac_model):
    nodes = [HEEDNode(i, positions[i][0], positions[i][1], cfg.initial_energy) for i in range(N)]
    proto = HEEDProtocol(nodes, (cfg.base_station_x, cfg.base_station_y),
                         tx_power_dbm=TX_POWER, channel_model=channel,
                         mac_collision_model=mac_model, enable_multihop_relay=True,
                         link_retx=1, link_retx_power_step=1.0)
    proto.run_simulation(ROUNDS)
    pdr = float(proto.total_bs_delivered) / max(1, proto.source_packets_expected)
    return pdr

def run_teen(cfg, positions, channel, mac_model):
    teen_cfg = TEENConfig(
        num_nodes=N, area_width=AREA, area_height=AREA,
        base_station_x=AREA/2, base_station_y=AREA+50,
        initial_energy=2.0, tx_power_dbm=TX_POWER,
        enable_channel=True, channel_env='indoor_office',
        link_retx=1, link_retx_power_step=1.0,
    )
    proto = TEENProtocol(teen_cfg, use_unified_energy_model=True,
                         mac_collision_model=mac_model, enable_multihop_relay=True)
    proto.initialize_network(positions)
    proto.run_simulation(max_rounds=ROUNDS)
    pdr = float(proto.bs_delivered_total) / max(1, proto.source_packets_expected)
    return pdr

def run_aeris(cfg, channel, mac_model):
    random.seed(SEED)
    proto = AerisProtocol(cfg, profile='robust', verbose=False, seed=SEED,
                          mac_collision_model=mac_model)
    result = proto.run_simulation(ROUNDS)
    return result.get('pdr_expected', result.get('packet_delivery_ratio', 0))

def main():
    positions = make_positions(N, SEED)
    cfg = make_cfg(positions)
    channel = RealisticChannelModel(EnvironmentType.INDOOR_OFFICE)
    mac_model = MACCollisionModel(MACCollisionConfig(enabled=True))

    print(f"=== Smoke Test: {N} nodes, {ROUNDS} rounds, indoor_office, MAC collision ON ===")
    results = {}
    for name, fn in [
        ("LEACH",   lambda: run_leach(cfg, positions, channel, mac_model)),
        ("PEGASIS", lambda: run_pegasis(cfg, positions, channel, mac_model)),
        ("HEED",    lambda: run_heed(cfg, positions, channel, mac_model)),
        ("TEEN",    lambda: run_teen(cfg, positions, channel, mac_model)),
        ("AERIS",   lambda: run_aeris(cfg, channel, mac_model)),
    ]:
        random.seed(SEED)
        try:
            pdr = fn()
            results[name] = pdr
            print(f"  {name:8s}: PDR = {pdr:.4f}")
        except Exception as e:
            results[name] = -1
            print(f"  {name:8s}: ERROR - {e}")

    # Sanity checks
    print("\n=== Sanity Checks ===")
    ok = True
    for name, pdr in results.items():
        if pdr < 0:
            print(f"  FAIL: {name} crashed")
            ok = False
        elif pdr == 0:
            print(f"  WARN: {name} PDR=0 (may be expected for short run)")
    if ok:
        print("  All protocols ran successfully with MAC collision model.")
    return 0 if ok else 1

if __name__ == "__main__":
    sys.exit(main())

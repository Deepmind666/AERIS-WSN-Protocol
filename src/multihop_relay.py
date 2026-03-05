#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-hop Relay Module for WSN Baseline Protocols.

Provides greedy geographic forwarding so that baseline protocols (LEACH, HEED, TEEN)
can relay CH→BS traffic through intermediate CHs, instead of single-hop direct
transmission. This levels the playing field with AERIS which already has multi-hop
relay via its gateway/skeleton modules.

Literature basis:
  - Greedy Geographic Forwarding is a standard WSN technique
  - LEACH-C (Heinzelman 2002) discusses CH inter-cooperation
  - HEED original paper mentions multi-hop as an extension

Author: AERIS Research Team
Date: 2026-02-15
"""

import math
from typing import Dict, List, Optional, Tuple


def _distance(ax: float, ay: float, bx: float, by: float) -> float:
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def build_ch_relay_tree(
    cluster_heads: List[dict],
    bs_x: float,
    bs_y: float,
    direct_threshold: float = 80.0,
) -> Dict[int, Optional[int]]:
    """Build a greedy geographic forwarding tree from CHs to BS.

    Each CH whose distance to BS exceeds *direct_threshold* selects the
    nearest CH that is closer to BS as its next-hop relay.  CHs within
    *direct_threshold* transmit directly.

    Args:
        cluster_heads: list of dicts with keys {id, x, y, is_alive}.
        bs_x, bs_y: base station coordinates.
        direct_threshold: distance below which a CH transmits directly to BS.

    Returns:
        {ch_id: next_hop_ch_id} where None means direct-to-BS.
    """
    relay_tree: Dict[int, Optional[int]] = {}
    alive_chs = [ch for ch in cluster_heads if ch.get("is_alive", True)]

    for ch in alive_chs:
        dist_to_bs = _distance(ch["x"], ch["y"], bs_x, bs_y)
        if dist_to_bs <= direct_threshold:
            relay_tree[ch["id"]] = None  # direct
            continue

        # Find nearest CH that is closer to BS
        best_next = None
        best_dist = float("inf")
        for candidate in alive_chs:
            if candidate["id"] == ch["id"]:
                continue
            cand_dist_to_bs = _distance(candidate["x"], candidate["y"], bs_x, bs_y)
            if cand_dist_to_bs >= dist_to_bs:
                continue  # must be strictly closer to BS
            hop_dist = _distance(ch["x"], ch["y"], candidate["x"], candidate["y"])
            if hop_dist < best_dist:
                best_dist = hop_dist
                best_next = candidate["id"]

        relay_tree[ch["id"]] = best_next  # None if no closer CH found (fallback direct)

    return relay_tree


def transmit_via_relay(
    source_ch: dict,
    relay_tree: Dict[int, Optional[int]],
    ch_lookup: Dict[int, dict],
    bs_x: float,
    bs_y: float,
    channel_model,
    tx_power_dbm: float,
    energy_model,
    packet_size_bits: int,
    collision_factor: float = 1.0,
    link_retx: int = 1,
    link_retx_power_step: float = 1.0,
    temperature_c: float = 25.0,
    humidity_ratio: float = 0.5,
) -> Tuple[bool, float, int]:
    """Transmit data from a CH to BS along the relay tree.

    Returns:
        (success, total_energy_consumed, hop_count)
    """
    import random as _random

    total_energy = 0.0
    hop_count = 0
    current_id = source_ch["id"]
    max_hops = 10  # safety limit to prevent loops

    while hop_count < max_hops:
        next_hop_id = relay_tree.get(current_id)
        current_ch = ch_lookup.get(current_id)
        if current_ch is None:
            return False, total_energy, hop_count

        if next_hop_id is None:
            # Direct to BS
            dist = _distance(current_ch["x"], current_ch["y"], bs_x, bs_y)
            target_x, target_y = bs_x, bs_y
        else:
            next_ch = ch_lookup.get(next_hop_id)
            if next_ch is None or not next_ch.get("is_alive", True):
                # Relay dead — fallback to direct
                dist = _distance(current_ch["x"], current_ch["y"], bs_x, bs_y)
                target_x, target_y = bs_x, bs_y
                next_hop_id = None
            else:
                dist = _distance(current_ch["x"], current_ch["y"], next_ch["x"], next_ch["y"])
                target_x, target_y = next_ch["x"], next_ch["y"]

        # TX energy
        tx_energy = energy_model.calculate_transmission_energy(
            data_size_bits=packet_size_bits,
            distance=dist,
            tx_power_dbm=tx_power_dbm,
            temperature_c=temperature_c,
            humidity_ratio=humidity_ratio,
        )
        total_energy += tx_energy
        hop_count += 1

        # Link success with retransmission and collision factor
        success = False
        if channel_model is not None:
            for attempt in range(link_retx + 1):
                power = tx_power_dbm + attempt * link_retx_power_step
                metrics = channel_model.calculate_link_metrics(
                    power, dist, temperature_c, humidity_ratio
                )
                effective_pdr = metrics.get("pdr", 0.0) * collision_factor
                if _random.random() < effective_pdr:
                    success = True
                    break
                # Extra energy for retransmission
                if attempt < link_retx:
                    total_energy += tx_energy * 0.5  # partial retx cost
        else:
            success = True

        if not success:
            return False, total_energy, hop_count

        # RX energy at relay node
        if next_hop_id is not None:
            rx_energy = energy_model.calculate_reception_energy(
                data_size_bits=packet_size_bits,
                temperature_c=temperature_c,
                humidity_ratio=humidity_ratio,
            )
            total_energy += rx_energy

        if next_hop_id is None:
            # Reached BS
            return True, total_energy, hop_count

        current_id = next_hop_id

    # Exceeded max hops
    return False, total_energy, hop_count

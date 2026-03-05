#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAC Collision Model for WSN Simulation.

Two-tier hybrid collision model:
  Tier 1 — Intra-cluster (member→CH): TDMA slot model
  Tier 2 — Uplink (CH→BS): Offered-load (slotted ALOHA) model

Literature basis:
  - IEEE Std 802.15.4-2006 MAC parameters
  - Pollin et al., IEEE TWC 2008 (slotted CSMA/CA analysis)
  - Heinzelman et al., HICSS 2000 (LEACH assumes intra-cluster TDMA)

Author: AERIS Research Team
Date: 2026-02-15
"""

import math
from dataclasses import dataclass


@dataclass
class MACCollisionConfig:
    """Configurable MAC collision parameters for sensitivity analysis."""
    enabled: bool = True
    slots_per_frame: int = 20        # TDMA slots per superframe (IEEE 802.15.4: 16 GTS + CAP)
    uplink_channel_slots: int = 16   # Available uplink slots (IEEE 802.15.4 CAP: 16 slots with CSMA/CA)
    # PEGASIS chain forwarding is sequential — no intra-cluster contention
    pegasis_chain_exempt: bool = True


def intra_cluster_collision_factor(cluster_size: int, slots_per_frame: int = 20) -> float:
    """Tier 1: TDMA slot model for intra-cluster (member→CH) contention.

    If cluster_size <= slots, each member gets a dedicated slot (no collision).
    If cluster_size > slots, excess members contend via slotted ALOHA.

    Returns a multiplicative factor in [0, 1] to apply to per-link PDR.
    """
    if cluster_size <= 0:
        return 1.0
    if cluster_size <= slots_per_frame:
        return 1.0
    excess = cluster_size - slots_per_frame
    # Slotted ALOHA: P_success = exp(-G), G = excess / slots
    return math.exp(-excess / slots_per_frame)


def uplink_collision_factor(num_concurrent_chs: int, uplink_slots: int = 8) -> float:
    """Tier 2: Offered-load model for uplink (CH→BS) contention.

    Multiple CHs transmit to BS in the same round, sharing the uplink channel.
    P_success = exp(-G), G = num_chs / uplink_slots.

    Returns a multiplicative factor in [0, 1] to apply to CH→BS PDR.
    """
    if num_concurrent_chs <= 1:
        return 1.0
    offered_load = num_concurrent_chs / max(1, uplink_slots)
    return math.exp(-offered_load)


class MACCollisionModel:
    """Stateless collision model that computes per-round collision factors."""

    def __init__(self, config: MACCollisionConfig = None):
        self.config = config or MACCollisionConfig()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def compute_intra_factor(self, cluster_size: int) -> float:
        """Collision factor for member→CH transmission within a cluster."""
        if not self.config.enabled:
            return 1.0
        return intra_cluster_collision_factor(cluster_size, self.config.slots_per_frame)

    def compute_uplink_factor(self, num_chs: int) -> float:
        """Collision factor for CH→BS uplink transmission."""
        if not self.config.enabled:
            return 1.0
        return uplink_collision_factor(num_chs, self.config.uplink_channel_slots)

    def compute_chain_factor(self, chain_length: int) -> float:
        """Collision factor for PEGASIS chain forwarding.

        PEGASIS uses sequential token-passing along the chain, so there is
        no intra-cluster contention. Only the leader→BS hop faces uplink
        contention (but PEGASIS has exactly 1 leader, so uplink factor ≈ 1.0).
        """
        if not self.config.enabled or self.config.pegasis_chain_exempt:
            return 1.0
        # If not exempt, treat chain as a long cluster
        return intra_cluster_collision_factor(chain_length, self.config.slots_per_frame)

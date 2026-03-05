#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LEACH (Low-Energy Adaptive Clustering Hierarchy) protocol implementation.
This module provides a basic LEACH simulation for WSN baseline comparison.

**MODIFIED 2025-11-04**: Now uses ImprovedEnergyModel for unified comparison with AERIS.

References:
Heinzelman, W. R., Chandrakasan, A., & Balakrishnan, H. (2000).
Energy-efficient communication protocol for wireless microsensor networks.
In Proceedings of the 33rd annual Hawaii international conference on system sciences.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import random
import math
from typing import List, Tuple, Dict, Optional
import matplotlib.pyplot as plt

# Import unified energy model
from improved_energy_model import ImprovedEnergyModel, HardwarePlatform

class LEACHNode:
    """LEACH node representing a sensor node."""
    
    def __init__(self, node_id: int, x: float, y: float, initial_energy: float = 2.0):
        self.node_id = node_id
        self.x = x
        self.y = y
        self.initial_energy = initial_energy
        self.current_energy = initial_energy
        self.is_alive = True
        self.is_cluster_head = False
        self.cluster_head_id = None
        self.cluster_members = []
        
        # LEACH specific parameters
        self.ch_probability = 0.0
        self.last_ch_round = -1
        self.data_packets = []
        
    def distance_to(self, other_node) -> float:
        """Compute Euclidean distance to another node."""
        return math.sqrt((self.x - other_node.x)**2 + (self.y - other_node.y)**2)
    
    def consume_energy(self, energy_amount: float):
        """Consume energy and update alive status."""
        self.current_energy -= energy_amount
        if self.current_energy <= 0:
            self.current_energy = 0
            self.is_alive = False
    
    def reset_cluster_info(self):
        """Reset cluster membership and head assignment."""
        self.is_cluster_head = False
        self.cluster_head_id = None
        self.cluster_members = []

class LEACHProtocol:
    """LEACH protocol simulator."""
    
    def __init__(self, nodes: List[LEACHNode], base_station: Tuple[float, float],
                 desired_ch_percentage: float = 0.1, use_unified_energy_model: bool = True,
                 tx_power_dbm: float = 10.0, channel_model=None,
                 mac_collision_model=None, enable_multihop_relay: bool = False,
                 link_retx: int = 1, link_retx_power_step: float = 1.0):
        """Initialize LEACH protocol.

        Args:
            nodes: List of LEACH nodes
            base_station: Base station coordinates (x, y)
            desired_ch_percentage: Desired percentage of cluster heads
            use_unified_energy_model: If True, use ImprovedEnergyModel (CC2420 parameters).
                                     If False, use legacy simplified parameters.
            tx_power_dbm: Transmission power in dBm (default 10.0 for fair comparison)
            channel_model: Optional RealisticChannelModel for PDR calculation
            mac_collision_model: Optional MACCollisionModel for contention modeling
            enable_multihop_relay: If True, CHs relay via closer CHs instead of direct-to-BS
            link_retx: Number of link-layer retransmission attempts
            link_retx_power_step: Power increase (dBm) per retry
        """
        self.nodes = nodes
        self.base_station = base_station
        self.desired_ch_percentage = desired_ch_percentage
        self.current_round = 0
        self.use_unified_energy_model = use_unified_energy_model
        self.tx_power_dbm = tx_power_dbm
        self.channel_model = channel_model
        self.mac_collision_model = mac_collision_model
        self.enable_multihop_relay = enable_multihop_relay
        self.link_retx = link_retx
        self.link_retx_power_step = link_retx_power_step

        # Packet size and radio range
        self.packet_size = 8192  # packet size (bits) - 统一为1024 bytes
        self.radio_range = 0.5 * 100 * math.sqrt(2)  # radio range

        if use_unified_energy_model:
            # Use unified real hardware model (CC2420 TelosB)
            self.energy_model = ImprovedEnergyModel(HardwarePlatform.CC2420_TELOSB)
            self.d_crossover = 87.0  # Use standard threshold
            print(f"[LEACH] Using unified energy model (CC2420 TelosB, 208.8 nJ/bit)")
        else:
            # Legacy simplified parameters (for backward compatibility)
            self.E_elec = 50e-9  # electronics energy (J/bit)
            self.E_fs = 10e-12   # free-space model (J/bit/m^2)
            self.E_mp = 0.0013e-12  # multi-path model (J/bit/m^4)
            self.E_DA = 5e-9     # data aggregation energy (J/bit)
            self.d_crossover = math.sqrt(self.E_fs / self.E_mp)  # threshold distance
            self.energy_model = None
            print(f"[LEACH] Using legacy energy model (50 nJ/bit)")

        # Initial energy parameter (classical LEACH often uses 2J)
        self.initial_energy = 2.0

        # Metrics
        self.total_energy_consumed = 0.0
        self.packets_sent = 0
        self.packets_received = 0
        self.total_source_packets = 0  # 实际尝试发送的包数 (attempted)
        self.source_packets_expected = 0  # 期望包数 = 每轮存活节点数累计
        self.total_bs_delivered = 0
        self.dead_nodes = 0
        self.network_lifetime = 0
        self.energy_consumption_per_round = []
        self.alive_nodes_per_round = []
        self._all_hop_counts = []

        print(f"[OK] LEACH initialization complete")
        print(f"   Nodes: {len(self.nodes)}")
        print(f"   Base station: {self.base_station}")
        print(f"   Desired CH percentage: {self.desired_ch_percentage*100:.1f}%")
    
    def calculate_transmission_energy(self, distance: float, packet_size: int,
                                     temperature_c: float = 25.0, humidity_ratio: float = 0.5) -> float:
        """Compute transmission energy per packet based on distance.

        Uses unified energy model if enabled, otherwise legacy simplified model.
        """
        if self.use_unified_energy_model:
            # Use ImprovedEnergyModel (real CC2420 parameters)
            return self.energy_model.calculate_transmission_energy(
                data_size_bits=packet_size,
                distance=distance,
                tx_power_dbm=self.tx_power_dbm,
                temperature_c=temperature_c,
                humidity_ratio=humidity_ratio
            )
        else:
            # Legacy simplified model
            if distance < self.d_crossover:
                return self.E_elec * packet_size + self.E_fs * packet_size * (distance ** 2)
            else:
                return self.E_elec * packet_size + self.E_mp * packet_size * (distance ** 4)

    def calculate_reception_energy(self, packet_size: int,
                                   temperature_c: float = 25.0, humidity_ratio: float = 0.5) -> float:
        """Compute reception energy per packet.

        Uses unified energy model if enabled, otherwise legacy simplified model.
        """
        if self.use_unified_energy_model:
            # Use ImprovedEnergyModel (real CC2420 parameters)
            return self.energy_model.calculate_reception_energy(
                data_size_bits=packet_size,
                temperature_c=temperature_c,
                humidity_ratio=humidity_ratio
            )
        else:
            # Legacy simplified model
            return self.E_elec * packet_size
    
    def cluster_head_selection(self) -> List[LEACHNode]:
        """Select cluster heads following LEACH threshold rule."""
        cluster_heads = []
        
        # Reset epoch if needed
        if self.current_round % (1 / self.desired_ch_percentage) == 0:
            for node in self.nodes:
                node.last_ch_round = -1
        
        for node in self.nodes:
            if not node.is_alive:
                continue
            rounds_since_ch = self.current_round - node.last_ch_round
            if rounds_since_ch < (1 / self.desired_ch_percentage):
                continue
            remaining_nodes = len([n for n in self.nodes if n.is_alive and 
                                 (self.current_round - n.last_ch_round) >= (1 / self.desired_ch_percentage)])
            if remaining_nodes == 0:
                continue
            threshold = self.desired_ch_percentage / (1 - self.desired_ch_percentage * 
                                                    (self.current_round % (1 / self.desired_ch_percentage)))
            random_value = random.random()
            if random_value < threshold:
                node.is_cluster_head = True
                node.last_ch_round = self.current_round
                cluster_heads.append(node)
        return cluster_heads
    
    def cluster_formation(self, cluster_heads: List[LEACHNode]):
        """Form clusters by assigning nodes to the closest CH under constraints."""
        for node in self.nodes:
            node.reset_cluster_info()
            node.cluster_head_id = None
        for ch in cluster_heads:
            ch.is_cluster_head = True
        for node in self.nodes:
            if not node.is_alive or node.is_cluster_head:
                continue
            min_distance = float('inf')
            closest_ch = None
            for ch in cluster_heads:
                distance = node.distance_to(ch)
                if distance < min_distance:
                    min_distance = distance
                    closest_ch = ch
            if closest_ch:
                # Fair LEACH assignment: attach to nearest CH without extra
                # "closer-than-BS" gating that suppresses baseline traffic.
                node.cluster_head_id = closest_ch.node_id
                closest_ch.cluster_members.append(node)
    
    def _link_check(self, distance: float, collision_factor: float = 1.0) -> bool:
        """Check link success with channel model, collision factor, and retransmission."""
        if self.channel_model is None:
            return True
        for attempt in range(self.link_retx + 1):
            power = self.tx_power_dbm + attempt * self.link_retx_power_step
            metrics = self.channel_model.calculate_link_metrics(power, distance, 25.0, 0.5)
            effective_pdr = metrics.get('pdr', 0.0) * collision_factor
            if random.random() < effective_pdr:
                return True
        return False

    def data_transmission_phase(self, cluster_heads: List[LEACHNode]):
        """Simulate one data transmission phase in LEACH.

        Integrates MAC collision model (intra-cluster + uplink contention)
        and optional multi-hop CH relay.
        """
        round_energy_consumption = 0.0
        if not hasattr(self, '_all_hop_counts'):
            self._all_hop_counts = []

        # Compute collision factors for this round
        num_alive_chs = sum(1 for ch in cluster_heads if ch.is_alive)
        mac = self.mac_collision_model
        uplink_factor = mac.compute_uplink_factor(num_alive_chs) if mac and mac.enabled else 1.0

        # Build multi-hop relay tree if enabled
        relay_tree = None
        ch_lookup = None
        if self.enable_multihop_relay and num_alive_chs > 1:
            try:
                from multihop_relay import build_ch_relay_tree
                ch_dicts = [
                    {"id": ch.node_id, "x": ch.x, "y": ch.y, "is_alive": ch.is_alive}
                    for ch in cluster_heads if ch.is_alive
                ]
                relay_tree = build_ch_relay_tree(
                    ch_dicts, self.base_station[0], self.base_station[1],
                    direct_threshold=80.0,
                )
                ch_lookup = {ch.node_id: ch for ch in cluster_heads if ch.is_alive}
            except ImportError:
                relay_tree = None

        # --- Phase 1: Intra-cluster (member → CH) ---
        if cluster_heads:
            for ch in cluster_heads:
                if not ch.is_alive or not ch.cluster_members:
                    continue
                cluster_size = len(ch.cluster_members)
                intra_factor = mac.compute_intra_factor(cluster_size) if mac and mac.enabled else 1.0

                for member in ch.cluster_members:
                    if not member.is_alive:
                        continue
                    distance = member.distance_to(ch)
                    tx_energy = self.calculate_transmission_energy(distance, self.packet_size, temperature_c=25.0, humidity_ratio=0.5)
                    rx_energy = self.calculate_reception_energy(self.packet_size, temperature_c=25.0, humidity_ratio=0.5)
                    if member.current_energy >= tx_energy and ch.current_energy >= rx_energy:
                        member.consume_energy(tx_energy)
                        ch.consume_energy(rx_energy)
                        round_energy_consumption += tx_energy + rx_energy
                        self.packets_sent += 1
                        self.total_source_packets += 1
                        member_to_ch_success = self._link_check(distance, intra_factor)
                        if member_to_ch_success:
                            if not hasattr(ch, '_packets_received_this_round'):
                                ch._packets_received_this_round = 0
                            ch._packets_received_this_round += 1

        # --- Phase 2: Unclustered nodes → BS (direct) ---
        for node in self.nodes:
            if not node.is_alive or node.is_cluster_head:
                continue
            if node.cluster_head_id is None:
                bs_distance = math.sqrt((node.x - self.base_station[0])**2 +
                                      (node.y - self.base_station[1])**2)
                tx_energy = self.calculate_transmission_energy(bs_distance, self.packet_size, temperature_c=25.0, humidity_ratio=0.5)
                if node.current_energy >= tx_energy:
                    node.consume_energy(tx_energy)
                    round_energy_consumption += tx_energy
                    self.packets_sent += 1
                    self.total_source_packets += 1
                    if self._link_check(bs_distance, uplink_factor):
                        self.total_bs_delivered += 1
                        self._all_hop_counts.append(1)

        # --- Phase 3: CH → BS (with aggregation, optional multi-hop relay) ---
        if cluster_heads:
            for ch in cluster_heads:
                if not ch.is_alive:
                    continue
                self.total_source_packets += 1
                if not hasattr(ch, '_packets_received_this_round'):
                    ch._packets_received_this_round = 0
                ch._packets_received_this_round += 1  # CH's own data
                packets_at_ch = ch._packets_received_this_round
                if packets_at_ch == 0:
                    continue

                # Data aggregation energy
                if self.use_unified_energy_model:
                    aggregation_energy = self.energy_model.calculate_processing_energy(
                        self.packet_size * packets_at_ch)
                else:
                    aggregation_energy = self.E_DA * self.packet_size * packets_at_ch

                ch.consume_energy(aggregation_energy)
                round_energy_consumption += aggregation_energy

                # CH → BS transmission (direct or multi-hop relay)
                bs_distance = math.sqrt((ch.x - self.base_station[0])**2 +
                                      (ch.y - self.base_station[1])**2)

                if relay_tree is not None and ch.node_id in relay_tree and relay_tree[ch.node_id] is not None:
                    # Multi-hop relay path
                    from multihop_relay import transmit_via_relay
                    ch_dict = {"id": ch.node_id, "x": ch.x, "y": ch.y, "is_alive": ch.is_alive}
                    ch_dict_lookup = {
                        cid: {"id": cid, "x": c.x, "y": c.y, "is_alive": c.is_alive}
                        for cid, c in ch_lookup.items()
                    }
                    success, relay_energy, hop_count = transmit_via_relay(
                        ch_dict, relay_tree, ch_dict_lookup,
                        self.base_station[0], self.base_station[1],
                        self.channel_model, self.tx_power_dbm, self.energy_model,
                        self.packet_size, collision_factor=uplink_factor,
                        link_retx=self.link_retx, link_retx_power_step=self.link_retx_power_step,
                    )
                    round_energy_consumption += relay_energy
                    self.packets_sent += 1
                    if success:
                        self.total_bs_delivered += packets_at_ch
                        for _ in range(packets_at_ch):
                            self._all_hop_counts.append(1 + hop_count)
                else:
                    # Direct CH → BS
                    tx_energy = self.calculate_transmission_energy(bs_distance, self.packet_size, temperature_c=25.0, humidity_ratio=0.5)
                    if ch.current_energy >= tx_energy:
                        ch.consume_energy(tx_energy)
                        round_energy_consumption += tx_energy
                        self.packets_sent += 1
                        if self._link_check(bs_distance, uplink_factor):
                            self.total_bs_delivered += packets_at_ch
                            for _ in range(packets_at_ch):
                                self._all_hop_counts.append(2)

                ch._packets_received_this_round = 0

        self.total_energy_consumed += round_energy_consumption
        self.energy_consumption_per_round.append(round_energy_consumption)
    
    def run_round(self) -> bool:
        """Run a single LEACH round."""
        alive_nodes = [n for n in self.nodes if n.is_alive]
        if not alive_nodes:
            return False
        # 累计期望包数 = 每轮存活节点数
        self.source_packets_expected += len(alive_nodes)
        self.current_round += 1
        cluster_heads = self.cluster_head_selection()
        self.cluster_formation(cluster_heads)
        self.data_transmission_phase(cluster_heads)
        current_alive = len(alive_nodes)
        current_dead = len(self.nodes) - current_alive
        if current_dead > self.dead_nodes:
            self.dead_nodes = current_dead
            if self.network_lifetime == 0 and current_dead > 0:
                self.network_lifetime = self.current_round
        self.alive_nodes_per_round.append(current_alive)
        return True
    
    def run_simulation(self, max_rounds: int = 1000) -> Dict:
        """Run a complete LEACH simulation and return summary metrics."""
        print(f">>> Start LEACH protocol simulation (max rounds: {max_rounds})")
        for round_num in range(max_rounds):
            success = self.run_round()
            if not success:
                print(f"[WARN] Network lifetime ended at round {round_num}")
                break
            if round_num % 100 == 0:
                alive_count = len([n for n in self.nodes if n.is_alive])
                print(f"   Round {round_num}: alive nodes={alive_count}, total energy {self.total_energy_consumed:.3f}J")
        results = {
            'protocol_name': 'LEACH',
            'total_rounds': self.current_round,
            'network_lifetime': self.network_lifetime if self.network_lifetime > 0 else self.current_round,
            'total_energy_consumed': self.total_energy_consumed,
            'packets_sent': self.packets_sent,
            'packets_received': self.packets_received,
            'packet_delivery_ratio': self.packets_received / max(self.packets_sent, 1),
            'packet_delivery_ratio_end2end': self.total_bs_delivered / max(self.total_source_packets, 1),
            'bs_delivered': self.total_bs_delivered,
            'source_packets': self.total_source_packets,
            'dead_nodes': self.dead_nodes,
            'alive_nodes': len(self.nodes) - self.dead_nodes,
            'energy_consumption_per_round': self.energy_consumption_per_round,
            'alive_nodes_per_round': self.alive_nodes_per_round,
            'average_energy_per_round': self.total_energy_consumed / max(self.current_round, 1),
            'avg_hops_to_bs': (sum(self._all_hop_counts) / len(self._all_hop_counts)) if hasattr(self, '_all_hop_counts') and self._all_hop_counts else 0,
            'hop_count_distribution': dict((h, self._all_hop_counts.count(h)) for h in set(self._all_hop_counts)) if hasattr(self, '_all_hop_counts') and self._all_hop_counts else {},
        }
        print(f"[OK] LEACH simulation complete")
        print(f"   Network lifetime: {results['network_lifetime']} rounds")
        print(f"   Total energy: {results['total_energy_consumed']:.3f} J")
        print(f"   PDR: {results['packet_delivery_ratio']*100:.1f}%")
        return results


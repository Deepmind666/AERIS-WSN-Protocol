#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AERIS 能耗模型模块
- 基于 2024-2025 最新文献参数与测试数据

研究性质与命名声明：
- 本模块用于算法仿真，不涉及任何硬件实现或驱动。
- 硬件平台枚举与参数仅作为仿真配置的参考模板，并不代表对实际硬件性能的声明。
- 项目统一名称为 AERIS；EEHFR 名称已永久废弃。

作者: AERIS Research Team
日期: 2025-01-30
版本: 2.0
"""

import numpy as np
import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class HardwarePlatform(Enum):
    """硬件平台类型"""
    CC2420_TELOSB = "cc2420_telosb"      # 经典 WSN 平台
    CC2650_SENSORTAG = "cc2650_sensortag" # 现代低功耗节点
    ESP32_LORA = "esp32_lora"            # LoRa 平台
    GENERIC_WSN = "generic_wsn"          # 通用 WSN 节点

@dataclass
class EnergyParameters:
    """能耗参数配置"""
    # 发送能耗（基于 2024-2025 最新文献）
    tx_energy_per_bit: float      # J/bit
    rx_energy_per_bit: float      # J/bit
    
    # 处理能耗
    processing_energy_per_bit: float  # J/bit
    
    # 静态功耗参数
    idle_power: float             # W (空闲功率)
    sleep_power: float            # W (休眠功率)
    
    # 功放与路径损耗
    amplifier_efficiency: float   # 功放效率
    path_loss_threshold: float    # 路径损耗阈值 (m)

class ImprovedEnergyModel:
    """
    改进的 WSN 能耗模型：基于 2024-2025 最新文献的完整能耗建模
    """
    
    def __init__(self, platform: HardwarePlatform = HardwarePlatform.GENERIC_WSN):
        self.platform = platform
        self.params = self._get_platform_parameters(platform)
        
        # 硬件平台影响系数
        self.temperature_coefficient = 0.02  # 每 1°C 变化的能耗响应系数
        self.humidity_coefficient = 0.01     # 湿度对能耗的影响系数

    def _get_platform_parameters(self, platform: HardwarePlatform) -> EnergyParameters:
        """根据硬件平台获取能耗参数"""
        params_dict = {
            HardwarePlatform.CC2420_TELOSB: EnergyParameters(
                tx_energy_per_bit=208.8e-9,
                rx_energy_per_bit=225.6e-9,
                processing_energy_per_bit=5e-9,
                idle_power=1.4e-3,
                sleep_power=15e-6,
                amplifier_efficiency=0.5,
                path_loss_threshold=87.0,
            ),
            HardwarePlatform.CC2650_SENSORTAG: EnergyParameters(
                tx_energy_per_bit=16.7e-9,
                rx_energy_per_bit=36.1e-9,
                processing_energy_per_bit=3e-9,
                idle_power=0.8e-3,
                sleep_power=5e-6,
                amplifier_efficiency=0.45,
                path_loss_threshold=100.0,
            ),
            HardwarePlatform.ESP32_LORA: EnergyParameters(
                tx_energy_per_bit=200e-9,
                rx_energy_per_bit=100e-9,
                processing_energy_per_bit=10e-9,
                idle_power=2.5e-3,
                sleep_power=100e-6,
                amplifier_efficiency=0.25,
                path_loss_threshold=1000.0,
            ),
            HardwarePlatform.GENERIC_WSN: EnergyParameters(
                tx_energy_per_bit=50e-9,
                rx_energy_per_bit=50e-9,
                processing_energy_per_bit=5e-9,
                idle_power=1.0e-3,
                sleep_power=10e-6,
                amplifier_efficiency=0.35,
                path_loss_threshold=87.0,
            ),
        }
        return params_dict[platform]
    
    def calculate_transmission_energy(self, 
                                    data_size_bits: int,
                                    distance: float,
                                    tx_power_dbm: float = 0.0,
                                    temperature_c: float = 25.0,
                                    humidity_ratio: float = 0.5) -> float:
        """
        计算发送能耗
        Args:
            data_size_bits: 数据大小 (bits)
            distance: 传输距离 (m)
            tx_power_dbm: 发射功率 (dBm)
            temperature_c: 环境温度 (°C)
            humidity_ratio: 相对湿度 (0-1)
        Returns:
            发送能耗 (J)
        """
        
        # 基础发送能耗
        base_tx_energy = data_size_bits * self.params.tx_energy_per_bit
        
        # 发射功率线性
        tx_power_linear = 10**(tx_power_dbm / 10) / 1000  # 功率线性
        # 距离选择
        if distance <= self.params.path_loss_threshold:
            # 短距离: Pamp = 蔚_fs * d^2
            # 距离变化: 澧炲ぇ速率
            amplifier_energy = (tx_power_linear / self.params.amplifier_efficiency) * \
                             (distance ** 2) * 1e-9 * data_size_bits  # 速率
        else:
            # 长距离: Pamp = 蔚_mp * d^4
            amplifier_energy = (tx_power_linear / self.params.amplifier_efficiency) * \
                             (distance ** 4) * 1e-12 * data_size_bits  # 数据大小
        
        # 硬件平台影响系数
        temp_factor = 1 + self.temperature_coefficient * abs(temperature_c - 25.0)
        humidity_factor = 1 + self.humidity_coefficient * humidity_ratio
        
        total_energy = (base_tx_energy + amplifier_energy) * temp_factor * humidity_factor
        
        return total_energy
    
    def calculate_reception_energy(self, 
                                 data_size_bits: int,
                                 temperature_c: float = 25.0,
                                 humidity_ratio: float = 0.5) -> float:
        """
        计算接收能耗
        Args:
            data_size_bits: 数据大小 (bits)
            temperature_c: 环境温度 (°C)
            humidity_ratio: 相对湿度 (0-1)
        Returns:
            接收能耗 (J)
        """
        
        # 基础接收能耗
        base_rx_energy = data_size_bits * self.params.rx_energy_per_bit
        
        # 硬件平台影响系数
        temp_factor = 1 + self.temperature_coefficient * abs(temperature_c - 25.0)
        humidity_factor = 1 + self.humidity_coefficient * humidity_ratio
        
        total_energy = base_rx_energy * temp_factor * humidity_factor
        
        return total_energy
    
    def calculate_processing_energy(self, 
                                  data_size_bits: int,
                                  processing_complexity: float = 1.0) -> float:
        """
        计算数据处理能耗
        Args:
            data_size_bits: 数据大小 (bits)
            processing_complexity: 处理复杂度系数 (1.0 为默认)
        Returns:
            处理能耗 (J)
        """
        
        base_processing_energy = data_size_bits * self.params.processing_energy_per_bit
        return base_processing_energy * processing_complexity
    
    def calculate_idle_energy(self, idle_time_seconds: float) -> float:
        """
        计算空闲能耗
        Args:
            idle_time_seconds: 空闲时间 (秒)
        Returns:
            空闲能耗 (J)
        """
        return self.params.idle_power * idle_time_seconds
    
    def calculate_sleep_energy(self, sleep_time_seconds: float) -> float:
        """
        计算休眠能耗
        Args:
            sleep_time_seconds: 休眠时间 (秒)
        Returns:
            休眠能耗 (J)
        """
        return self.params.sleep_power * sleep_time_seconds
    
    def calculate_total_communication_energy(self,
                                           data_size_bits: int,
                                           distance: float,
                                           tx_power_dbm: float = 0.0,
                                           include_processing: bool = True,
                                           temperature_c: float = 25.0,
                                           humidity_ratio: float = 0.5) -> Dict[str, float]:
        """
        计算完整的通信能耗（发送+接收+处理，可选）
        Returns:
            包含各组件能耗的字典
        """
        
        tx_energy = self.calculate_transmission_energy(
            data_size_bits, distance, tx_power_dbm, temperature_c, humidity_ratio
        )
        
        rx_energy = self.calculate_reception_energy(
            data_size_bits, temperature_c, humidity_ratio
        )
        
        processing_energy = 0.0
        if include_processing:
            processing_energy = self.calculate_processing_energy(data_size_bits)
        
        total_energy = tx_energy + rx_energy + processing_energy
        
        return {
            'transmission_energy': tx_energy,
            'reception_energy': rx_energy,
            'processing_energy': processing_energy,
            'total_energy': total_energy,
            'energy_breakdown': {
                'tx_percentage': (tx_energy / total_energy) * 100,
                'rx_percentage': (rx_energy / total_energy) * 100,
                'processing_percentage': (processing_energy / total_energy) * 100
            }
        }
    
    def get_energy_efficiency_metrics(self, 
                                    data_size_bits: int,
                                    distance: float) -> Dict[str, float]:
        """
        计算能耗效率
        
        Returns:
            能耗效率
        """
        
        energy_result = self.calculate_total_communication_energy(data_size_bits, distance)
        
        # 能耗效率
        energy_per_bit = energy_result['total_energy'] / data_size_bits
        energy_per_meter = energy_result['total_energy'] / distance if distance > 0 else 0
        
        return {
            'energy_per_bit': energy_per_bit,           # J/bit
            'energy_per_meter': energy_per_meter,       # J/m
            'energy_per_byte': energy_per_bit * 8,      # J/byte
            'platform': self.platform.value,
            'total_energy': energy_result['total_energy']
        }

# 测试入口
def test_energy_model():
    """测试改进的能耗模型"""
    print("Test energy model")
    print("=" * 50)

    platforms = [
        HardwarePlatform.CC2420_TELOSB,
        HardwarePlatform.CC2650_SENSORTAG,
        HardwarePlatform.ESP32_LORA,
    ]

    test_data_size = 1024  # 1KB
    test_distance = 50.0   # 50 米

    for platform in platforms:
        print(f"\nPlatform: {platform.value}")
        energy_model = ImprovedEnergyModel(platform)
        energy_result = energy_model.calculate_total_communication_energy(
            test_data_size * 8,  # 转换为 bits
            test_distance,
        )
        print(f"   Transmission energy: {energy_result['transmission_energy']*1e6:.2f} uJ")
        print(f"   Reception energy: {energy_result['reception_energy']*1e6:.2f} uJ")
        print(f"   Processing energy: {energy_result['processing_energy']*1e6:.2f} uJ")
        print(f"   Total energy: {energy_result['total_energy']*1e6:.2f} uJ")

        efficiency = energy_model.get_energy_efficiency_metrics(
            test_data_size * 8, test_distance
        )
        print(f"   Energy efficiency: {efficiency['energy_per_bit']*1e9:.2f} nJ/bit")

if __name__ == "__main__":
    test_energy_model()


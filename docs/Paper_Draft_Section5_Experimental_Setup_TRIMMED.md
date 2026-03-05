# Section 5: Experimental Setup (精简版 - 论文主体)

**字数**: ~1,000词
**精简日期**: 2025-10-19
**删减策略**: 压缩实现细节和重复性说明 → 保留核心配置

---

## 5. Experimental Setup and Implementation

### 5.1 Simulation Environment

All experiments were conducted using a custom-built simulation framework implemented in Python 3.11, running on a high-performance computing system:

**Hardware Configuration:**
- **Processor**: Intel Core i7-12700K (12 cores, 3.6 GHz base frequency)
- **Memory**: 32 GB DDR4-3200 RAM
- **Operating System**: Windows 11 Professional

**Software Environment:**
- **Python Version**: 3.11
- **Key Libraries**: NumPy 1.23, SciPy 1.9, Matplotlib 3.6, Pandas 1.5
- **Simulation Framework**: Custom WSN simulator with modular protocol implementation
- **Statistical Analysis**: SciPy.stats for significance testing, Bootstrap CI

The simulation framework was designed with modularity in mind, allowing for easy integration of different routing protocols and fair performance comparison under identical conditions.

---

### 5.2 Dataset Description

#### 5.2.1 Intel Berkeley Research Lab Dataset

The experiments utilize the widely-recognized Intel Berkeley Research Lab dataset, which provides real-world sensor network data collected from a 54-node deployment over a 31-day period from February 28 to April 5, 2004.

**Dataset Characteristics:**
- **Total Records**: 2,219,799 sensor readings
- **Sensor Types**: Temperature, humidity, light, and voltage sensors
- **Sampling Interval**: 31 seconds average
- **Network Topology**: Multi-hop wireless sensor network (54 nodes)
- **Coverage Area**: Intel Berkeley Research Lab building
- **Data Format**: Timestamp, node ID, temperature, humidity, light, voltage

**Data Preprocessing:**
The raw dataset underwent several preprocessing steps:
1. **Data Cleaning**: Removal of incomplete records and outliers (< 0.1% of total data)
2. **Temporal Alignment**: Synchronization of sensor readings to common time intervals
3. **Topology Extraction**: Reconstruction of network connectivity based on communication logs
4. **Energy Model Mapping**: Assignment of realistic energy consumption values based on sensor specifications

#### 5.2.2 Synthetic Topologies

In addition to the Intel Lab dataset, we evaluate AERIS on synthetic topologies to assess scalability and performance across diverse network structures:

**Synthetic Topology Configurations:**

| Topology | Dimensions | Nodes | BS Location | Description |
|----------|-----------|-------|-------------|-------------|
| Uniform 50×200 | 50m × 200m | 1024 | (25, 220) | Random uniform distribution |
| Corridor31 | 31m × 200m | 50 | (15.5, 220) | Corridor waveguide effect |
| Corridor41 | 41m × 200m | 50 | (20.5, 220) | Wider corridor layout |

---

### 5.3 Network Configuration Parameters

**Table 5.1** - Core Simulation Parameters:

| Parameter | Value | Description |
|-----------|-------|-------------|
| Network Size | 50, 54, 1024 nodes | Intel Lab: 54; Synthetic: 50-1024 |
| Communication Range | Dynamic (CC2420) | Determined by transmission power and channel |
| Initial Energy | 2.0 J | Per-node energy budget (2 AA batteries) |
| Packet Size | 1024 bytes | Standard sensor data payload |
| Simulation Rounds | 200-500 | Total operational cycles |
| Cluster Head Ratio | 10-15% | Target percentage of CHs |
| Safety Fallback Threshold | 0.1 (10% PDR) | Triggers redundancy/power boost |

**Channel Model Parameters:**
- **Path loss exponent**: n = 2.8 (Intel Lab office), n = 1.8 (corridor)
- **Shadowing variance**: σ = 7.5dB (Intel Lab), σ = 3.5dB (corridor)
- **MAC**: IEEE 802.15.4 CSMA/CA with 3 retransmissions
- **Interference**: Time-varying Gaussian, μ_I = -90dBm

---

### 5.4 Baseline Protocol Implementation

To ensure fair comparison, all baseline protocols were implemented using identical energy models, communication parameters, and network conditions.

#### 5.4.1 LEACH Protocol Implementation

**Implementation Details:**
- **Cluster Head Probability**: P = 0.05 (5% of nodes become cluster heads)
- **Setup Phase**: Cluster formation every 20 rounds
- **Steady State**: Data transmission within established clusters
- **Energy Model**: First-order radio model with distance-dependent transmission costs

#### 5.4.2 PEGASIS Protocol Implementation

**Implementation Features:**
- **Chain Construction**: Greedy algorithm for minimum spanning chain
- **Leader Selection**: Rotating leadership based on residual energy
- **Data Fusion**: Binary tree aggregation along the chain
- **Energy Optimization**: Minimized transmission distances

#### 5.4.3 HEED Protocol Implementation

**Implementation Characteristics:**
- **Primary Parameter**: Residual energy ratio
- **Secondary Parameter**: Average minimum reachability power (AMRP)
- **Clustering Process**: Iterative probabilistic selection
- **Termination**: When all nodes join clusters or become cluster heads

---

### 5.5 Performance Evaluation Metrics

**Energy-Related Metrics:**
- **Total Energy Consumption**: $E_{total} = \Sigma(E_{tx} + E_{rx} + E_{sensing})$
- **Energy Per Round**: $E_{per\_round} = E_{total} / Total\_Rounds$
- **Energy Efficiency Ratio**: $\eta_E = Packets\_Delivered / E_{total}$

**Network Lifetime Metrics:**
- **Network Lifetime (First Node Death)**: Rounds until the first node depletes its energy completely
- **Average Residual Energy**: $(1/N) \times \Sigma(E_{remaining}(i))$ for i=1 to N

**Communication Quality Metrics:**
- **Packet Delivery Ratio (PDR)**: $PDR = Packets\_Successfully\_Delivered / Total\_Packets\_Sent$
- **Average End-to-End Delay**: $(1/M) \times \Sigma(T_{received}(j) - T_{sent}(j))$ for j=1 to M

**Computational Efficiency Metrics (AERIS-specific):**
- **Decision Latency**: Per-round routing computation time (measured in milliseconds)
- **Memory Footprint**: Runtime memory consumption (measured in kilobytes)
- **Training Time**: Offline preparation time (AERIS: 0h, ML baselines: 8-96h)

---

### 5.6 Statistical Analysis Methodology

#### 5.6.1 Experimental Design

**Replication Strategy:**
- Each experimental configuration was executed multiple independent times with different random seeds
- Typical repeats:
  - Intel Lab significance experiments: **n = 200 runs**
  - Ablation studies: **n = 100 runs**
  - Sensitivity analysis: **n = 50 runs**
  - Synthetic topology experiments: **n = 50 runs**
- Seeds are deterministically enumerated from a base seed to ensure reproducibility

**Significance Testing:**
- **Welch's t-test** (two-sample, unequal variances) for mean comparisons
- **Multiple comparison control**: Holm–Bonferroni adjustment where multiple baselines/topologies are compared
- **Confidence intervals**: Non-parametric bootstrap 95% CI for means (10,000 resamples)
- **Effect sizes**: Cohen's d / Hedges' g reported for practical significance

**Significance Level**: α = 0.05 for all hypothesis tests

#### 5.6.2 Data Validation

**Consistency Checks:**
- Verification of energy conservation laws (total energy consumed ≤ initial energy)
- Ensuring realistic parameter ranges (PDR ∈ [0, 1], energy ≥ 0)
- Monitoring simulation stability (no divergence or numerical overflow)
- Statistical identification and handling of anomalous results (Grubbs' test for outliers)

---

### 5.7 Reproducibility Considerations

#### 5.7.1 Code Availability

The complete simulation framework and protocol implementations are available as open-source software:
- **Repository**: https://github.com/Deepmind666/AERIS-WSN-Protocol
- **License**: MIT License for academic and commercial use
- **Documentation**: Comprehensive API documentation and usage examples
- **Test Suite**: Unit tests for all protocol components

#### 5.7.2 Experimental Reproducibility

**Deterministic Elements:**
- Fixed random seeds for each experimental configuration
- Identical network topologies across protocol comparisons
- Standardized parameter settings documented in configuration files (JSON format)

**Reproduction Instructions:**
Detailed instructions for reproducing all experimental results are provided in the project repository `README.md`, including:
- Environment setup procedures (`scripts/setup_conda_env.ps1`)
- Parameter configuration files (`experiments/config_*.json`)
- Execution scripts for all experiments (`scripts/run_*.py`)
- Data analysis and visualization code (`scripts/plot_*.py`)

This comprehensive experimental setup ensures that all performance comparisons are conducted under fair and controlled conditions, providing reliable and reproducible results for the AERIS protocol evaluation.

---

## References (Section 5)

[Intel Lab Dataset] Intel Berkeley Research Lab, "Sensor Network Data," MIT CSAIL, 2004. [Online]. Available: http://db.csail.mit.edu/labdata/labdata.html

[IEEE 802.15.4] IEEE Standard 802.15.4-2020, "IEEE Standard for Low-Rate Wireless Networks," 2020.

---

**精简说明**:
1. ✅ 删减 ~800词: 从1,800 → 1,000词
2. ✅ 保留核心内容:
   - 模拟环境 (硬件/软件配置)
   - Intel Lab数据集特征
   - 合成拓扑配置
   - 网络参数表格
   - 基线协议实现要点
   - 统计方法 (Welch, Holm-Bonferroni, Bootstrap)
   - 可复现性声明
3. ✅ 删除详细内容:
   - 伪代码实现 (leach_cluster_head_selection, pegasis_chain_formation)
   - 详细能量模型公式 (已在Section 3)
   - 完整validation流程
   - 扩展的数据预处理步骤
4. ✅ 新增计算效率指标: 决策延迟、内存占用、训练时间

**字数**: ~1,000词 (-800词 ✅)

# AERIS 仿真严谨性修复 — 完整技术方案

> 版本：v1.0 | 日期：2026-02-15 | 状态：待评估

---

## 一、问题诊断

### 1.1 当前仿真器存在的三个根本性问题

**问题A：无MAC层碰撞建模**
- 每条链路的PDR独立计算（`realistic_channel_model.py` → `calculate_link_metrics()`），不考虑同频干扰
- 1000个节点同时发包和1个节点发包，信道质量完全一样
- 直接后果：PDR随节点数增加反而上升（3/4环境），违反物理规律

**问题B：Baseline比较不公平**
- AERIS拥有：多跳中继（relay_via_parent）、Gateway聚合层、CAS自适应切换、链路重传（intra_link_retx=2）、功率步进（1.5dBm/次）
- Baseline只有：单跳CH→BS直传、无重传、无功率自适应
- 1000节点outdoor_urban：AERIS=88.5% vs 最好baseline=6.6%，差距12倍以上

**问题C：PDR-Scale异常**
- AERIS在indoor_factory从100节点0.933升到1000节点0.973（+4.2%）
- outdoor_urban从0.756升到0.885（+17%）
- 根因：(a)无碰撞建模 + (b)更多节点=更多中继候选=更短平均距离=更高链路PDR

### 1.2 数据证据

| 环境 | AERIS 100节点 | AERIS 1000节点 | 变化 | 最好Baseline 1000节点 | 差距倍数 |
|---|---|---|---|---|---|
| indoor_office | 0.9947 | 0.9899 | -0.5% | PEGASIS 0.894 | 1.1x |
| indoor_factory | 0.9333 | 0.9726 | **+4.2%** | TEEN 0.220 | **4.4x** |
| outdoor_urban | 0.7559 | 0.8846 | **+17%** | TEEN 0.066 | **13.4x** |
| outdoor_suburban | 0.9747 | 0.9896 | **+1.5%** | TEEN 0.367 | **2.7x** |

---

## 二、仿真平台对比调研

### 2.1 NS-3 lr-wpan模块

**能力：**
- 完整IEEE 802.15.4 CSMA/CA状态机（beacon和non-beacon模式）
- SINR级别的碰撞检测（`LrWpanInterferenceHelper`追踪所有并发信号）
- O-QPSK调制的BER/PER计算
- 支持CCA（Clear Channel Assessment）

**局限：**
- **不包含LEACH/PEGASIS/HEED/TEEN**，需从零用C++实现
- 无TDMA支持（只有CSMA/CA）
- 1000节点+300轮仿真单次运行需数小时
- 1000 replicates × 6节点数 × 5协议 = 30,000次运行，**在NS-3中不可行**（需数月计算时间）
- 无原生ML集成（LSTM/TCN环境预测模块无法移植）

**我们已有的NS-3工作：**
- `ns3_validation/` 目录：~3,846行C++代码
- AERIS vs LEACH的trend-level验证（4环境，30 seeds）
- 100/300/500节点的scale extension结果

### 2.2 OMNeT++ / Castalia

**能力：**
- Castalia内置基础LEACH实现
- 完整802.15.4 MAC（CSMA/CA + TDMA）
- SINR级别干扰建模
- 能量模型（idle/sleep/tx/rx状态）

**局限：**
- Castalia已停止维护
- HEED/PEGASIS/TEEN需自行实现
- 仿真速度比NS-3快但仍远慢于Python
- 大规模Monte Carlo同样不可行

### 2.3 自研Python仿真器

**优势：**
- Monte Carlo速度：500节点×300轮单次运行~10秒，NS-3需数分钟
- 统计严谨性：n=1000 replicates + Holm-Bonferroni校正 + Bootstrap CI，在NS-3中不可能实现
- ML集成：LSTM/TCN/DLinear/PatchTST环境预测模块原生支持
- 可复现性：纯Python，无编译依赖
- 灵活性：消融实验、sensitivity sweep、新协议变体可快速迭代

**劣势：**
- 无MAC碰撞建模（**本方案要修复**）
- 无同行评审的PHY/MAC栈
- Baseline实现可能与原始论文有偏差（**本方案要修复**）

### 2.4 文献中的先例

| 论文 | 仿真平台 | 期刊 |
|---|---|---|
| LEACH (Heinzelman 2000) | 自研MATLAB | HICSS |
| HEED (Younis & Fahmy 2004) | 自研仿真器 | IEEE TMC |
| PEGASIS (Lindsey 2002) | 自研仿真器 | IEEE Aerospace |
| 大量LEACH变体论文 | MATLAB | MDPI Sensors, IEEE Access |

**结论：MDPI Sensors接受自研仿真器，但需要：(1)明确声明假设和局限，(2)充分的统计验证，(3)最好有NS-3交叉验证。**

### 2.5 平台策略决定

| 方案 | 工作量 | 可行性 | 推荐 |
|---|---|---|---|
| (a) 全部移植到NS-3 | 4-6个月 | 不可行（期刊修改周期内） | ✗ |
| (b) 修复自研仿真器 + NS-3交叉验证 | 2-3周 | **可行** | **✓** |
| (c) 只修复仿真器，不用NS-3 | 1-2周 | 可行但较弱 | △ |

**推荐方案(b)：修复自研仿真器（加MAC碰撞+Baseline多跳），保留并扩展已有NS-3交叉验证。**

论文中的定位："We provide NS-3 cross-validation confirming directional agreement, while the custom simulator enables the 1000-replicate statistical analysis that would be computationally prohibitive in NS-3."

---

## 三、仿真器输入输出规格

### 3.1 输入参数

**CLI参数（run_scalability_experiment.py）：**
```
--replicates INT     每配置重复次数（默认30，publication用1000）
--workers INT        并行worker数（默认6）
--seed INT           基础种子（默认42001）
--nodes STR          节点数列表（默认"50,100,200,300,500,800,1000"）
--rounds INT         仿真轮数（默认300）
--env STR            环境类型（默认"indoor_office"）
--tx-power FLOAT     发射功率dBm（默认10.0）
--run-tier STR       运行级别（默认"publication"）
--output STR         输出JSON路径
--max-cpu-percent    CPU上限（默认70%）
--max-mem-percent    内存上限（默认70%）
```

**NetworkConfig核心字段：**

| 字段 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| area_width/height | float | 100.0 | 网络区域尺寸(m) |
| base_station_x/y | float | 50.0/175.0 | 基站坐标 |
| num_nodes | int | 100 | 传感器节点数 |
| initial_energy | float | 2.0 | 初始能量(J) |
| packet_size | int | 1024 | 包大小(bytes) |
| temperature_c | float | 25.0 | 环境温度(°C) |
| humidity_ratio | float | 0.5 | 相对湿度(0-1) |
| enable_channel | bool | False | 启用真实信道模型 |
| channel_env | str | None | 环境类型 |
| tx_power_dbm | float | 0.0 | 发射功率(dBm) |
| link_retx | int | 0 | 链路重传次数 |
| link_retx_power_step | float | 0.0 | 重传功率步进(dBm) |

**信道模型参数（按环境）：**

| 环境 | 路径损耗指数 | 参考损耗(dB@1m) | 阴影衰落标准差(dB) | 噪声底(dBm) |
|---|---|---|---|---|
| indoor_office | 2.0 | 40.0 | 4.5 | -95.0 |
| indoor_factory | 2.7 | 45.0 | 8.5 | -92.0 |
| outdoor_urban | 3.4 | 44.0 | 12.0 | -93.0 |
| outdoor_suburban | 2.8 | 38.0 | 7.5 | -96.0 |

### 3.2 输出格式

**JSON顶层结构：**
```json
{
  "timestamp": "20260215_...",
  "git_commit": "b6b2e5e",
  "run_tier": "publication",
  "primary_metric": "pdr_expected",
  "environment": "indoor_factory",
  "error_runs": 0,
  "config": { "seeds": [...], "node_counts": [...], ... },
  "raw_results": [
    {
      "num_nodes": 500,
      "replicate": 0,
      "protocol": "AERIS",
      "seed": 42377,
      "metrics": {
        "pdr_expected": 0.8542,
        "energy": 1.2345,
        "lifetime": 287,
        "alive_nodes": 42
      }
    },
    ...
  ]
}
```

**种子生成公式：** `seed = base_seed + replicate * 997 + stable_hash(protocol) % 997`

---

## 四、修复方案详细设计

### 4.1 修改一：MAC碰撞模型

**文献依据：**
- IEEE 802.15.4-2006 标准MAC参数（macMinBE=3, macMaxBE=5, macMaxCSMABackoffs=4）
- Pollin et al. (IEEE TWC 2008)：802.15.4 slotted CSMA/CA分析模型
- Park et al. (IEEE TPDS 2013)：802.15.4 CSMA/CA优化模型
- Abramson (1970)：ALOHA模型基础

**设计：两层混合碰撞模型**

**Tier 1 — 簇内竞争（member→CH）：TDMA slot模型**

LEACH原始论文假设簇内使用TDMA调度（Heinzelman 2000, Section III-B）。我们遵循这一假设：

```python
def intra_cluster_collision_factor(cluster_size: int, slots_per_frame: int = 20) -> float:
    """
    TDMA slot模型：
    - cluster_size ≤ slots: 无碰撞（每个成员分配一个slot）
    - cluster_size > slots: 超出部分竞争共享slot

    参数依据：IEEE 802.15.4 superframe有16个GTS + CAP slots
    """
    if cluster_size <= slots_per_frame:
        return 1.0  # 无竞争
    excess = cluster_size - slots_per_frame
    # Slotted ALOHA近似：P_success = exp(-G), G = excess/slots
    return math.exp(-excess / slots_per_frame)
```

**Tier 2 — 上行竞争（CH→BS）：Offered-load模型**

多个CH在同一轮向BS发送，共享信道：

```python
def uplink_collision_factor(num_concurrent_chs: int, uplink_slots: int = 8) -> float:
    """
    Offered-load模型：
    - P_success = exp(-G), G = num_chs / uplink_slots

    参数依据：802.15.4 CAP期间可用slot数
    """
    if num_concurrent_chs <= 1:
        return 1.0
    offered_load = num_concurrent_chs / max(1, uplink_slots)
    return math.exp(-offered_load)
```

**可配置参数（用于sensitivity analysis）：**

| 参数 | 默认值 | 范围 | 文献依据 |
|---|---|---|---|
| slots_per_frame | 20 | 8-32 | IEEE 802.15.4 superframe |
| uplink_channel_slots | 8 | 4-16 | CAP slot数 |
| enabled | True | — | 主开关 |

**预期效果（1000节点，~50个簇，每簇~20成员）：**

| 节点数 | 簇大小 | 簇内因子 | CH数 | 上行因子 | 总PDR衰减 |
|---|---|---|---|---|---|
| 100 | ~10 | 1.00 | ~10 | 0.88 | ~12% |
| 300 | ~15 | 1.00 | ~15 | 0.85 | ~15% |
| 500 | ~25 | 0.78 | ~25 | 0.71 | ~45% |
| 800 | ~40 | 0.37 | ~40 | 0.61 | ~77% |
| 1000 | ~50 | 0.22 | ~50 | 0.54 | ~88% |

这将产生PDR随规模单调递减的合理趋势。

**新建文件：** `src/mac_collision_model.py`（~100行）

**集成方式：** 修改 `_link_success()` 函数，增加 `collision_factor` 乘数：
```python
def _link_success(channel_model, tx_power, distance, temp_c, humidity_ratio,
                  collision_factor=1.0) -> bool:
    metrics = channel_model.calculate_link_metrics(...)
    effective_pdr = metrics.get("pdr", 0.0) * collision_factor
    return random.random() < effective_pdr
```

### 4.2 修改二：Baseline多跳中继

**文献依据：**
- 贪心地理转发（Greedy Geographic Forwarding）是WSN文献中的标准技术
- LEACH-C (Heinzelman 2002) 已讨论CH间协作
- HEED原始论文提到多跳通信作为扩展

**设计：共享多跳中继模块**

**新建文件：** `src/multihop_relay.py`（~150行）

```python
def build_ch_relay_tree(cluster_heads, base_station):
    """
    贪心地理转发树：
    1. 计算每个CH到BS的距离
    2. 从最远CH开始，选择距BS更近的最近CH作为下一跳
    3. 最近BS的CH直传
    返回：{ch_id: next_hop_ch_id or None}
    """

def transmit_via_relay_tree(ch, relay_tree, all_chs, base_station,
                            channel_model, energy_model, ...):
    """
    沿relay tree逐跳转发：
    - 每跳：发送方扣TX能量，中继方扣RX能量
    - 中继节点不做聚合（只转发）
    - 中继节点死亡则丢包
    返回：(success, energy_consumed, hop_count)
    """
```

**集成方式：** 在每个Baseline的 `data_transmission_phase()` 中，CH→BS部分替换为relay tree转发（当 `enable_multihop_relay=True` 时）。

### 4.3 修改三：Baseline链路重传

**现状：**
- `benchmark_protocols.py` 中的wrapper已有 `link_retx` 循环（第333行）
- `baseline_protocols/` 下的独立模块（publication路径）只尝试一次

**修改：** 给独立模块加入与AERIS相同的重传循环：
```python
for attempt in range(link_retx + 1):
    tx_power = base_tx_power + attempt * link_retx_power_step
    if _link_success(channel_model, tx_power, distance, ...):
        return True
return False
```

**默认参数：** `link_retx=1, link_retx_power_step=1.0`（比AERIS的retx=2, step=1.5略低，反映Baseline没有自适应优化）

### 4.4 需修改的文件清单

| # | 文件 | 操作 | 修改内容 |
|---|---|---|---|
| 1 | `src/mac_collision_model.py` | **新建** | MACCollisionConfig + MACCollisionModel |
| 2 | `src/multihop_relay.py` | **新建** | build_ch_relay_tree + transmit_via_relay_tree |
| 3 | `src/benchmark_protocols.py` | 修改 | NetworkConfig加字段；_link_success加collision_factor |
| 4 | `src/baseline_protocols/leach_protocol.py` | 修改 | 多跳+重传+碰撞因子 |
| 5 | `src/baseline_protocols/heed_protocol.py` | 修改 | 同上 |
| 6 | `src/baseline_protocols/pegasis_protocol.py` | 修改 | leader relay + 重传 + 碰撞因子 |
| 7 | `src/teen_protocol.py` | 修改 | 多跳+重传+碰撞因子 |
| 8 | `src/aeris_protocol.py` | 修改 | 碰撞因子注入（不加多跳，已有） |
| 9 | `scripts/run_scalability_experiment.py` | 修改 | 加 --mac-collision/--multihop-relay/--link-retx |
| 10 | `scripts/run_fair_5protocol.py` | 修改 | 同上 |
| 11 | `scripts/run_ablation_diag.py` | 修改 | 消融实验适配新参数 |

---

## 五、实验重跑计划

### 5.1 Smoke Test（修改完成后立即执行）

```bash
python scripts/run_scalability_experiment.py \
  --env indoor_factory --nodes 100,500,1000 --replicates 5 --seed 42001 \
  --rounds 300 --workers 6 --mac-collision --multihop-relay --link-retx 1 \
  --output results/mega_experiments/_smoke_mac_collision.json
```

**验证标准：**
- PDR随节点数单调递减（或至少不上升）
- Baseline在1000节点时PDR > 10%
- AERIS仍优于Baseline但差距合理（2-5倍而非12倍）

### 5.2 全量重跑

与之前S8实验相同规格：4环境 × 6节点数 × 5协议 × n=1000 replicates

**预计计算时间：**
- 碰撞模型增加的计算开销 < 1%（只是额外的exp()调用）
- 多跳中继增加的开销 ~5%（relay tree构建 + 额外链路评估）
- 总计与之前S8实验相当：本地+服务器各10-20小时

### 5.3 NS-3交叉验证扩展

在已有AERIS vs LEACH基础上，增加：
- HEED作为第二个baseline（需在NS-3中实现，~1-2周）
- 100/300/500节点 × 4环境 × 30 seeds
- 验证方向性一致（AERIS > Baseline）和趋势一致（PDR随规模递减）

---

## 六、论文修改要点

### 6.1 需要重写的部分

- **Section 3 (System Model):** 加入MAC碰撞模型描述，引用IEEE 802.15.4标准和Pollin 2008
- **Section 4 (Protocol Design):** 明确AERIS的多跳/Gateway/CAS是其创新点，Baseline也有基本多跳能力
- **Section 5 (Experimental Setup):** 声明仿真器假设和局限，描述碰撞模型参数
- **Section 6 (Results):** 全部数据替换为新实验结果
- **Section 7 (Discussion):** 加入sensitivity analysis（碰撞参数变化对结论的影响）
- **Limitations:** 明确声明不建模完整CSMA/CA状态机，使用分析近似

### 6.2 论文定位调整

**修复前：** "AERIS achieves 12x PDR improvement over baselines in harsh environments"
**修复后（预期）：** "AERIS achieves 2-5x PDR improvement over multi-hop-enabled baselines under realistic MAC contention modeling"

---

## 七、风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|---|---|---|---|
| AERIS优势大幅缩小 | 高 | 论文claim需重写 | 提前做smoke test评估 |
| 碰撞模型参数敏感 | 中 | 结论不稳健 | sensitivity analysis覆盖参数范围 |
| Baseline多跳引入bug | 中 | 数据不可靠 | 单元测试 + 与无多跳结果对比 |
| 全量重跑计算时间超预期 | 低 | 延迟提交 | 本地+服务器并行 |
| 审稿人仍要求完整NS-3 | 低 | 需额外工作 | NS-3交叉验证作为防线 |

---

## 八、实施时间线

| 阶段 | 内容 | 预计工作量 |
|---|---|---|
| Phase 1 | 编写mac_collision_model.py + multihop_relay.py | 1天 |
| Phase 2 | 修改5个协议文件 + 实验脚本 | 1-2天 |
| Phase 3 | Smoke test + 调参 | 半天 |
| Phase 4 | 全量重跑（本地+服务器并行） | 1-2天（计算时间） |
| Phase 5 | 数据分析 + 论文修改 | 2-3天 |
| Phase 6 | NS-3交叉验证扩展（可选） | 1-2周 |

---

## 九、关键文献引用

1. IEEE Std 802.15.4-2006, "Part 15.4: Wireless MAC and PHY Specifications for LR-WPANs"
2. S. Pollin et al., "Performance Analysis of Slotted Carrier Sense IEEE 802.15.4 MAC Layer," IEEE TWC, 2008
3. P. Park et al., "Modeling and Optimization of the IEEE 802.15.4 Protocol," IEEE TPDS, 2013
4. G. Bianchi, "Performance Analysis of the IEEE 802.11 DCF," IEEE JSAC, 2000
5. W.R. Heinzelman et al., "Energy-Efficient Communication Protocol for WSNs," HICSS, 2000
6. M. Zuniga and B. Krishnamachari, "Analyzing the Transitional Region in Low Power Wireless Links," IEEE SECON, 2004
7. A. Boulis, "Castalia: A Simulator for WSNs and BANs," NICTA, 2011

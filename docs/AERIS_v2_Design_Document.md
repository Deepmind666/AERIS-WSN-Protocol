# AERIS v2 算法设计文档

**版本**: 2.0 Draft
**日期**: 2026-01-26
**作者**: Claude Opus 4.5 & Kangrui Li

---

## 1. 当前 AERIS v1 分析

### 1.1 核心架构

当前 AERIS v1 采用三层架构：

```
┌─────────────────────────────────────────────────────────┐
│                    Base Station (BS)                     │
└─────────────────────────────────────────────────────────┘
                           ▲
                           │ Gateway Uplink
┌─────────────────────────────────────────────────────────┐
│              Gateway Layer (k=1~3 CHs)                   │
│         - 距离BS最近的CH作为网关                          │
│         - 负责聚合远端CH数据                              │
└─────────────────────────────────────────────────────────┘
                           ▲
                           │ Skeleton Relay
┌─────────────────────────────────────────────────────────┐
│              Skeleton Layer (骨干路由)                    │
│         - PCA主轴选择骨干CH                              │
│         - 远端CH通过骨干中继                              │
└─────────────────────────────────────────────────────────┘
                           ▲
                           │ CAS Mode Selection
┌─────────────────────────────────────────────────────────┐
│              Cluster Layer (簇内通信)                     │
│         - CAS: Direct / Chain / TwoHop                   │
│         - 基于环境特征动态选择模式                         │
└─────────────────────────────────────────────────────────┘
```

### 1.2 核心组件

| 组件 | 功能 | 当前实现 |
|------|------|----------|
| **CAS** | 簇内传输模式选择 | 线性加权评分，EMA平滑 |
| **Gateway** | 网关CH选择 | 距离+中心性评分 |
| **Skeleton** | 骨干路由选择 | PCA主轴+中心性 |
| **Safety** | 可靠性保障 | 阈值触发回退 |

### 1.3 性能瓶颈分析

基于实验数据（修复后的真实PDR）：

| 规模 | AERIS PDR | 问题分析 |
|------|-----------|----------|
| 100节点 | 99.2% | 表现优秀 |
| 200节点 | 88.6% | 开始下降 |
| 300节点 | 83.5% | 明显下降 |
| 500节点 | 79.7% | 大规模瓶颈 |

**识别的关键瓶颈**：

1. **Gateway 单点瓶颈**: k=1 时网关负载过重
2. **CAS 静态权重**: 无法适应动态网络变化
3. **Skeleton 覆盖不足**: 大规模时骨干节点不够
4. **缺乏预测能力**: 被动响应而非主动预测
5. **能量感知不足**: 未充分利用节点能量状态

---

## 2. AERIS v2 设计目标

### 2.1 核心目标

| 目标 | 指标 | v1 现状 | v2 目标 |
|------|------|---------|---------|
| **可靠性** | PDR@500节点 | 79.7% | ≥90% |
| **可扩展性** | PDR下降率 | ~20% | ≤10% |
| **能效** | 能耗/成功包 | 较高 | 降低20% |
| **自适应** | 动态响应 | 被动 | 主动预测 |

### 2.2 设计原则

1. **轻量化优先**: 适合资源受限节点
2. **可解释性**: 避免黑盒ML模型
3. **渐进增强**: 兼容v1，可选启用新特性
4. **数据驱动**: 基于实验反馈迭代优化

---

## 3. AERIS v2 核心改进

### 3.1 改进一：自适应多网关机制 (Adaptive Multi-Gateway, AMG)

**问题**: v1 使用固定 k=1 网关，大规模时成为瓶颈

**解决方案**: 动态调整网关数量

```python
# 伪代码
def adaptive_gateway_count(num_nodes, num_chs, current_pdr):
    base_k = max(1, num_chs // 10)  # 基础：每10个CH配1个网关

    # PDR反馈调节
    if current_pdr < 0.85:
        k = base_k + 1  # PDR低时增加网关
    elif current_pdr > 0.95:
        k = max(1, base_k - 1)  # PDR高时减少网关节能
    else:
        k = base_k

    # 规模约束
    k = min(k, num_chs // 3)  # 最多1/3的CH作为网关
    return max(1, k)
```

**预期效果**: 500节点时 PDR 提升 5-8%

### 3.2 改进二：在线CAS权重学习 (Online CAS Learning, OCL)

**问题**: v1 CAS 使用静态权重，无法适应不同网络条件

**解决方案**: 基于 Bandit 算法的在线权重调整

```python
class OnlineCASLearner:
    def __init__(self, learning_rate=0.1):
        self.lr = learning_rate
        self.mode_rewards = {
            'DIRECT': deque(maxlen=50),
            'CHAIN': deque(maxlen=50),
            'TWO_HOP': deque(maxlen=50)
        }

    def update(self, mode, success, energy_cost):
        # 奖励 = PDR成功 - 能耗惩罚
        reward = (1.0 if success else 0.0) - 0.1 * energy_cost
        self.mode_rewards[mode].append(reward)

    def get_mode_bonus(self, mode):
        rewards = self.mode_rewards[mode]
        if len(rewards) < 5:
            return 0.0
        return np.mean(rewards) * self.lr
```

**预期效果**: 动态场景 PDR 提升 3-5%

### 3.3 改进三：预测性链路质量估计 (Predictive Link Quality, PLQ)

**问题**: v1 仅使用当前链路质量，无法预测即将发生的链路退化

**解决方案**: 基于滑动窗口的链路质量趋势预测

```python
class PredictiveLinkQuality:
    def __init__(self, window_size=10):
        self.history = deque(maxlen=window_size)

    def update(self, lqi):
        self.history.append(lqi)

    def predict_next(self):
        if len(self.history) < 3:
            return self.history[-1] if self.history else 0.5

        # 简单线性趋势预测
        n = len(self.history)
        x = np.arange(n)
        y = np.array(self.history)
        slope = np.polyfit(x, y, 1)[0]

        predicted = self.history[-1] + slope
        return max(0.0, min(1.0, predicted))

    def is_degrading(self, threshold=0.05):
        if len(self.history) < 5:
            return False
        recent = list(self.history)[-5:]
        return recent[-1] - recent[0] < -threshold
```

**预期效果**: 提前触发路由切换，减少丢包

### 3.4 改进四：能量感知骨干扩展 (Energy-Aware Skeleton, EAS)

**问题**: v1 骨干节点数量固定，大规模时覆盖不足

**解决方案**: 根据网络规模和能量分布动态扩展骨干

```python
def adaptive_skeleton_count(num_chs, avg_energy_ratio, area_size):
    # 基础骨干数量
    base_k = max(1, int(np.sqrt(num_chs)))

    # 能量感知调节
    if avg_energy_ratio < 0.3:
        k = base_k + 2  # 能量低时增加骨干分担负载
    elif avg_energy_ratio > 0.7:
        k = base_k  # 能量充足时保持基础
    else:
        k = base_k + 1

    # 面积约束
    area_factor = area_size / 10000  # 归一化到100x100
    k = int(k * max(1.0, np.sqrt(area_factor)))

    return min(k, num_chs // 2)
```

**预期效果**: 大规模网络 PDR 提升 3-5%

### 3.5 改进五：协作重传机制 (Cooperative Retransmission, CR)

**问题**: v1 单路径传输，链路失败时直接丢包

**解决方案**: 邻居节点协作重传

```python
class CooperativeRetransmission:
    def __init__(self, max_retries=2):
        self.max_retries = max_retries

    def transmit_with_cooperation(self, sender, receiver, neighbors):
        # 首次尝试直接传输
        if self.try_transmit(sender, receiver):
            return True

        # 失败后尝试通过邻居中继
        for retry in range(self.max_retries):
            # 选择最佳邻居作为中继
            relay = self.select_best_relay(sender, receiver, neighbors)
            if relay and self.try_relay(sender, relay, receiver):
                return True

        return False

    def select_best_relay(self, sender, receiver, neighbors):
        best = None
        best_score = -1
        for n in neighbors:
            if n.id == sender.id or n.id == receiver.id:
                continue
            # 评分 = 到发送者的链路质量 × 到接收者的链路质量
            score = n.lqi_to(sender) * n.lqi_to(receiver)
            if score > best_score:
                best_score = score
                best = n
        return best if best_score > 0.5 else None
```

**预期效果**: 链路不稳定场景 PDR 提升 5-10%

---

## 4. AERIS v2 整体架构

### 4.1 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      Base Station (BS)                       │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│           Adaptive Multi-Gateway Layer (AMG)                 │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐                    │
│    │ GW-1    │  │ GW-2    │  │ GW-k    │  k = f(n, PDR)     │
│    └─────────┘  └─────────┘  └─────────┘                    │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ Cooperative Retransmission
┌─────────────────────────────────────────────────────────────┐
│           Energy-Aware Skeleton Layer (EAS)                  │
│    ┌─────────┐  ┌─────────┐  ┌─────────┐                    │
│    │ SK-1    │  │ SK-2    │  │ SK-m    │  m = f(n, E)       │
│    └─────────┘  └─────────┘  └─────────┘                    │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │ Predictive Link Quality
┌─────────────────────────────────────────────────────────────┐
│           Online CAS Learning Layer (OCL)                    │
│         DIRECT ←→ CHAIN ←→ TWO_HOP                          │
│         (动态权重调整，Bandit反馈)                            │
└─────────────────────────────────────────────────────────────┘
                              ▲
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Sensor Nodes                              │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 组件交互流程

```
每轮执行流程:
1. PLQ 更新链路质量历史，预测下一轮质量
2. OCL 根据上轮反馈调整 CAS 权重
3. CAS 选择簇内传输模式
4. EAS 动态调整骨干节点数量
5. AMG 根据 PDR 反馈调整网关数量
6. CR 在传输失败时触发协作重传
```

---

## 5. 实现计划

### 5.1 阶段划分

| 阶段 | 内容 | 优先级 |
|------|------|--------|
| P0 | AMG 自适应多网关 | 高 |
| P1 | EAS 能量感知骨干 | 高 |
| P2 | CR 协作重传 | 中 |
| P3 | PLQ 预测链路质量 | 中 |
| P4 | OCL 在线CAS学习 | 低 |

### 5.2 预期效果汇总

| 改进 | PDR提升 | 能耗影响 |
|------|---------|----------|
| AMG | +5-8% | +5% |
| EAS | +3-5% | +3% |
| CR | +5-10% | +10% |
| PLQ | +2-3% | 0% |
| OCL | +3-5% | 0% |
| **总计** | **+15-25%** | **+15-20%** |

**v2 目标**: 500节点 PDR 从 79.7% 提升至 90%+

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| 能耗增加过多 | 网络寿命缩短 | 设置能耗上限，动态降级 |
| 计算复杂度 | MCU资源不足 | 使用查表法替代复杂计算 |
| 参数调优困难 | 性能不稳定 | 提供默认参数集 |

---

## 7. 下一步行动

1. **P0 实现**: 先实现 AMG 自适应多网关
2. **验证实验**: 运行 500 节点实验验证效果
3. **迭代优化**: 根据实验结果调整参数
4. **论文更新**: 将 v2 改进纳入论文

---

*文档版本: 2026-01-26 Draft*
*作者: Claude Opus 4.5*


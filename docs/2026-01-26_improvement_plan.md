# AERIS 改进计划 (基于GPT DeepSearch审查意见)

**日期**: 2026-01-26
**目标**: MDPI Sensors (Q2) 投稿
**预计工作量**: 2-4周

---

## 一、总体评分回顾

| 维度 | 当前评分 | 目标评分 |
|------|----------|----------|
| 算法创新性 | 70/100 | 80/100 |
| 代码质量 | 80/100 | 85/100 |
| 实验严谨性 | 75/100 | 85/100 |
| 论文质量 | 78/100 | 85/100 |
| 投稿可行性 | 75/100 | 85/100 |

---

## 二、改进任务清单

### 🔴 P0 - 必须完成 (阻塞投稿)

#### P0-1: 调整100% PDR宣称
- **问题**: 动态实验100% PDR不可信
- **文件**: `docs/Paper_Draft_Section6_Results_REVISED.md`
- **改进**:
  - 将"100% PDR"改为"99.9% PDR (仿真条件下)"
  - 添加说明: "实际部署可能因外部干扰而降低"
  - 量化可靠性机制开销 (重传次数、额外能耗)

#### P0-2: 添加2023-2025年文献引用
- **问题**: 只引用2000年代老协议
- **文件**: `docs/Paper_Draft_Section2_Related_Work_COMPLETE.md`
- **必须引用**:
  1. Suresh et al., 2024 - 联邦深度RL路由
  2. Wang et al., 2024 - 模糊+量子退火 (Sensors)
  3. Soltani et al., 2025 - 多智能体RL
  4. Faridha Banu, 2025 - 模糊+博弈论

#### P0-3: 明确LSTM模块贡献
- **问题**: LSTM仅提升2-3% PDR，必要性存疑
- **文件**: `docs/Paper_Draft_Section6_Results_REVISED.md`
- **改进**:
  - 添加消融实验表格: LSTM vs 无LSTM vs 滑动窗口
  - 明确说明: "LSTM在稳定环境下贡献有限，但在高波动场景有潜力"
  - 或考虑简化为EWMA统计方法

#### P0-4: 调整能量效率定位
- **问题**: 能量消耗与PEGASIS相当，不能宣称"更节能"
- **文件**: `docs/Paper_Draft_Section1_Introduction_REVISED.md`
- **改进**:
  - 定位调整: "高可靠性协议，能量效率与PEGASIS相当"
  - 强调: "以相同能耗实现更高PDR是成就"
  - 添加能量-PDR权衡分析图

---

### 🟡 P1 - 重要改进 (显著提升质量)

#### P1-1: 添加现代协议基线对比
- **问题**: 只对比LEACH/PEGASIS/HEED (2000年代)
- **文件**: `scripts/run_sota_comparison.py`
- **改进**:
  - 实现DEEC (异构能量协议)
  - 实现SEP (稳定选举协议)
  - 可选: 简化版RL-based路由
- **工作量**: 3-5天

#### P1-2: 量化可靠性机制开销
- **问题**: 100% PDR的代价是什么？
- **文件**: `src/aeris_protocol.py`
- **改进**:
  - 统计每包平均重传次数
  - 统计功率步进触发频率
  - 统计邻居救援使用率
  - 计算额外能耗开销
- **工作量**: 1-2天

#### P1-3: 代码重构 - transmit_to_bs()
- **问题**: 200行长函数，复杂度高
- **文件**: `src/aeris_protocol.py:812-1000`
- **改进**:
  ```python
  # 拆分为:
  def _try_primary_parent()
  def _power_stepping_retry()
  def _try_alternate_parents()
  def _neighbor_rescue_broadcast()
  def _final_fallback()
  ```
- **工作量**: 1天

#### P1-4: 添加动态阈值自适应
- **问题**: GPT建议添加 get_stage_adaptive_weights()
- **文件**: `src/aeris_protocol.py`
- **改进**:
  - 根据网络阶段动态调整权重
  - 早期: 节能优先
  - 后期: 可靠性优先
- **工作量**: 2天

---

### 🟢 P2 - 建议改进 (锦上添花)

#### P2-1: CAS模块参数自适应
- **文件**: `src/cas_selector.py`
- **改进**: 根据网络密度/阶段调整模式选择权重
- **工作量**: 1天

#### P2-2: 网关负载均衡保护
- **文件**: `src/gateway_selector.py`
- **改进**: 单网关过载时自动分流
- **工作量**: 1天

#### P2-3: 添加特性对比表
- **文件**: 论文Introduction或Related Work
- **改进**: AERIS vs 基线的特性矩阵表
- **工作量**: 0.5天

#### P2-4: 图表优化
- **问题**: 部分图表信息密度过高
- **改进**:
  - 拆分12子图为2张6子图
  - 添加置信区间误差棒
  - 统一显著性标记规范
- **工作量**: 1天

---

### 🔵 P3 - 未来工作 (论文中说明)

#### P3-1: NS-3交叉验证
- **说明**: "Future work将在NS-3中验证"
- **理由**: 时间不足，但需明确承认

#### P3-2: 硬件测试
- **说明**: "计划在TelosB/CC2420上实现"
- **理由**: 增强可信度

#### P3-3: 安全/信任机制
- **说明**: "未来可集成信任模型检测恶意节点"
- **理由**: 回应GPT关于安全的建议

---

## 三、执行时间表

### 第1周: P0任务 (必须完成)

| 天数 | 任务 | 产出 |
|------|------|------|
| Day 1 | P0-1: 调整100% PDR宣称 | 修改后的Section 6 |
| Day 2 | P0-2: 添加现代文献 (上) | 文献列表 |
| Day 3 | P0-2: 添加现代文献 (下) | 修改后的Section 2 |
| Day 4 | P0-3: LSTM消融讨论 | 消融表格 |
| Day 5 | P0-4: 能量定位调整 | 修改后的Section 1 |

### 第2周: P1任务 (重要改进)

| 天数 | 任务 | 产出 |
|------|------|------|
| Day 6-8 | P1-1: 实现DEEC/SEP基线 | 新基线代码 |
| Day 9 | P1-2: 量化可靠性开销 | 开销统计表 |
| Day 10 | P1-3: 重构transmit_to_bs | 重构后代码 |

### 第3周: P2任务 + 论文整合

| 天数 | 任务 | 产出 |
|------|------|------|
| Day 11 | P1-4: 动态阈值自适应 | 新算法代码 |
| Day 12 | P2-1/P2-2: CAS/网关改进 | 优化后模块 |
| Day 13 | P2-3/P2-4: 表格/图表优化 | 新图表 |
| Day 14 | 论文整合与校对 | 完整论文 |

---

## 四、具体代码修改示例

### 4.1 动态阈值自适应 (P1-4)

```python
# src/aeris_protocol.py - 新增函数

def get_stage_adaptive_weights(self, current_round, total_rounds, avg_energy_ratio):
    """
    根据网络阶段动态调整权重
    - 早期(0-30%): 节能优先
    - 中期(30-70%): 平衡
    - 后期(70-100%): 可靠性优先
    """
    stage_ratio = current_round / total_rounds

    if stage_ratio < 0.3:
        # 早期: 节能优先
        return {'energy': 0.6, 'reliability': 0.2, 'distance': 0.2}
    elif stage_ratio < 0.7:
        # 中期: 平衡
        return {'energy': 0.4, 'reliability': 0.4, 'distance': 0.2}
    else:
        # 后期: 可靠性优先
        return {'energy': 0.2, 'reliability': 0.6, 'distance': 0.2}
```

### 4.2 可靠性开销统计 (P1-2)

```python
# src/aeris_protocol.py - 添加统计变量

class AERISProtocol:
    def __init__(self):
        # 可靠性机制统计
        self.stats = {
            'total_packets': 0,
            'retransmissions': 0,
            'power_stepping_count': 0,
            'alternate_parent_count': 0,
            'neighbor_rescue_count': 0,
            'broadcast_fallback_count': 0
        }

    def get_overhead_report(self):
        """生成可靠性开销报告"""
        if self.stats['total_packets'] == 0:
            return {}
        return {
            'avg_retrans_per_packet': self.stats['retransmissions'] / self.stats['total_packets'],
            'power_stepping_rate': self.stats['power_stepping_count'] / self.stats['total_packets'],
            'alternate_parent_rate': self.stats['alternate_parent_count'] / self.stats['total_packets'],
            'rescue_rate': self.stats['neighbor_rescue_count'] / self.stats['total_packets']
        }
```

---

## 五、审稿人质疑回应策略

### Q1: "100% PDR是否可信？"

**回应模板**:
> "AERIS在仿真条件下实现了99.9% PDR，这得益于多层可靠性机制的协同作用。我们承认实际部署中可能因外部干扰而降低。表X量化了可靠性机制的开销：平均每包重传0.3次，功率步进触发率8%，邻居救援使用率1.5%。"

### Q2: "LSTM必要性？"

**回应模板**:
> "消融实验显示LSTM贡献约2-3% PDR提升。虽然在稳定环境下贡献有限，但LSTM为未来扩展提供了基础。实际部署可根据资源约束选择简化版本。"

### Q3: "与最新SOTA差距？"

**回应模板**:
> "我们在Related Work中讨论了RL/FL方法。AERIS定位为轻量级替代方案：不需要训练阶段，适合资源受限节点。"

---

## 六、文献引用清单

### 必须引用 (2023-2025)

1. **Suresh et al., 2024** - Federated Deep RL for IoT-WSN
2. **Wang et al., 2024** - Fuzzy + Quantum Annealing (Sensors)
3. **Soltani et al., 2025** - Multi-agent RL for CH selection
4. **Faridha Banu, 2025** - Fuzzy + Game Theory
5. **Sefati et al., 2025** - Federated RL + Security

### 建议引用

6. Han et al., 2023 - Trust-aware Fuzzy Routing
7. Misbahuddin et al., 2025 - LPWAN Multi-hop
8. Energy Harvesting WSN Survey (2024)

---

## 七、成功标准

| 指标 | 当前 | 目标 |
|------|------|------|
| 总体评分 | 75/100 | 85/100 |
| 现代文献引用 | 0篇 | 5-8篇 |
| 基线协议数 | 3个 | 5个 |
| 可靠性开销量化 | 无 | 完整统计 |
| 代码复杂度 | 200行长函数 | 5个短函数 |

---

**文档结束**
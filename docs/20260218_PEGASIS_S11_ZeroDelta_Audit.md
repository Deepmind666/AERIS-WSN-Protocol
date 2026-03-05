# PEGASIS 在 S11 indoor_factory 中 delta=0 的代码审计（2026-02-18）

## 结论

S11 中 PEGASIS 在 `indoor_factory` 下 6 个节点规模全部 `delta=0`，**更可能是当前补丁配置的设计结果**，不是随机统计噪声。

核心原因是三点同时成立：
1. PEGASIS 链路碰撞默认豁免（`pegasis_chain_exempt=True`）。
2. PEGASIS 每轮只有一个 leader 上行，`uplink_factor(1)=1.0`。
3. PEGASIS 路径没有接入 baseline multihop relay 机制。

因此在 patch/control 对比中，PEGASIS 的有效传输概率在代码路径上可能保持完全一致，出现 `t=0, p=1.0` 是可解释的。

---

## 证据锚点

### 1) patch 模式的碰撞模型创建
- 文件：`scripts/run_scalability_experiment.py`
- 位置：`run_protocol()` 内
- 逻辑：当 `--mac-collision` 打开时，调用
  - `MACCollisionModel(MACCollisionConfig(enabled=True))`
- 关键点：未显式覆盖 `pegasis_chain_exempt`，因此使用默认值。

### 2) PEGASIS 链路碰撞豁免默认开启
- 文件：`src/mac_collision_model.py`
- `MACCollisionConfig` 默认：
  - `pegasis_chain_exempt: bool = True`
- `compute_chain_factor()`：
  - 当 `pegasis_chain_exempt=True` 时直接返回 `1.0`。

### 3) PEGASIS 传输阶段实际使用的碰撞因子
- 文件：`src/baseline_protocols/pegasis_protocol.py`
- `data_transmission_phase()`：
  - `chain_factor = mac.compute_chain_factor(len(self.chain))`
  - `uplink_factor = mac.compute_uplink_factor(1)`
- 关键点：
  - `compute_uplink_factor(1)` 在模型中返回 `1.0`（无并发上行冲突）。
  - leader->BS 仅此一路径，因此 patch 对 PEGASIS 的“碰撞惩罚”可为零。

### 4) S11 数据侧验证
- 文件：`results/mega_experiments/s11_matched_4env_patch_vs_control_20260217_delta.csv`
- 现象：`environment=indoor_factory, protocol=PEGASIS` 的 6 个 `num_nodes` 的 `delta` 全为 `0.0`。
- 文件：`results/mega_experiments/s11_matched_4env_patch_vs_control_20260217_significance.csv`
- 现象：对应 6 行 `t_stat=0.0, p_raw=1.0, p_holm=1.0, hedges_g=0.0`。

---

## 对论文写作的影响

1. 不应把该结果写成“PEGASIS 对现实碰撞天然鲁棒”。  
2. 应写成“在当前补丁实现中，PEGASIS 的碰撞路径被配置为豁免/弱影响，因此该组结果主要反映实现边界”。  
3. 应在 Limitations 中明确该实现边界，避免审稿人理解为通用物理结论。

---

## 建议（后续实验，不阻塞当前 v31）

若要验证“真实碰撞下 PEGASIS 也受影响”，建议新增一个小矩阵：
- 仅 PEGASIS，`indoor_factory`
- 节点：`100, 500, 1000`
- `n=200`/cell
- patch 分两组：
  - A 组：`pegasis_chain_exempt=True`（当前）
  - B 组：`pegasis_chain_exempt=False`（强制链路碰撞惩罚）

如果 B 组出现显著负 delta，就能把当前异常从“结果争议”转为“实现设定可解释差异”。

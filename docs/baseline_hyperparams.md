# 基线参数审计（已按代码默认值核验）

> 目的：确保“对比公平”。当前仅记录**实现默认值**，未进行参数搜索。
> 若后续进行调参，请补全搜索范围/预算并同步更新 `configs/baseline_params.yaml`。

## 统一公共设定（从 `configs/phy_energy.yaml` 读取）
- 信道模型：Log-Normal Shadowing（INDOOR_OFFICE）
- 能耗模型：ImprovedEnergyModel / CC2420-TELOSB
- 包长/速率：默认 4000 bits（TEEN 为 1024 bytes），250 kbps
- 随机种子策略：seed_list（由脚本设置）

---

## 协议级参数表（默认实现）

### LEACH
- 关键参数：p_ch
- 默认值：0.10
- 搜索范围：未搜索
- 搜索策略：未搜索
- 预算：未搜索
- 来源：`src/baseline_protocols/leach_protocol.py`

### HEED
- 关键参数：c_prob、cluster_radius
- 默认值：c_prob=0.05，cluster_radius=50 m
- 搜索范围：未搜索
- 搜索策略：未搜索
- 预算：未搜索
- 来源：`src/baseline_protocols/heed_protocol.py`

### PEGASIS
- 关键参数：无显式超参（链式路由）
- 默认值：packet=4000 bits，E0=2.0 J
- 搜索范围：未搜索
- 搜索策略：未搜索
- 预算：未搜索
- 来源：`src/baseline_protocols/pegasis_protocol.py`

### TEEN
- 关键参数：hard_threshold、soft_threshold、p_ch、max_time_interval
- 默认值：HT=45，ST=0.5，p_ch=0.08，max_time_interval=3
- 搜索范围：未搜索
- 搜索策略：未搜索
- 预算：未搜索
- 来源：`src/teen_protocol.py`

### SEP
- 关键参数：p_opt、m、alpha
- 默认值：p_opt=0.10，m=0.10，alpha=1.0
- 搜索范围：未搜索
- 搜索策略：未搜索
- 预算：未搜索
- 来源：`src/baseline_protocols/sep_protocol.py`

### AERIS‑E / AERIS‑R
- 关键参数：profile、CAS/Skeleton/Gateway/Fairness 开关
- 默认值：profile=energy/robust；enable_cas/enable_skeleton/enable_gateway/enable_fairness=True
- 搜索范围：未搜索
- 搜索策略：未搜索
- 预算：未搜索
- 来源：`src/aeris_protocol.py`

---

## 审计规则
1. **禁止只调 AERIS**：若进行调参，基线与 AERIS 使用相同调参预算。
2. 若基线无法复现：必须记录原因与替代方案。
3. 所有最终参数必须写入 `configs/baseline_params.yaml`。

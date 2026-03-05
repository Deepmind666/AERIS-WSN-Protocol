# 基线对比矩阵（已按代码默认参数核验）

> 目的：让审稿人清晰看到“为什么这些基线被选中”以及能力维度是否对齐。
> 下表依据**代码默认实现**填写，若后续调整参数请同步更新。

**字段说明**
- 环境感知：是否显式使用温湿度/链路质量作为路由依据
- 可靠性机制：是否包含重传/多路径/协作/容错
- 聚类策略：是否采用聚类/链式/多跳/分层
- 控制开销：相对量级（高/中/低）
- 计算复杂度：常数/线性/启发式/优化/学习
- 来源：文献/开源实现/内部实现（必须可追溯）

| 协议 | 环境感知 | 可靠性机制 | 聚类策略 | 控制开销 | 计算复杂度 | 典型参数 | 来源 | 备注 |
|---|---|---|---|---|---|---|---|---|
| LEACH | 否（无温湿度/LQI输入） | 无显式重传/多路径 | 概率 CH 轮换 | 中 | 线性 | p_ch=0.1, packet=4000 bits, E0=2.0 J | `src/baseline_protocols/leach_protocol.py` | 仅仿真版 |
| HEED | 否 | 无显式重传/多路径 | 残能+邻居度迭代选 CH | 中 | 线性/启发式 | c_prob=0.05, cluster_radius=50 m, packet=4000 bits | `src/baseline_protocols/heed_protocol.py` | 仅仿真版 |
| PEGASIS | 否 | 无显式重传/多路径 | 链式汇聚（单链） | 低 | 线性 | packet=4000 bits, E0=2.0 J | `src/baseline_protocols/pegasis_protocol.py` | 仅仿真版 |
| TEEN | 否（事件阈值，不含环境模型） | 无显式重传/多路径 | 阈值触发分层聚类 | 中 | 线性 | HT=45, ST=0.5, p_ch=0.08, packet=1024 B | `src/teen_protocol.py` | 仅仿真版 |
| SEP | 否 | 无显式重传/多路径 | 异构能量加权 CH | 中 | 线性 | p_opt=0.1, m=0.1, alpha=1.0, packet=4000 bits | `src/baseline_protocols/sep_protocol.py` | 仅仿真版 |
| AERIS‑E | 是（温湿度+真实信道/LQI） | 网关+安全钩子+CAS | CAS+骨架+网关 | 中 | 线性/启发式 | profile=energy, CAS/Skeleton/Gateway 开启 | `src/aeris_protocol.py` | AERIS 能源档 |
| AERIS‑R | 是（温湿度+真实信道/LQI） | 网关+安全钩子+CAS | CAS+骨架+网关 | 中 | 线性/启发式 | profile=robust, CAS/Skeleton/Gateway 开启 | `src/aeris_protocol.py` | AERIS 可靠档 |

**填表规则**
1. 所有条目必须可追溯（论文页码/开源实现 commit/内部代码路径）。
2. 如果某协议无法公平复现，必须在“备注”说明原因与替代基线。
3. 任何“优化/调参”要在 `docs/baseline_hyperparams.md` 中给出预算与搜索策略。

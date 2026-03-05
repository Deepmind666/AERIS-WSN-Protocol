# AERIS：面向真实环境的自适应物联网传感器路由协议

说明语言：简体中文（UTF-8）。本项目统一品牌为 “AERIS”。历史代码路径（如 `src/Enhanced-EEHFR-WSN-Protocol/`）为兼容旧脚本暂时保留，详情见 `docs/Legacy_Path_Mapping.md`。

- 项目主页：`https://github.com/Deepmind666/AERIS-WSN-Protocol`
- 许可协议：MIT（见 `LICENSE`）
- 评估数据集：Intel Berkeley Research Lab（2.22M 记录，54 节点）

重要说明（算法研究、非硬件实现）：
- 本仓库仅包含算法与仿真代码，用于学术研究与方法验证；不涉及任何实际硬件控制、驱动或设备操作。
- 与 IEEE 802.15.4 的一致性仅体现在信道/链路质量的保守建模与参数化（RSSI/LQI/PDR 等），不包含完整 MAC 行为（如 CSMA/CA、ACK/重传时序）。
- 术语统一：EEHFR 命名已废弃，统一更名为 AERIS；遗留路径/脚本仅为兼容性保留（参见 `docs/Legacy_Path_Mapping.md`）。

---

## 摘要

AERIS（Adaptive Environment‑aware Routing for IoT Sensors）聚焦无线传感器网络在真实部署中的两大核心问题：能耗与可靠性。大量研究在理想化仿真中取得高性能，但在真实环境（湿度、温度、干扰、非均匀拓扑）下往往出现仿真‑现实鸿沟。AERIS 以 IEEE 802.15.4 一致的保守信道建模和真实数据驱动的环境映射为基础，通过“环境上下文选择（CAS）+ 骨干路由（Skeleton）+ 网关协作（Gateway Coordination）”的分层架构，实现无需深度学习推理的轻量自适应。核心价值：

- 面向真实环境的可部署性：不依赖重训练与高算力；适配资源受限节点。
- 能耗‑可靠性平衡：在较低能耗下提高端到端投递率（PDR）。
- 可重复性与严谨统计：提供脚本化实验、JSON 指标与图表，使用 Welch t、Bootstrap 置信区间、Holm‑Bonferroni 多重检验。

主要功能：
- 协议实现与基线对比（AERIS、LEACH、HEED、PEGASIS、TEEN）。
- 真实数据驱动的环境映射与保守功率控制。
- 统计验证、图表生成与导出（面向论文与报告）。
- 模型导出与推理基准（ONNX Runtime，CPU/DML/NPU）。

---

## 功能特性

| 模块 | 说明 |
|---|---|
| 环境上下文选择（CAS） | 基于环境与几何特征选择直传/链式/两跳模式，降低长距离传输失败率 |
| 骨干路由（Skeleton） | 通过高能量、连通性较好的骨干节点形成可靠路径 |
| 网关协作（Gateway） | 两跳协作支持远簇头，提高端到端 PDR |
| 保守功率控制 | 按环境类型调整发射功率，避免不必要的高功率开销 |
| 真实信道建模 | 基于 IEEE 802.15.4 与对湿度/温度的保守映射，贴近真实传播 |
| 基线协议实现 | LEACH/HEED/PEGASIS/TEEN 的标准化实现与统一指标对齐 |
| 统计显著性 | Welch t、Bootstrap CI、Holm‑Bonferroni 多重检验脚本 |
| 图表生成 | 论文级 SVG/PNG 图表生成与整理脚本（可直接用于投稿） |
| ONNX 导出与基准 | 导出 LSTM/TCN 等模型并进行 CPU/ORT 推理基准测试 |

---

## 系统架构（示意）

```mermaid
flowchart LR
  D[数据资产\nIntel Lab 原始记录\n节点位置] --> F[特征提取]
  F --> C[环境分类/上下文选择 (CAS)]
  C --> R1[骨干路由 (Skeleton)]
  R1 --> G[网关协作 (两跳)]
  G --> M[MAC/PHY 仿真\nIEEE 802.15.4]
  M --> O[结果与指标\nJSON/CSV]
  O --> S[统计验证\nWelch t / Bootstrap / Holm-Bonferroni]
  O --> P[图表生成\n论文级 SVG/PNG]

  subgraph Baselines
    B1[LEACH]:::b
    B2[HEED]:::b
    B3[PEGASIS]:::b
    B4[TEEN]:::b
  end
  classDef b fill:#EEE,stroke:#333,stroke-width:1px

  D --> B1
  D --> B2
  D --> B3
  D --> B4

  subgraph Export
    E1[ONNX 导出]:::b
    E2[推理基准 (ORT)]:::b
  end
  O --> E1
  E1 --> E2
```

---

## 安装与使用（分步骤）

### 1. 环境准备
- Python ≥ 3.8（建议使用 Conda）
- 可选：`scripts/conda_env.yml` 一键创建环境：
  - Windows PowerShell：
    - `conda env create -f scripts/conda_env.yml`
    - `conda activate aeris-py311`（或 yml 中的具体名称）
- 或使用 `requirements.txt`：`pip install -r requirements.txt`

### 2. 获取数据与元数据
- 运行：`python scripts/download_intel_assets.py`
- 或手动将资产置于 `data/Intel_Lab_Data/`，并确保存在 `mote_locs.txt`。

### 3. 运行核心实验与基线
- 统一入口：`python scripts/run_experiments.py`
- 典型基线：`python scripts/run_intel_baselines_all.py`
- 敏感性与消融：
  - `python scripts/run_intel_sensitivity.py`
  - `python scripts/run_intel_ablation.py`
- 大规模/走廊拓扑：
  - `python scripts/run_compare_multi_topo.py`
- 动态场景：
  - `python scripts/run_dynamic_corridor_compare.py`
  - `python scripts/run_dynamic_moving_bs_compare.py`
  - `python scripts/run_dynamic_dropout_compare.py`
  - `python scripts/run_large_scale_long.py`
  - `python scripts/summarize_dynamic_stats.py`
  - `python scripts/compute_monte_carlo_stats.py`
- 长时/蒙特卡洛：
  - `python scripts/run_large_scale_long.py`
  - `python scripts/run_monte_carlo_uniform.py`
- 更多统计与图表说明可见 `docs/Supplementary_Results.md`。
  - *提示：`docs/Supplementary_Results.md` 第 7 节列出了所有主要脚本的“命令 / 种子 / 输出 / 图表”对照表，便于审稿人与合作者复现每一个图。*

### 4. 统计验证与图表生成
- 统计显著性：
  - `python scripts/run_significance_intel.py`
  - `python scripts/run_stats_multitest.py`
  - `python scripts/compute_dynamic_significance.py`
  - `python scripts/compute_monte_carlo_stats.py`
  - `python scripts/compute_aeris_round_significance.py`
- 论文图表：
  - `python scripts/plot_paper_figures.py`
  - `python scripts/curate_figures.py`
  - `python scripts/plot_gateway_limit_effect.py --limits 1,2,3,4 --dataset 'Uniform-300=results/gateway_sweep_uniform300_dualbs_limit{}.json' --dataset 'Uniform-500=results/gateway_sweep_uniform500_dualbs_limit{}.json'`
- 结果位置：`results/benchmark_experiments/`、`results/publication_figures/`、`results/*.json`

> 提示：`scripts/run_gateway_sweep.py` 现支持 `--extra-bases`（多基站）、`--skeleton-*`（骨干参数）以及 `--gateway-limit`（单 gateway 服务上限）等可选项，便于复现双基站与限额实验。

### 5. 模型导出与推理基准
- 导出与基准：`python scripts/export_onnx_and_bench.py`
- 支持：CPU / ONNX Runtime（可选 DML/NPU）
- 产物：`results/*.onnx` 与 `results/inference_bench.json`

### 6. 常用命令示例（Windows）
- 查看环境信息：`python scripts/print_pyinfo.py`
- 连续推理演示（ORT DML）：
  - `python scripts/continuous_infer.py --engine ort_dml --model tcn --seq-len 3072`

---

## 仓库结构（简版）

```
src/                      # 协议与模型实现（AERIS 与基线）
scripts/                  # 实验、统计、导出与图表脚本
data/                     # 数据与元数据（含节点位置）
results/                  # 指标、图表与导出模型产物
docs/                     # 研究记录与论文草稿
```

---

## 贡献与约定
- 提交前请确保脚本在本地可复现（固定随机种子与配置）。
- 图表与指标由脚本生成；README 不直接嵌入图片数据，只保留指向产物的路径。

## 可复现实验对照表
- 仓库根目录下的 `docs/Reproduction_Table.md` 汇总了“场景→脚本命令→种子→产物→对应图表”的一览表，可直接点击查看或拷贝命令运行。
- 该表由 `docs/reproduction_manifest.json` 驱动，使用 `python scripts/generate_reproduction_table.py` 可一键更新，确保论文、README 与实际脚本保持一致。

---

## 引用（BibTeX）

```bibtex
@misc{aeris_wsns_2025,
  title  = {AERIS: Adaptive Environment-aware Routing for IoT Sensors},
  author = {Deepmind666},
  year   = {2025},
  url    = {https://github.com/Deepmind666/AERIS-WSN-Protocol}
}
```

---

## 联系方式
- GitHub：`@Deepmind666`
- Email：`1403073295@qq.com`

---

## 许可协议
本项目采用 MIT 许可。允许研究与商业使用；分发时须保留许可与版权说明。
# Distilled CAS Usage（AERIS 兼容模式）

Enable distilled CAS via environment variable or constructor flag:

```bash
# 环境变量启用
set USE_DISTILLED_CAS=1
python scripts/smoke_test.py

# 或构造参数启用
python - <<'PY'
import os, sys
sys.path.append('src')
from benchmark_protocols import NetworkConfig
from aeris_protocol import AerisProtocol
cfg = NetworkConfig(num_nodes=25, area_width=100, area_height=100, initial_energy=2.0, packet_size=1024)
proto = AerisProtocol(cfg, enable_cas=True, enable_fairness=True, use_distilled_cas=True)
res = proto.run_simulation(100)
print(res['packet_delivery_ratio_end2end'], res['total_energy_consumed'])
PY
```

Optional weight file: place `data/distilled_cas_weights.npz` with arrays `W1,b1,W2,b2` (`int32`).

## Evaluate Distilled vs Rule-based

Run comparative evaluation and save summary:

```bash
python scripts/run_distilled_cas_eval.py --nodes 50 --rounds 200 --seeds 5 --output results/distilled_eval.json
```

The JSON contains per-run results and a summary of PDR (hop & end-to-end), energy, lifetime, and inference time stats.

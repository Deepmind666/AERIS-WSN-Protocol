# AERIS：面向真实环境的自适应物联网传感器路由协议

说明语言：简体中文（纯 UTF-8，无表情符号与特殊标记）

项目品牌统一为“AERIS”。代码中的历史路径（如 `src/Enhanced-EEHFR-WSN-Protocol/`）暂时保留以避免破坏脚本，后续将逐步迁移到 `src/aeris/`。

- 项目主页：`https://github.com/Deepmind666/AERIS-WSN-Protocol`
- 许可协议：MIT（见 `LICENSE`）
- 评估数据集：Intel Berkeley Research Lab（2.22M 记录，54 节点）

## 摘要
AERIS 是一个面向真实环境的轻量级自适应路由协议。在与 IEEE 802.15.4 一致的信道建模与真实数据驱动的环境映射下，AERIS 采用“环境上下文选择 + 骨干路由 + 网关协作”的分层架构，在不依赖深度学习推理的前提下实现对环境与几何变化的自适应，并以严格的统计方法（Welch t、Bootstrap 置信区间）保证结果的可重复性。

## 方法概览
- 环境感知：从传感器与链路指标提取特征，进行环境类别识别与保守功率控制。
- 结构化路由：骨干路径与两跳协作提高长距离簇头的端到端投递率（PDR）。
- 轻量自适应：使用简化的在线权重更新与离散决策表，适合资源受限节点。

## 快速开始
- 安装依赖：`pip install -r requirements.txt`
- 准备数据：`python scripts/download_intel_assets.py`（或将资产放入 `data/Intel_Lab_Data/`）
- 运行核心实验：`python scripts/run_experiments.py`
- 导出与基准：`python scripts/export_onnx_and_bench.py`
- 生成论文图表：`python scripts/plot_paper_figures.py` 与 `python scripts/curate_figures.py`
- 结果目录：`results/benchmark_experiments/` 与 `results/publication_figures/`

## 仓库结构
- `src/`：协议与模型（AERIS，LEACH/HEED/PEGASIS/TEEN，预测环境模块 等）
- `scripts/`：实验与导出脚本（统一入口 `run_experiments.py`）
- `data/`：数据与元数据（含 Intel Lab 节点位置信息）
- `results/`：实验结果与论文级图表（README 不展示图片细节）
- `docs/`：研究记录与论文草稿

## 可重复性
- 固定随机种子，脚本输出 JSON 指标与 SVG 图表。
- 统计显著性：`scripts/run_significance_intel.py` 与 `scripts/run_stats_multitest.py`。
- 导出与推理基准：`scripts/export_onnx_and_bench.py`（支持 CPU/ONNXRuntime）。

## 许可协议
本项目采用 MIT 许可，允许研究与商业使用，需保留许可与版权说明。

## 引用（BibTeX）
```bibtex
@misc{aeris_wsns_2025,
  title  = {AERIS: Adaptive Environment-aware Routing for IoT Sensors},
  author = {Deepmind666},
  year   = {2025},
  url    = {https://github.com/Deepmind666/AERIS-WSN-Protocol}
}
```

## 联系方式
- GitHub：`@Deepmind666`
- Email：`1403073295@qq.com`

注：README 保持论文风格与最小必要信息；图片内容与细节不在此处展开，图表请参考 `results/publication_figures/` 与生成脚本的产物。
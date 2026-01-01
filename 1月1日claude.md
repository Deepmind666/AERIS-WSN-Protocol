# AERIS项目深度评估报告
**评估日期**: 2026-01-01
**评估者**: Claude (Opus 4.5)
**项目版本**: AERIS 2.0 (Integrated)

---

## 一、项目概述

### 1.1 项目定位
AERIS (Adaptive Environment-aware Routing for IoT Sensors) 是一个面向真实环境的自适应物联网传感器路由协议研究项目。该项目旨在解决无线传感器网络在真实部署中的两大核心问题：**能耗**与**可靠性**。

### 1.2 核心创新点
1. **环境上下文选择 (CAS)** - 基于环境与几何特征选择直传/链式/两跳模式
2. **骨干路由 (Skeleton)** - 通过高能量、连通性较好的骨干节点形成可靠路径
3. **网关协作 (Gateway)** - 两跳协作支持远簇头，提高端到端PDR
4. **安全阈值机制 (Safety)** - 自适应安全切换机制

### 1.3 目标期刊
**MDPI Sensors** - 传感器网络/无线传感器网络专栏

### 1.4 作者信息 (来自初稿)
- **第一作者**: Kangrui Li (李康锐)
  - 邮箱: 1403073295@mails.gdut.edu.cn
- **通讯作者**: Xiaobo Zhang (张晓波)
  - 邮箱: zxb_leng@gdut.edu.cn
- **合作者**: Junyi Lin (林俊毅)
  - 邮箱: 3123001378@mail2.gdut.edu.cn
- **单位**: Faculty of Automation, Guangdong University of Technology, Guangzhou 510006, China

---

## 二、项目结构分析

### 2.1 代码架构
```
AERIS-WSN-Protocol/
├── src/                          # 核心源代码 (~40个Python文件)
│   ├── aeris_protocol.py         # 主协议实现 (1465行, 核心)
│   ├── cas_selector.py           # CAS选择器
│   ├── gateway_selector.py       # 网关选择器
│   ├── skeleton_selector.py      # 骨干选择器
│   ├── improved_energy_model.py  # CC2420能量模型
│   ├── realistic_channel_model.py # 真实信道建模
│   ├── baseline_protocols/       # 基线协议
│   │   ├── leach_protocol.py
│   │   ├── heed_protocol.py
│   │   ├── pegasis_protocol.py
│   │   └── teen_protocol.py
│   └── ...
├── scripts/                      # 实验脚本 (~80个)
├── results/                      # 实验结果 (~100+ JSON文件)
├── docs/                         # 文档 (~100+ 文档文件)
├── for_submission/               # 投稿材料
├── data/                         # 数据集
│   └── Intel_Lab_Data/           # Intel Lab原始数据
└── tests/                        # 测试文件
```

### 2.2 核心组件评估

| 组件 | 状态 | 代码质量 | 完整性 |
|------|------|----------|--------|
| aeris_protocol.py | ✅ 完整 | 中等 | 1465行，功能完整但有部分注释乱码 |
| CAS选择器 | ✅ 完整 | 良好 | 支持蒸馏版本 |
| Gateway选择器 | ✅ 完整 | 良好 | 支持多网关、负载均衡 |
| Skeleton选择器 | ✅ 完整 | 良好 | 主轴近似算法 |
| 能量模型 | ✅ 完整 | 优秀 | CC2420参数化 |
| 信道模型 | ✅ 完整 | 优秀 | 对数正态阴影衰落 |
| LEACH基线 | ✅ 完整 | 良好 | 标准实现 |
| HEED基线 | ✅ 完整 | 良好 | 标准实现 |
| PEGASIS基线 | ✅ 完整 | 良好 | 标准实现 |
| TEEN基线 | ✅ 完整 | 良好 | 标准实现 |

---

## 三、实验数据验证

### 3.1 数据集
- **Intel Lab数据集**: 2.22M记录，54节点（公开数据集）
- **合成拓扑**: Uniform, Corridor, Dense clusters

### 3.2 关键实验结果文件

| 实验类型 | 文件 | 数据点数 | 状态 |
|----------|------|----------|------|
| 消融实验 | intel_ablation.json | 250点 (50×5) | ✅ 已验证 |
| 敏感性分析 | intel_sensitivity.json | 360点 (40×9) | ✅ 已验证 |
| 基线对比 | intel_baselines_all.json | 多组数据 | ✅ 已验证 |
| 蒙特卡洛 | monte_carlo_uniform50.json | 100×200轮 | ✅ 已验证 |
| 动态场景 | dynamic_corridor_compare.json | 50×4阶段 | ✅ 已验证 |
| 大规模 | large_scale_long.json | 300/500节点×1000轮 | ✅ 已验证 |

### 3.3 核心性能数值

**效应量统计（消融实验, n=50）**:
| 组件 | Hedges' g | 效应大小 | PDR变化 |
|------|-----------|----------|---------|
| Gateway | 4.48 | Large | +24.4% |
| Safety | 3.48 | Large | +29.4% |
| Fairness | -0.10 | Negligible | -0.5% |
| CAS | -0.15 | Negligible | -0.8% |

**关键性能指标**:
- Intel Replay (robust profile): PDR 0.389 → 0.524 (+34.6%)
- 50×100 Monte Carlo: PDR 0.817, Energy 36.8J
- 大规模 (300/500节点, reliability overlays): PDR ≈ 1.0

---

## 四、论文初稿评估

### 4.1 已有论文版本

1. **for_submission/final_paper.pdf** (12月9日版本, 16页)
   - 结构完整，符合MDPI格式
   - 包含11个图表
   - 14篇参考文献 (偏少)

2. **for_submission/aeris_paper_final.tex** (内部草稿)
   - 更简化的版本，7个图表
   - 5篇参考文献 (严重不足)

### 4.2 论文结构分析

**当前结构** (final_paper.pdf):
1. Introduction ✅
2. Related Work ✅ (但需扩展)
3. System Model and Problem Definition ✅
4. AERIS Algorithm ✅
5. Experimental Setup ✅
6. Results and Analysis ✅
7. Conclusions ✅

**问题诊断**:

| 问题类型 | 严重程度 | 描述 |
|----------|----------|------|
| 参考文献不足 | 🔴 高 | 仅14篇，MDPI要求45-60篇 |
| Related Work薄弱 | 🔴 高 | 需要2023-2025最新文献 |
| TODO标记未清理 | 🟡 中 | 多处"TODO: verify bibliographic details" |
| 图表引用不一致 | 🟡 中 | 部分图表编号与内容不匹配 |
| 作者信息占位 | 🟡 中 | 需替换为正式信息 |

### 4.3 图表质量评估

**已生成图表** (results/plots/ 目录):
- 77个PDF图表
- 大部分为论文级质量
- 支持SVG矢量格式

**关键图表**:
| 图表 | 文件 | 状态 | 质量 |
|------|------|------|------|
| 方法流程图 | paper_method_flowchart.pdf | ✅ | 良好 |
| Intel基线对比 | paper_intel_baselines_panels.pdf | ✅ | 优秀 |
| 消融实验 | paper_intel_ablation_pdr.pdf | ✅ | 优秀 |
| 多拓扑显著性 | paper_multi_topo_sig_pdr.pdf | ✅ | 优秀 |
| 动态场景 | paper_dynamic_corridor_compare.pdf | ✅ | 良好 |
| 大规模PDR分解 | paper_pdr_breakdown_large_scale.pdf | ✅ | 良好 |

---

## 五、核心问题与改进计划

### 5.1 高优先级问题 (P0)

#### 问题1: 参考文献严重不足
- **现状**: 14篇 (final_paper.pdf)
- **要求**: 45-60篇
- **解决方案**:
  - 添加2023-2025年Sensors/IEEE IoT Journal/Ad Hoc Networks文献
  - 补充经典WSN协议文献 (LEACH, HEED, PEGASIS, TEEN原文)
  - 添加统计方法文献 (Bootstrap, Welch t-test, Effect size)

#### 问题2: Related Work章节薄弱
- **现状**: ~20行，仅泛泛提及
- **要求**: 2-3页，系统性综述
- **解决方案**:
  - 分类综述: 经典聚类、环境感知、可靠性、元启发式
  - 明确AERIS的定位与差异化

#### 问题3: TODO标记未清理
- **现状**: 多处"TODO: verify bibliographic details"
- **解决方案**: 逐一验证并补充完整引用信息

### 5.2 中优先级问题 (P1)

#### 问题4: 代码注释乱码
- **位置**: aeris_protocol.py等
- **原因**: UTF-8编码问题
- **影响**: 不影响运行，但影响代码可读性

#### 问题5: 统计报告一致性
- **问题**: 不同文档中数值存在细微差异
- **解决方案**: 统一使用最新实验结果

#### 问题6: 实验复现文档
- **现状**: 有reproduction_manifest.json，但README说明不够详细
- **解决方案**: 增强Supplementary Materials

### 5.3 低优先级问题 (P2)

- 部分冗余的旧代码文件
- 图表命名规范不统一
- 文档中的日期不一致

---

## 六、Sensors期刊投稿差距分析

### 6.1 MDPI Sensors要求 vs 当前状态

| 要求 | 当前状态 | 差距 |
|------|----------|------|
| Abstract 200-250词 | ~170词 | 需扩展 |
| 参考文献 45-60篇 | 14篇 | 🔴 严重不足 |
| 数据可用性声明 | ✅ 有 | OK |
| 代码可用性声明 | ✅ 有 | OK |
| 利益冲突声明 | ✅ 有 | OK |
| 作者贡献声明 | ✅ 有 | OK |
| 图表质量 | 良好 | OK |
| 统计严谨性 | 优秀 | OK |

### 6.2 投稿前必须完成的工作

1. **扩展参考文献至50+篇** (~2天工作量)
2. **重写Related Work章节** (~1天工作量)
3. **清理所有TODO标记** (~0.5天工作量)
4. **统一数值报告** (~0.5天工作量)
5. **完善Abstract** (~0.5天工作量)
6. **最终PDF编译与校对** (~1天工作量)

---

## 七、行动计划

### 第一阶段: 文献补充 (优先级最高)

**任务清单**:
- [ ] 收集2023-2025年WSN路由相关文献30篇
- [ ] 添加经典协议原始论文引用
- [ ] 补充统计方法引用
- [ ] 更新bibliography.bib

**目标**: 参考文献达到55篇

### 第二阶段: Related Work重写

**任务清单**:
- [ ] 经典聚类协议综述 (LEACH, HEED, PEGASIS, TEEN)
- [ ] 环境感知路由综述 (2023-2025)
- [ ] 可靠性增强方法综述
- [ ] 元启发式优化方法综述
- [ ] AERIS定位与创新点阐述

### 第三阶段: 论文精修

**任务清单**:
- [ ] 清理所有TODO标记
- [ ] 统一数值报告
- [ ] 扩展Abstract
- [ ] 校对全文语法
- [ ] 编译最终PDF

### 第四阶段: 验证与提交

**任务清单**:
- [ ] 运行烟雾测试验证代码
- [ ] 确认所有图表可复现
- [ ] 准备Supplementary Materials
- [ ] 提交前最终检查

---

## 八、项目优势总结

1. **统计严谨性出色**: Welch t-test, Bootstrap CI, Holm-Bonferroni校正, Effect size
2. **实验充分**: 250+数据点消融实验，360点敏感性分析，100轮蒙特卡洛
3. **代码完整**: 协议实现、基线对比、统计分析、图表生成全流程
4. **可复现性好**: JSON结果存档、种子固定、脚本化流程
5. **图表质量高**: 论文级PDF/SVG矢量图

---

## 九、风险评估

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 参考文献补充耗时过长 | 中 | 高 | 优先使用已验证的DOI |
| Related Work深度不足 | 低 | 高 | 参考现有docs/中的文献综述草稿 |
| 审稿人质疑实验规模 | 低 | 中 | 已有大规模实验数据支撑 |
| 代码运行环境问题 | 低 | 低 | conda环境配置完善 |

---

## 十、后续工作记录区

### 2026-01-01 记录
- ✅ 完成项目深度评估
- ✅ 识别主要问题: 参考文献不足、Related Work薄弱
- ✅ 制定改进计划
- ✅ 创建评估文档

### 待记录事项
- [ ] 文献补充进度
- [ ] 论文修改进度
- [ ] 实验验证进度
- [ ] 提交状态

---

## 附录A: 关键文件路径速查

| 类别 | 文件路径 |
|------|----------|
| 主协议代码 | `src/aeris_protocol.py` |
| 论文最新版 | `for_submission/final_paper.pdf` |
| LaTeX源文件 | `for_submission/aeris_paper_final.tex` |
| 消融实验数据 | `results/intel_ablation.json` |
| 敏感性数据 | `results/intel_sensitivity.json` |
| 统计检验结果 | `results/significance_compare_intel.json` |
| 图表目录 | `results/plots/` |
| 投稿材料 | `for_submission/` |
| 复现指南 | `docs/Reproduction_Table.md` |

## 进展记录: 实验补齐 (2026-01-01)

已补充/生成:
- 基线对比复现实验: `results/baseline_comparison.json` (Intel Lab几何 + 环境采样, 50次复现)
- 参数敏感性ANOVA: `results/sensitivity_anova_results.json`
- 可扩展性实验: `results/scalability_experiment.json` (30/50/70/100节点, 每规模30次)

新增脚本:
- `scripts/run_baseline_comparison_intel.py`
- `scripts/run_scalability_experiment.py`

下一步:
- 依据新数据更新图表与正文叙述 (基线对比、可扩展性、统计检验)
- 补充实验结果到 `for_submission/final_paper.tex` 并编译

## 附录B: 常用命令

```bash
# 运行烟雾测试
python scripts/smoke_test.py

# 运行基线对比实验
python scripts/run_intel_baselines_all.py

# 生成论文图表
python scripts/plot_paper_figures.py

# 统计显著性检验
python scripts/run_significance_intel.py
```

## 附录C: 项目Git状态

当前分支: main
最近提交:
- 7d77420: Add Jan 1 project evaluation for Sensors
- a2da3d5: docs: rewrite README (AERIS, CN UTF-8)
- e60fcf5: chore: branding rename to AERIS (non-breaking)

---

## 十一、M0里程碑完成记录 (2026-01-01 续)

### 11.1 代码可运行性验证 ✅

**烟雾测试结果**:
```
>>> Starting AERIS simulation (profile: default, max rounds: 100)
   Environment type: outdoor_open
   Node count: 25
   Round 0: alive nodes 25, remaining energy 49.903 J
[SUCCESS] Simulation completed: network ended after 100 rounds.
{'packet_delivery_ratio': 0.9525, 'total_energy_consumed': 10.30J, 'network_lifetime': 100}
```

**基线协议验证**:
| 协议 | PDR | 能耗(J) | 状态 |
|------|-----|---------|------|
| LEACH | 0.00* | 2.29 | ⚠️ 返回值问题 |
| HEED | 0.58 | 7.89 | ✅ 可运行 |
| PEGASIS | 0.96 | 2.23 | ✅ 可运行 |
| AERIS | 0.95 | 10.30 | ✅ 可运行 |

*注：LEACH的PDR=0可能是API返回值问题，协议本身可执行。

### 11.2 已有实验数据盘点

**JSON结果文件**: 85+ 个
**PDF图表文件**: 77 个

**关键数据完整性**:
| 数据集 | 文件 | 样本量 | 状态 |
|--------|------|--------|------|
| 消融实验 | intel_ablation.json | n=50×5=250 | ✅ |
| 效应量 | effect_sizes_summary.json | 5组对比 | ✅ |
| 基线对比 | baseline_comparison.json | n=50×6=300 | ✅ |
| 多拓扑显著性 | significance_compare_multi_topo_50x200.json | n=10×2 | ✅ |
| 动态场景 | dynamic_*_compare.json | 3场景×50 | ✅ |
| 大规模 | large_scale_long.json | 300/500节点 | ✅ |

### 11.3 图表质量严格评估 (按规范要求)

**评估标准** (来自 `项目开发规范提示词.md`):
- 信息结构：至少1组"同类多面板组合图"（建议≥8子图）
- 对比充分：每子图≥3条对比线
- 统计可信：误差带/置信区间/多seed，且标注n和统计口径
- 可读性：统一坐标范围/刻度、图例不遮挡
- 可复现：图由脚本生成

**现有图表评估结果**:

| 图表类别 | 代表文件 | 子图数 | 对比线数 | 误差带 | 判定 |
|----------|----------|--------|----------|--------|------|
| Intel基线面板 | paper_intel_baselines_panels.pdf | 2 | 6 | ✅ | ⚠️ 需扩展子图 |
| 消融PDR | paper_intel_ablation_pdr.pdf | 1 | 5 | ✅ | ❌ 单图不足 |
| 多拓扑显著性 | paper_multi_topo_sig_pdr.pdf | 1 | 2 | ✅ | ❌ 单图不足 |
| 动态场景 | paper_dynamic_corridor_curated.pdf | 2 | 3 | ✅ | ⚠️ 需合并 |
| 网关热力图 | paper_gateway_limit_heatmap_*.pdf | 1 | - | ✅ | ❌ 需组合 |
| 流程图 | paper_method_flowchart.pdf | 1 | - | - | ✅ 架构图可单独 |

**图表质量总结**:
- **Pass**: 0 个完全符合多面板标准
- **需改进**: 77 个 (全部需要重新组织为多面板组合)

### 11.4 图表改进计划

按规范要求，重新规划5组多面板图表：

**图组1: 环境-链路关联 (3×2=6子图)**
- 子图1-3: 温度/湿度/RSSI三指标的时序变化
- 子图4-6: 三种环境类型下的PDR-距离关系
- 对比线: Intel实测 vs Log-Normal模型 vs AERIS预测

**图组2: 消融实验森林图 (2×4=8子图)**
- 子图1-4: 各组件效应量(Hedges' g)及95%CI
- 子图5-8: PDR/Energy/Lifetime/Fairness四指标对比
- 对比线: FULL vs 各消融配置

**图组3: 基线对比全景 (3×3=9子图)**
- 行: Uniform/Corridor31/Corridor41 三拓扑
- 列: PDR/Energy/Lifetime 三指标
- 对比线: LEACH/HEED/PEGASIS/TEEN/AERIS-E/AERIS-R

**图组4: 动态场景综合 (2×3=6子图)**
- 行: 走廊渐变/基站移动/节点掉线
- 列: PDR时序/能耗累积
- 对比线: Baseline vs AERIS，含阶段分界

**图组5: 统计验证汇总 (2×4=8子图)**
- 子图1-4: Gardner-Altman配对差异图
- 子图5-8: Bootstrap分布/p值热力图/效应量森林图

---

## 十二、下一步行动计划

### 12.1 立即执行 (同事任务)

**任务1**: 验证LEACH的PDR返回值问题
- 输入: `src/baseline_protocols/leach_protocol.py`
- 产出: 修复后的协议或问题报告
- 验收: LEACH PDR > 0

**任务2**: 生成多面板图表脚本
- 输入: 现有绘图脚本 `scripts/plot_paper_figures.py`
- 产出: 新脚本生成5组多面板图
- 验收: 每组图≥6子图，每子图≥3对比线

### 12.2 等待审查

以下事项等待用户确认后执行：
1. 是否需要补充2023-2025年新基线方法？（已确认：是）
2. 图表配色方案偏好？
3. 是否需要中文版本图表？

---

*本文档将持续更新，记录项目完善过程中的所有重要信息。*
*最后更新: 2026-01-01 (M0完成)*

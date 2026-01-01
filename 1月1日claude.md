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

*本文档将持续更新，记录项目完善过程中的所有重要信息。*
*最后更新: 2026-01-01*

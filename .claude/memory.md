# AERIS 项目核心记忆（Claude 会话间持久化）

> 本文件记录跨会话需要保留的关键状态、决策和经验。每次 `/onboard` 时读取。

---

## 1. 项目身份

| 字段 | 值 |
|------|-----|
| 项目名称 | AERIS-WSN-Protocol |
| 目标期刊 | MDPI Sensors (IF=3.9, Q2) |
| 当前分支 | `v50-rigor` |
| 最新论文 | `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260304_v98.tex` |
| 投稿包 | `AERIS_Sensors_Submission/manuscript.tex` |
| 核心指标 | `pdr_expected = bs_delivered / source_packets_expected` |

---

## 2. 架构决策记录（ADR）

### ADR-001: 双目标编译架构
- **决策**: 维护两个 TeX 编译目标（开发版 + 投稿版）
- **原因**: MDPI 投稿要求短文件名 (fig1-fig10)，开发版保留带时间戳的长文件名便于版本追踪
- **约束**: 两者文本必须 TEXT DIFF=0（仅路径差异）

### ADR-002: 图表生成脚本链
- **决策**: s93（基础库）→ s97（Fig6等）→ s98（Fig2升级）
- **原因**: 增量开发，每个版本只修改需要更新的图，其余从基础库继承
- **风险**: s98 运行时会同时重新生成 fig2 和 fig6，容易遗漏 fig6 同步

### ADR-003: Holm 校正 family 定义
- **决策**: 每个表/图定义独立 family（100-node m=4, ablation m=8, scalability m=24/env）
- **原因**: 避免跨表混合增大 family size 导致过度保守
- **教训**: v98 会话中发现 raw p 被误标为 Holm p（P0 级错误）

### ADR-004: 统一英文注释
- **决策**: 所有 Python 脚本注释统一使用英文
- **原因**: Windows GBK 编码环境下中文注释导致 UnicodeEncodeError

---

## 3. 投稿包图表映射（10 张）

```
fig1.pdf  ↔ fig0_aeris_workflow_20260302_s97.pdf       (AERIS 流程图)
fig2.pdf  ↔ fig1_env_pdr_panel_20260302_s97.pdf        (100节点环境对比)
fig3.pdf  ↔ fig2_ablation_panel_20260304_s98.pdf       (消融3面板)
fig4.pdf  ↔ fig3_scalability_panel_20260302_s97.pdf     (可扩展性趋势)
fig5.pdf  ↔ fig8_s8_significance_heatmap_20260302_s97.pdf (显著性热图)
fig6.pdf  ↔ fig6_s10_delta_maps_20260302_s97.pdf       (功率敏感性delta)
fig7.pdf  ↔ fig10_s10_absolute_profiles_20260302_s97.pdf (绝对PDR曲线)
fig8.pdf  ↔ fig5_s11_patch_control_delta_20260302_s97.pdf (压力测试delta)
fig9.pdf  ↔ fig4_tradeoff_panel_20260302_s97.pdf       (权衡面板)
fig10.pdf ↔ fig7_ns3_trend_panel_20260302_s97.pdf      (NS-3趋势验证)
```

---

## 4. 关键数据源（证据白名单摘要）

| ID | 文件 | 用途 |
|----|------|------|
| W1 | `results/compare_50x200.json` | 100节点5协议对比 |
| W2 | `results/intel_ablation.json` | 消融实验 |
| W3 | `results/large_scale_scalability_verified.json` | 主力大规模矩阵 |
| W4 | `results/significance_compare_50x200.json` | 100节点显著性 |
| W5 | `results/significance_compare_multi_topo_50x200.json` | 多环境显著性 |
| W6 | `ns3_validation/results/` | NS-3 趋势验证 |
| W7 | `results/corridor_*_grid_50x200.json` | 功率敏感性矩阵 |
| W8 | `results/multitest_holm_bonferroni.json` | Holm 校正统计 |

---

## 5. 已知陷阱（按严重度排序）

### P0 级（曾导致阻塞）
1. **Raw p ≠ Holm p**: 消融节 raw p=0.000657 被误报为 "p<0.001"，实际 Holm p=0.00526
2. **投稿包图表未同步**: 脚本重新生成图后只同步了 fig3，遗漏了同时重新生成的 fig6
3. **`\texttt{}` 残留**: MDPI 正文禁止代码变量名，需用自然语言替代

### P1 级（曾导致返工）
1. **conda run 多行脚本失败**: Windows 下必须写入临时 .py 文件
2. **多面板 y 轴顺序不一致**: Panel (c) 的环境排列与 Panel (b) 相反
3. **方法节遗漏 family m 值**: ablation m=8 未在方法节声明
4. **tabular 列数不匹配**: `{lcccccc}` 声明 7 列但数据只有 6 列

### P2 级（建议级）
1. **图表字号 <9pt**: 8pt 在打印时不可读，统一≥9pt
2. **脚本中文注释**: Windows GBK 下报 UnicodeEncodeError

---

## 6. 环境与工具速查

| 场景 | 正确方式 | 错误方式 |
|------|----------|----------|
| 本地 Python | `C:/Users/admin/anaconda3/envs/aether-wsn/python.exe` | `conda run -n aether-wsn python -c "多行"` |
| LaTeX 编译 | `pdflatex → bibtex → pdflatex × 2` | 只跑一次 pdflatex |
| SSH 服务器 | `ssh FatMachine` | `ssh admin@100.104.82.45` |
| 远程 Python | `C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe` | `conda activate` |
| 大 JSON 读取 | `python -c "import json; ..."` 提取字段 | 全量 Read 到对话 |

---

## 7. 版本里程碑

| 版本 | 日期 | 关键变更 |
|------|------|----------|
| v68 | 2026-02-25 | 首次通过严格审查 P0=0 |
| v93 | 2026-03-01 | 基础图表库定型（s93 脚本） |
| v97 | 2026-03-02 | Fig6 无数值标注、Fig2 GridSpec 布局 |
| v98 | 2026-03-04 | Fig2 升级为 3 面板（热图+边际效应+显著性）、p 值修复 |

---

## 8. 待办事项

- [ ] Fig7 (NS-3 trend panel) 美观度升级（用户明确不满意）
- [ ] 剩余 P2 建议项（方程硬编码引用、360 比较 family 说明等）

---

*最后更新: 2026-03-04, v98 会话*

# AERIS 项目核心记忆（Claude 会话间持久化）

> 本文件记录跨会话需要保留的关键状态、决策和经验。每次 `/onboard` 时读取。
> **此文件仅限本地使用，已加入 .gitignore，不上传 GitHub。**

---

## 1. 项目身份

| 字段 | 值 |
|------|-----|
| 项目名称 | AERIS-WSN-Protocol |
| 目标期刊 | MDPI Sensors (IF=3.9, Q2) |
| 当前分支 | `main`（从 v50-rigor 孤儿分支清理而来） |
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

### ADR-005: 仓库瘦身（2026-03-05）
- **决策**: 用孤儿分支清理历史，从 5.79 GiB 降到 85 MiB
- **原因**: .tools/（MiKTeX 安装包）、node_modules/ 等占 ~5GB，超出 GitHub 2GB 限制
- **影响**: 旧 v50-rigor 和 main 分支历史已清除，当前只有一个 commit

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

## 4. 已知陷阱（按严重度排序）

### P0 级（曾导致阻塞）
1. **Raw p ≠ Holm p**: 消融节 raw p=0.000657 被误报为 "p<0.001"，实际 Holm p=0.00526
2. **投稿包图表未同步**: 脚本重新生成图后只同步了 fig3，遗漏了同时重新生成的 fig6
3. **仓库超大**: .tools/ 里 MiKTeX 等占 5GB+，超出 GitHub 2GB pack 限制

### P1 级（曾导致返工）
1. **conda run 多行脚本失败**: Windows 下必须写入临时 .py 文件
2. **多面板 y 轴顺序不一致**: Panel (c) 的环境排列与 Panel (b) 相反
3. **方法节遗漏 family m 值**: ablation m=8 未在方法节声明

### P2 级（建议级）
1. **图表字号 <9pt**: 8pt 在打印时不可读，统一≥9pt
2. **脚本中文注释**: Windows GBK 下报 UnicodeEncodeError

---

## 5. 环境与工具速查

| 场景 | 正确方式 | 错误方式 |
|------|----------|----------|
| 本地 Python | `C:/Users/admin/anaconda3/envs/aether-wsn/python.exe` | `conda run -n aether-wsn python -c "多行"` |
| LaTeX 编译 | `pdflatex → bibtex → pdflatex × 2` | 只跑一次 pdflatex |
| SSH 服务器 | `ssh FatMachine` | `ssh admin@100.104.82.45` |
| GitHub SSH | `git@github.com:Deepmind666/AERIS-WSN-Protocol.git` | HTTPS（超大仓库会 500） |
| 大 JSON 读取 | `python -c "import json; ..."` 提取字段 | 全量 Read 到对话 |

---

## 6. 待办事项

- [ ] Fig7 (NS-3 trend panel) 美观度升级（用户明确不满意）
- [ ] 剩余 P2 建议项（方程硬编码引用、360 比较 family 说明等）

---

*最后更新: 2026-03-05, v98 会话*

---

## LCN26 memory updates

- [2026-04-30] [Decision] For current AERIS work, ignore old Sensors submission state unless explicitly requested; focus only on LCN26.
- [2026-04-30] [Finding] The original LCN26 dual-machine card says FatMachine owns NS-3 reruns while local owns the AERIS mechanism matrix. The 2026-04-30 expanded NS-3 batch appears to have run through FatMachine scripts only; if the intended plan was to split NS-3 shards across both local and FatMachine, that still needs explicit execution/provenance.
- [2026-04-30] [Finding] LCN26 NS-3 supplementary/expanded audit exists at `ns3_validation/results/lcn26_ns3_expanded_20260430_173108/summary/`. It was launched through FatMachine scripts and completed 28/28 shards, 2520 experiments: protocols `AERIS, LEACH, HEED, PEGASIS, TEEN, RPL-MRHOF, CTP`, environments `indoor_office, indoor_factory, outdoor_suburban, outdoor_urban`, nodes `100,500,1000`, `n=30` per cell.
- [2026-04-30] [Finding] Expanded NS-3 results change the story if treated as canonical: CTP/RPL-MRHOF beat AERIS in office; RPL-MRHOF is strongest in factory and urban; AERIS remains strongest or effectively tied in suburban. Keep the existing 5-protocol corrected NS-3 audit as the current paper anchor until the expanded baseline interpretation is deliberately rewritten.
- [2026-04-30] [Finding] Dual-machine LCN26 NS-3 sweep completed and merged at `ns3_validation/results/lcn26_ns3_dual_combined_20260430_191527_191528/summary/`: 28 shards, 5880 experiments, 7 protocols, 4 environments, nodes `50,100,200,300,500,800,1000`, `n=30` per cell. Local ran office+factory; FatMachine ran suburban+urban. Key winners: office CTP, factory mostly RPL-MRHOF except 50-node AERIS, suburban AERIS except 1000-node near-tie/RPL-MRHOF, urban RPL-MRHOF.

---

## AERIS writing and figure preferences

- Write like a strict IEEE/LCN reviewer, not a cheerleader. Keep claims bounded and evidence-linked.
- For the paper, separate the evidence layers clearly: classical NS-3 audit, expanded seven-protocol boundary sweep, ablation, strict-physics stress layer, and mechanism/trade-off study.
- Do not let figures, captions, formulas, and prose describe different baselines or different denominators.
- Prefer compact, readable figures over large multi-row layouts. Avoid overcrowded heatmaps when a bar chart, margin plot, CDF, or small multiple is clearer.
- Keep figure/table text aligned with the venue template. Do not invent formatting; follow the template exactly.
- Keep subfigure titles, legends, and captions consistent with the plotted data and the actual claim boundary.
- Use Times-like fonts in plots and keep plot text readable in the final PDF.
- Avoid large blank spaces and avoid gluing tables and figures together with little or no explanatory text.
- If a table or figure needs a long verbal defense, simplify the design or add a clearer caption and supporting sentence.
- Method sections should be concise but not skeletal: enough equations, symbols, thresholds, and fallback logic must be present for reproduction.
- If coefficients or thresholds are fixed by hand, say so explicitly; do not imply a search or optimization unless one was actually run.
- Do not overstate AERIS. The stable story is: reliability-first, strong in selected harsh regimes, with a clear reliability-lifetime trade-off and a deployment boundary once stronger collection-tree/RPL-style baselines are included.
- For deadlines, prioritize removing contradictions, unsupported claims, and reproducibility gaps over adding new experiments.

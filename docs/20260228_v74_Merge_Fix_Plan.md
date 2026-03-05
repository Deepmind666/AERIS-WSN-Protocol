# v74 合并修复计划（Claude v72 审稿 + Codex v73 审查）

**日期**: 2026-02-28 | **基准**: v73.tex + build_sensors_figures_s73.py
**原则**: 不加新实验，只改绘图规范和文稿表达

---

## 已确认 v73 已修复的项（无需再动）

- [x] Fig 10 (absolute profiles) 15线拆分 → 已改为 2×4 面板
- [x] Fig 6 (delta heatmap) colorbar 标签 → 已改为 "pp"
- [x] Table 5 regime_map → 已更新为 S10R 正确描述
- [x] PEGASIS 技术假说 → 已补充

---

## v74 修复清单

### A. 绘图脚本修改 (build_sensors_figures_s73.py → s74)

#### A1. 全局字号下限提升到 ≥9pt [P1-2]
- line 741: `fontsize=7.6` → `fontsize=9.0` (fig7 delta@1000 标注)
- line 763: `fontsize=7.6` → `fontsize=9.0` (fig7 non-significant 说明)
- line 821: `fontsize=7.8` → `fontsize=9.0` (fig8 heatmap cell 数值)
- line 850: `fontsize=7.5` → `fontsize=9.0` (fig8 底部注释)
- line 866: `fontsize=7.8` → `fontsize=9.0` (fig8 colorbar 注释)

#### A2. Fig 1 (fig1) outdoor_urban 低值区 inset [P1-1]
- 在 outdoor_urban 面板添加 inset axes（放大 y=0~0.3 区域）
- caption 补充 "unified 0–1 y-axis with inset for low-PDR detail"

#### A3. NS-3 图 (fig7) context 线降透明 [P2-3 + P1-F6]
- PEGASIS/HEED/TEEN context 线: alpha 降至 0.35, linewidth 降至 1.0
- AERIS/LEACH 主比较线: linewidth 提升至 2.5, alpha=1.0
- 非显著标记: 改为红色空心圆 (已有) + 增大 markersize

#### A4. Fig 8 (tradeoff) energy 子面板加单位 [P2-F4]
- y轴标签加 "(J)" 或 "(mJ)"

#### A5. "ranking" 图题改名 [P2-2]
- lines 475-478 附近: "ranking" → "Reliability profile" / "Energy profile" 等

### B. 论文 tex 修改 (v73.tex → v74.tex)

#### B1. 删除调试句 [P1-3]
- line 152: 删除 "Figure asset filenames are stable aliases..."

#### B2. 精简 4 个核心 caption [P1-4]
- line 331 (fig3 scalability): 删除解释性判断，只保留"图显示什么+符号含义"
- line 434 (fig6 delta maps): 精简，移除括号内冗余解释
- line 441 (fig10 absolute profiles): 精简
- line 534 (fig7 NS-3 trend): 精简，解释性内容移回正文

#### B3. 术语规范化 [P2-1]
- "S10R" → 首处定义后改为 "power-sensitivity matrix"
- "patch-control" → "matched stress matrix"
- "strict-physics block" → "physics-fidelity block"

### C. 不做的事

- 不加新实验
- 不改数据/统计
- 不改 Fig 2 (ablation) / Fig 3 (scalability) / Fig 4 (tradeoff) 的核心内容
- 不改配色方案（已合规）

---

## 执行顺序

1. 复制 s73.py → s74.py，执行 A1-A5
2. 运行 s74.py 生成新图表
3. 复制 v73.tex → v74.tex，执行 B1-B3
4. 编译 v74.pdf
5. 快速自审 pass 确认无残留问题

# v74 图文硬修提示词（Codex 执行，Claude 评审）

**日期**: 2026-02-28
**基准文件**: `scripts/build_sensors_figures_s73.py` + `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260228_v73.tex`
**原则**: 不加新实验，不改数据，不改配色方案，只改绘图规范和文稿表达。

---

## 任务总览

你需要产出两个文件：
1. `scripts/build_sensors_figures_s74.py`（从 s73 复制后修改）
2. `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260228_v74.tex`（从 v73 复制后修改）

然后运行绘图脚本生成新图表，编译 tex 生成 v74.pdf。

---

## Part A: 绘图脚本修改（s73.py → s74.py）

### A1. 全局字号下限提升到 ≥9pt

精确修改以下 5 处（行号基于 s73.py）：

1. **line 741** `fontsize=7.6` → `fontsize=9.0`
   - 位置: `plot_fig7_ns3_trend()` 内 delta@1000 标注框
2. **line 763** `fontsize=7.6` → `fontsize=9.0`
   - 位置: `plot_fig7_ns3_trend()` 内 "red hollow circle" 说明文字
3. **line 821** `fontsize=7.8` → `fontsize=9.0`
   - 位置: `plot_fig8_s8_significance_heatmap()` 内 heatmap cell 数值标注
4. **line 850** `fontsize=7.5` → `fontsize=9.0`
   - 位置: `plot_fig8_s8_significance_heatmap()` 内底部注释
5. **line 866** `fontsize=7.8` → `fontsize=9.0`
   - 位置: `plot_fig8_s8_significance_heatmap()` 内 colorbar 区域注释

**验证方法**: `grep -n "fontsize=7\." scripts/build_sensors_figures_s74.py` 应返回 0 行。

### A2. Fig 1 (env_pdr_panel) outdoor_urban 加 inset

在 `plot_fig1_env_pdr_panel()` 函数中，对 outdoor_urban 面板（第3个面板，index=2）添加 inset axes：

```python
# 在 outdoor_urban 面板绘制完成后添加：
if env == "outdoor_urban":
    inset = ax.inset_axes([0.55, 0.45, 0.42, 0.50])  # [x, y, width, height] 相对坐标
    # 重绘该面板的数据到 inset，y 轴限制为 [0, 0.35]
    for proto in PROTOCOL_ORDER:
        # ... 复用同样的绘图逻辑 ...
        pass
    inset.set_ylim(0, 0.35)
    inset.set_title("Low-PDR detail", fontsize=9)
    inset.tick_params(labelsize=8)
    # 用灰色虚线框标注放大区域
    ax.indicate_inset_zoom(inset, edgecolor="gray", linestyle="--", alpha=0.6)
```

**注意**: inset 内的线条/marker 样式必须与主图一致，不需要单独图例。

### A3. NS-3 图 (fig7) 主次层级重构

在 `plot_fig7_ns3_trend()` 函数中：

1. **context 线降级**（PEGASIS, HEED, TEEN）:
   - `alpha` 降至 `0.30`（当前约 0.7-0.8）
   - `linewidth` 降至 `1.0`（当前约 1.8）
   - 不加 marker（或 marker 极小 markersize=2）

2. **主比较线升级**（AERIS, LEACH）:
   - `linewidth` 提升至 `2.8`
   - `alpha=1.0`
   - `markersize` 提升至 `7`

3. **图例分组**: 在图例中用分隔线或标签区分 "Primary pair" 和 "Context"。可以用两行图例：
   ```
   — AERIS  -- LEACH  (primary comparison)
   ·· PEGASIS  ·· HEED  ·· TEEN  (context)
   ```

4. **非显著标记**: 红色空心圆 markersize 提升至 `10`，确保不被主线遮挡。

### A4. Fig 7 (tradeoff) energy 子面板加单位

在 `plot_fig7_tradeoff()` 函数中（注意：论文中的 Fig 7 tradeoff 对应脚本中的 plot_fig7 或类似函数名）：

找到 energy 子面板的 xlabel 或 ylabel，确保包含单位 "(J)"。

当前 line 476: `("energy", "Average total energy (J)", False, "Energy ranking")`
- xlabel 已有 "(J)"，确认 bar chart 的轴标签确实显示了这个字符串。如果是 ylabel 缺失，补上。

### A5. "ranking" 图题改为 "profile"

修改 line 475-478 的 title 字段：

```python
metrics = [
    ("pdr", "Average PDR", True, "Reliability profile"),
    ("energy", "Average total energy (J)", False, "Energy profile"),
    ("hops", "Average hops to BS", False, "Hop-latency profile"),
    ("life", "Average lifetime (rounds)", False, "Lifetime profile"),
]
```

### A6. 更新 SUFFIX

```python
SUFFIX = "20260228_s74"
```

---

## Part B: 论文 tex 修改（v73.tex → v74.tex）

### B1. 删除调试句

**删除 line 152 整行**:
```
Figure asset filenames are stable aliases for reproducible builds; manuscript numbering and references follow LaTeX labels rather than filename suffixes.
```

### B2. 精简 4 个 caption

#### B2-1. line 331 (fig3 scalability)

当前（过长）:
```
Primary large-scale scalability trends over node counts (balanced n=3200 per environment-node-protocol cell) with 95\% CI bands. A common 0--1 y-axis is used across panels for direct visual comparability; in several cells, CI bands are narrower than line width due to the large sample size. In indoor\_office, PEGASIS remains near a high-stability plateau (\(\sim\)0.99), which is shown explicitly rather than rescaled to preserve cross-environment comparability.
```

改为:
```
Primary large-scale scalability trends (n=3200 per cell, 95\% CI bands, common 0--1 y-axis).
```

将删除的解释内容移到正文中该图的引用段落。

#### B2-2. line 434 (fig6 delta maps)

当前（过长）:
```
Full-matrix power sensitivity (tx5 vs tx15). Panels (a)--(d) show per-environment protocol-by-scale delta maps (\(\Delta\)PDR = tx5 -- tx15, in percentage points) for all 120 tested cells (4 environments \(\times\) 5 protocols \(\times\) 6 node scales). Cross markers denote non-significant cells after Holm correction (none in this slice).
```

改为:
```
Power-sensitivity delta maps (\(\Delta\)PDR = tx5 $-$ tx15, percentage points; 120 cells). Cross markers: Holm non-significant.
```

#### B2-3. line 441 (fig10 absolute profiles)

当前（过长）:
```
Absolute PDR profiles across the same S10R matrix, split for readability into an AERIS-focused row and a baseline-focused row. Line style denotes tx5/tx10/tx15, while color denotes protocol. This view complements Figure~\ref{fig:s10_power_sensitivity} by showing absolute levels rather than only deltas.
```

改为:
```
Absolute PDR profiles from the power-sensitivity matrix. Top row: AERIS (line style = tx level). Bottom row: baselines.
```

#### B2-4. line 534 (fig7 NS-3 trend)

当前（过长）:
```
NS-3 trend panel over 50--1000 nodes. AERIS and LEACH are emphasized because Holm significance markers and \(\Delta@1000\) annotations are defined for this pair; PEGASIS/HEED/TEEN are shown as contextual curves only. Red hollow-circle markers indicate node scales where the AERIS--LEACH difference is not significant after Holm correction.
```

改为:
```
NS-3 trend validation (50--1000 nodes). Primary pair: AERIS vs LEACH (bold lines, significance markers). Context: PEGASIS/HEED/TEEN (faded). Red hollow circles: Holm non-significant scales.
```

### B3. 术语规范化

在全文中做以下替换（首次出现保留原名并定义，后续统一用新名）：

1. **S10R** → 首次: "the power-sensitivity matrix (S10R)" → 后续全部用 "power-sensitivity matrix"
2. **patch-control** → 首次: "matched stress comparison (patch-control)" → 后续用 "matched stress comparison"
3. **strict-physics block** → 首次: "physics-fidelity block (strict-physics)" → 后续用 "physics-fidelity block"

**注意**: Table 5 (regime_map) 中保留原始技术名称不变，因为那是定义表。

### B4. 更新图表文件引用

将所有 `_s73` 或 `_s70` 后缀的图表文件名更新为 `_s74`（如果 tex 中硬编码了文件名的话）。如果 tex 使用的是稳定别名（如 `fig1_env_pdr_panel.pdf`），则无需修改。

---

## Part C: 执行与验证

### C1. 运行绘图脚本
```bash
conda run -n aether-wsn python scripts/build_sensors_figures_s74.py
```

### C2. 复制图表到稳定别名（如果脚本输出带 SUFFIX）
确保 tex 引用的文件名能找到新生成的图表。

### C3. 编译 tex
```bash
cd for_submission
pdflatex AERIS_Sensors_MDPI_Submission_Draft_20260228_v74.tex
bibtex AERIS_Sensors_MDPI_Submission_Draft_20260228_v74
pdflatex AERIS_Sensors_MDPI_Submission_Draft_20260228_v74.tex
pdflatex AERIS_Sensors_MDPI_Submission_Draft_20260228_v74.tex
```

### C4. 自审 checklist（完成后逐项确认）

- [ ] `grep -n "fontsize=7\." scripts/build_sensors_figures_s74.py` 返回 0 行
- [ ] `grep -n "Figure asset filenames" for_submission/*v74.tex` 返回 0 行
- [ ] `grep -n "ranking" scripts/build_sensors_figures_s74.py` 中无 "ranking" 作为图标题
- [ ] v74.pdf 编译无错误
- [ ] Fig 1 outdoor_urban 面板有 inset
- [ ] NS-3 图中 context 线明显比主线淡
- [ ] 4 个 caption 均不超过 2 行

---

## 禁止事项

1. 不得修改任何数据文件（JSON/CSV）
2. 不得修改 PROTO_COLORS 配色
3. 不得修改统计方法或 p-value 阈值
4. 不得添加新的图表（只改现有图表）
5. 不得修改 Table 5 (regime_map) 的内容
6. 不得修改 Fig 2 (ablation) 和 Fig 3 (scalability) 的核心绘图逻辑

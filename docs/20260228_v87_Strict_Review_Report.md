# AERIS v87 严格审稿报告（Sensors MDPI 资深审稿人视角）

**审稿日期**: 2026-02-28
**论文版本**: AERIS_Sensors_MDPI_Submission_Draft_20260228_v87.tex (633行)
**审查者**: 独立会话严格审稿
**数据版本**: 与 v79 相同（无新实验）
**对照基准**: v79 审稿报告 + v79 独立审查报告
**图脚本**: build_sensors_figures_s87.py

---

## 总体判定：Accept（条件性，偏强 Accept）

v87 在 v79 基础上完成了三项关键图表改进和术语规范化。v79 遗留的 3 个 P1 中，P1-1（patch-control 术语）已完全修复。P1-2 和 P1-3 部分改善。当前无 P0 级阻塞项。

---

## v79→v87 关键变更确认

| 编号 | v79 问题 | v87 状态 |
|------|---------|---------|
| P1-1 | "patch-control" 术语未规范化（18处） | ✅ 已全部替换为 "stress-delta" / "patch-minus-control"，0 处残留 |
| P1-2 | outdoor_urban 800节点高方差未讨论 | ⚠️ 部分改善：line 454 新增 CV 值和 "intermittent percolation" 解释 |
| P1-3 | indoor_factory 500节点反转未具体讨论 | ⚠️ 部分改善：line 454 新增 tx10>tx5>tx15 具体数值 |
| 图表 | Fig 1 inset 重复结构 | ✅ 已改为单面板 + low-range band annotation |
| 图表 | Fig 3 per-panel inset 重复 | ✅ 已删除 inset，改为指向 Fig 10 |
| 图表 | Fig 6 语义不清 | ✅ 已明确全覆盖矩阵 + Holm 非显著标记 |

---

## 逐项检查结果

### 1. 调试句/术语残留
- `S10R / s10r`: 0 匹配 — PASS
- `patch-control`: 0 匹配 — PASS（v79 有 18 处，v87 全部替换为 stress-delta）
- `debug/TODO/FIXME/HACK/XXX/TEMP/Figure asset/print(`: 0 匹配 — PASS

### 2. 图引用一致性
- tex 中 10 个 `\includegraphics`，全部指向 `_20260228_s87.pdf` — PASS
- 文件系统 12 个 s87 PDF 齐全（fig0-fig11）
- fig9/fig11 为备用资产，v79 起即未在正文引用，非遗漏

### 3. 绘图脚本字号（build_sensors_figures_s87.py）
- axes.labelsize: 12.8pt — PASS
- xtick/ytick.labelsize: 11.0pt — PASS
- legend.fontsize: 10.6pt — PASS
- fontsize<8pt 残留: 0 处 — PASS

### 4. 数据一致性
- v87 未引入新实验数据，所有 Table 数值与 v79 完全一致
- 抽检 4 个关键数值（0.9739, 0.6771, 0.8176, 0.0582）均在正确位置 — PASS
- v79 已完成 52/52 cells 全量交叉验证，v87 继承该结果

### 5. Caption 检查
- Fig 1 (line 248): 单面板 + low-range band，≤2行 — PASS
- Fig 3 (line 330): 删除 inset 描述，指向 Fig 10，≤2行 — PASS
- Fig 6 (line 442): 全覆盖矩阵语义明确，≤2行 — PASS
- 其余 caption 与 v79 一致 — PASS

---

## P0 — 阻塞发布

无。

---

## P1 — 应修复

### [P1-1] "stress-delta" 术语首处缺少正式定义
- **位置**: v87.tex:367（首次出现 "Stress-Delta Matrix A"）
- **问题**: v87 将 "patch-control" 全部替换为 "stress-delta"，术语本身更学术化，但首次出现时未给出明确定义（如 "stress-delta: the difference in PDR between patch and control arms"）。审稿人可能在首次遇到时不理解 "stress-delta" 的具体含义。
- **修复**: 在 line 367 首次出现处加一句括号定义。
- **工作量**: 2 min

### [P1-2] outdoor_urban 800节点高方差：已改善但可再加一句
- **位置**: v87.tex:454
- **问题**: v87 新增了 CV 值和 "intermittent percolation" 解释，比 v79 有实质改善。但 Discussion/Limitations 中仍未显式提及 mean±std 在高 CV 场景下的局限性。
- **修复**: 在 Limitations 中加一句 "In high-CV cells (e.g., outdoor\_urban at 800 nodes), mean±std may understate distributional complexity."
- **工作量**: 3 min

---

## P2 — 建议改进（全部可选）

### [P2-1] Fig 10 下行 baseline 面板 y 轴范围差异大
- **位置**: fig10_s10_absolute_profiles 下行面板
- **修复**: 可选——在每个面板右上角标注 y 轴范围提示

### [P2-2] 能耗分析仍偏弱
- **位置**: v87.tex:501-516
- **修复**: 建议在 Limitations 中显式声明 future work

### [P2-3] indoor_office PEGASIS 极端差距未充分讨论
- **位置**: 数据层面
- **修复**: 可选——Discussion 中补 1 句

---

## v79→v87 改进确认（正面评价）

1. **术语规范化彻底**：patch-control → stress-delta / patch-minus-control，全文 0 残留
2. **Fig 1 单面板化**：消除了"子图套图"的重复结构，审稿人不会再质疑信息冗余
3. **Fig 3 结构统一**：删除 per-panel inset，改为指向 Fig 10，避免"同一信息重复绘制"
4. **Fig 6 语义明确**：全覆盖矩阵 + Holm 非显著标记，审稿人可直接验证
5. **Conclusion 新增具体示例**：line 617 "indoor\_factory 500-node cell favors tx10"，比 v79 更具体
6. **line 454 新增 percolation 解释**：outdoor\_urban 高方差有了物理机制假说

---

## 修复优先级汇总

| 优先级 | 编号 | 预计工作量 |
|--------|------|-----------|
| P1 | P1-1 stress-delta 首处定义 | 2 min |
| P1 | P1-2 高 CV 局限性声明 | 3 min |
| P2 | P2-1~P2-3 | 各 5 min，全部可选 |

---

## 结论

v87 相比 v79 有明确质量提升：术语规范化彻底（patch-control 0 残留）、三个关键图表消除了重复结构、Conclusion 新增具体部署示例。P0=0，P1 从 v79 的 3 项降至 2 项（且均为 2-3 分钟的文字补充），P2=3（全部可选）。

建议：修完 P1-1（stress-delta 首处定义）和 P1-2（高 CV 局限性声明）后即可作为送审稿。P2 项可在审稿人反馈后视情况处理。

v87 已达到 Sensors MDPI 送审水平。

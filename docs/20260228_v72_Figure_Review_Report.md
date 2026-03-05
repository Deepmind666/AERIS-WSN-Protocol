# AERIS v72 图表质量审稿报告

**日期**: 2026-02-28 | **版本**: v72 | **重点**: 图表可读性与出版质量

---

## v70 修复确认

| 编号 | 状态 | 说明 |
|------|------|------|
| P0-1 | ✅ | Table 5 已更新 |
| P1-4 | ✅ | PEGASIS 技术假说已补 |
| Abstract | ✅ | S10R 口径已修正 |

---

## 逐张审查

### Fig 1-3: ✅ PASS
- Fig 1 (workflow): 清晰完整
- Fig 2 (100-node bar): 配色好，误差棒完整
- Fig 3 (ablation heatmap): 数值标注清晰

### Fig 4 (scalability): ⚠️ CONDITIONAL
- **[P1-F1]** indoor_office PEGASIS ~0.99 水平线 vs 其他协议急剧下降，视觉反差极端。需补文字说明。

### Fig 5 (delta heatmap): ❌ NEEDS FIX
- **[P1-F2]** 120 cell 数值字号~7pt，打印不可读。需增至 8-9pt。
- **[P1-F3]** colorbar 写 "percentage points" 但数值是小数（+0.277 非 +27.7）。单位标签与数值不一致。

### Fig 6 (absolute profiles): ❌ NEEDS FIX — 本轮最大问题
- **[P1-F4]** 每面板 15 条线（5协议×3功率），低 PDR 区严重重叠，outdoor_urban 中 LEACH/HEED/TEEN 完全重合于 y≈0。
- **[P1-F5]** dashed/dotted/solid 在密集线条中无法区分。建议改用同色系深/中/浅区分功率。

### Fig 7 (patch-control delta): ✅ PASS
- **[P2-F3]** 100节点处 4 环境线几乎重合，可加 marker 微调。

### Fig 8 (tradeoff): ⚠️ CONDITIONAL
- **[P2-F4]** energy 子面板缺单位标注（J? mJ?）。

### Fig 9 (NS-3 trend): ⚠️ CONDITIONAL
- **[P1-F6]** x marker（非显著）与线条重叠辨识度低。建议改为红色空心圆或灰色背景条。
- **[P2-F5]** delta@1000 标注框字号偏小。

---

## 汇总

| 图 | 判定 | 关键问题 |
|----|------|----------|
| 1-3 | ✅ PASS | — |
| 4 | ⚠️ | P1-F1: PEGASIS 水平线需说明 |
| 5 | ❌ | P1-F2/F3: 字号+单位 |
| 6 | ❌ | P1-F4/F5: 15线重叠 |
| 7 | ✅ | P2 微调 |
| 8 | ⚠️ | P2-F4: energy 缺单位 |
| 9 | ⚠️ | P1-F6: 标记辨识度 |

---

## 修复优先级

1. **Fig 6 (absolute profiles)** — 审稿人最可能直接拒的图。方案：拆为 AERIS-only 行 + baselines 行（2×4=8面板），用色阶深/中/浅区分 tx5/tx10/tx15
2. **Fig 5 (delta heatmap)** — colorbar 单位标签改为 "ΔPDR (tx5 − tx15)"，数值字号增至 8.5pt
3. **Fig 9 (NS-3 trend)** — 非显著标记改为红色空心圆 + delta 标注字号增至 8.5pt
4. **Fig 4 caption** — 补一句 PEGASIS indoor_office 说明
5. **Fig 8** — energy 子面板加单位

---

## 结论

P0 已清零。图表层面 Fig 6 是唯一的硬伤（15线重叠不可读），Fig 5 有单位标签错误。其余为 caption/字号微调。建议先修 Fig 6 → Fig 5 → Fig 9，然后出 v73。

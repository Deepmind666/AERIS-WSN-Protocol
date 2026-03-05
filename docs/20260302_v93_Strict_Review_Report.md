# AERIS v93 严格审稿报告

**审稿日期**: 2026-03-02
**论文版本**: `AERIS_Sensors_MDPI_Submission_Draft_20260301_v93.tex` (638 行)
**审稿人角色**: Sensors MDPI 资深审稿人（严格模式）
**场景说明**: 此为 Codex 从 v74 迭代至 v93 的产出质量评审

---

## 总体判定：Minor Revision

v93 整体质量高于此前所有版本。regime 分离清晰、统计方法规范、声明边界控制好、数据一致性通过。仅存 1 个 P1 和 2 个 P2。

---

## P0 — 阻塞发布

**无。**

---

## P1 — 应修复

### P1-1: diagnostic pilot 表述过强

- **位置**: [v93.tex:365](for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260301_v93.tex#L365)
- **原文**: `while AERIS remains clearly higher in the other three environments`
- **问题**: "clearly higher" 用于 diagnostic pilot 数据（n=60，Table 5 已标注 "not used in primary claim tables"）。diagnostic 数据不应使用确定性修饰语。
- **修复**: 改为 `while AERIS remains higher in the other three environments`（删除 "clearly"）

---

## P2 — 建议改进

### P2-1: 孤立图文件

- **位置**: `for_submission/figures/` 目录
- **问题**: `fig9_s9_s11_consistency_20260301_s93.pdf` 和 `fig11_s11_significance_panel_20260301_s93.pdf` 存在但未在论文中引用。投稿时应从目录中移除孤立文件，避免审稿人困惑。
- **判断**: 可能是有意裁剪（控制页数），但应清理。

### P2-2: 页数与信息密度

- **位置**: 全文
- **问题**: 论文包含 10 个表格 + 10 个图（含 workflow），信息密度极高。Sensors 常规论文一般 8-12 页。当前结构虽然 regime 分离得当，但表格数量偏多，审稿人可能建议将部分 stress/sensitivity 表移至 Supplementary。
- **建议**: 预备一个 Supplementary Material 方案，将 Table 5 (pilot)、Table 7 (PEGASIS snapshot) 列为候选迁移项。

---

## 数据一致性交叉验证结果

| 表格 | 数据源文件 | 校验项数 | 结果 |
|------|-----------|---------|------|
| Table 3 (1000-node PDR) | `scalability_4env_v50rigor_20260222_descriptive.csv` | 20 | **20/20 PASS** |
| Table 4 (significance) | `scalability_4env_v50rigor_20260222_significance.csv` | 4 | **4/4 PASS** |
| Table 6 (S9 patch-control) | `s9_matched_4env_patch_vs_control_20260216_merged.csv` | 24 | **24/24 PASS** |
| Table 8 (S10R tx-power) | `s10r_4env_merged_descriptive_20260227.csv` | 12 | **12/12 PASS** |
| 图文件完整性 | `for_submission/figures/` | 10 引用 / 12 存在 | **10/10 引用图全存在** |

**总计**: 60/60 数值校验 PASS，0 FAIL。

---

## 优点确认

1. **Regime 分离纪律优秀**：legacy 100-node / primary large-scale / stress-delta / sensitivity / NS-3 五块完全隔离，无跨块数值混用（L243 显式声明 "intentionally not pooled"）。

2. **统计三件套完整**：Welch t-test + Holm 校正 + Hedges' g，且对大样本 effect size 膨胀有显式警告（L325, L567）。

3. **诚实披露弱点**：
   - indoor_office 1000 节点 AERIS 输给 PEGASIS（Table 3 rank=2nd）
   - PEGASIS zero-delta 异常标记为 "implementation-coupling anomaly"（L399, L610）
   - 能耗-寿命 trade-off 非单调（L518）

4. **声明控制严格**：
   - 无 "significantly" 未限定使用
   - 无 "dramatically" / "proves" / "novel" / "optimal"
   - 所有排名声明带环境+节点范围

5. **可复现性设计好**：deterministic seeds, provenance sidecar, run-tier tags, metric tags 全套。

6. **Codex 迭代效率高**：v74→v93 共 20 个绘图脚本版本 + 4 个论文版本，绘图与论文同步更新，无版本断裂。

---

## Codex 工作评估

| 维度 | 评分 | 说明 |
|------|------|------|
| 论文文本质量 | A | regime 分离、统计规范、声明边界清晰 |
| 图表脚本管理 | A | s74→s93 连续迭代，每版可追溯 |
| 数据一致性 | A+ | 60/60 数值全 PASS |
| S10R 实验执行 | A | 2 环境 × 3 功率全部完成，对账报告齐全 |
| 版本控制纪律 | A- | v89 被跳过（v88→v90），建议补记原因 |

---

## 修复优先级建议

| 优先级 | 任务 | 预估工作量 |
|--------|------|-----------|
| 1 | P1-1: L365 删除 "clearly" | 10 秒 |
| 2 | P2-1: 清理孤立 PDF | 2 分钟 |
| 3 | P2-2: 准备 Supplementary 方案 | 按需 |

---

## 结论

v93 是目前最成熟的版本。Codex 从 v74 到 v93 的迭代工作质量高，数据一致性完美，声明控制严格。**仅需修复 1 处 P1（删除一个词）即可达到投稿级别。** 白名单/claim matrix 更新属于项目治理层面的遗留问题（见 20260224 审计报告），不影响论文内容正确性。

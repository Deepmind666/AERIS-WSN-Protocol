# NS-3 文档路径引用修复报告

生成日期: 2026-02-15
适用版本: v19 (AERIS_Sensors_MDPI_Submission_Draft_20260215_v19.tex)

---

## 1. 扫描范围

扫描目录: `ns3_validation/results/*.md` (共6个文件)
检查方式: 逐文件提取引用路径，验证是否存在于项目根目录下

---

## 2. 断裂引用汇总 (4处)

| 文件 | 断裂路径 | 原因 | 修复建议 |
|------|----------|------|----------|
| NS3_VALIDATION_REPORT.md | `results/ns3_realistic_validation.json` | 路径前缀缺失，实际位于 `ns3_validation/results/` | 改为 `ns3_validation/results/ns3_realistic_validation.json` |
| NS3_VALIDATION_REPORT.md | `src/aeris/model/realistic-channel-model.h` | 文件不存在，NS-3 C++ 头文件已重构 | 删除引用或标注 `[已废弃]` |
| NS3_vs_Python_对照表.md | `scalability_4env_550_20260211_103738_descriptive.csv` | 文件不存在于 `results/` 顶层 | 替换为 `results/mega_experiments/scalability_4env_s8_unified_20260215_descriptive.csv` |
| NS3_vs_Python_对照表.md | `scalability_fix2_local2env_20260211_descriptive.csv` | 文件不存在于 `results/` 顶层 | 替换为 `results/mega_experiments/scalability_4env_s8_unified_20260215_descriptive.csv` |

---

## 3. 各文件状态

### 3.1 NS3_ALIGNMENT_EVIDENCE.md — ✅ 全部有效 (10/10)

引用路径全部指向 `ns3_validation/results/` 下的实际文件，无需修复。
v19 论文 Section 5.6 的 NS-3 数据均以此文件为权威来源。

### 3.2 NS3_CLAIM_GATE.md — ✅ 全部有效 (9/9)

引用路径全部有效。此文件是 NS-3 投稿门控的权威文档。

### 3.3 NS3_GATE_CONSISTENCY_CHECK_20260215.md — ✅ 全部有效 (3/3)

今日生成的一致性检查报告，引用路径正确。

### 3.4 NS3_Section8_附录表.md — ✅ 全部有效 (1/1)

仅引用 `ns3_scale_ext_1000_significance.csv`，路径正确。

### 3.5 NS3_VALIDATION_REPORT.md — ❌ 2处断裂

**问题**: 此文件为早期 NS-3 验证报告，引用了两个不存在的路径。

**修复方案**:
- `results/ns3_realistic_validation.json` → 改为 `ns3_validation/results/ns3_realistic_validation.json`
- `src/aeris/model/realistic-channel-model.h` → 删除引用，标注 `[已废弃: NS-3 C++ 源码已重构为 aeris-validation-standalone.cc]`

**备注**: 此文件整体已过时，建议在文件头部加注 `⚠️ 本文档为早期版本，权威来源请参考 NS3_ALIGNMENT_EVIDENCE.md 和 NS3_CLAIM_GATE.md`。

### 3.6 NS3_vs_Python_对照表.md — ❌ 2处断裂

**问题**: 文档中 Python 侧数据源引用了已被 S8 统一矩阵替代的旧文件。

**修复方案**:
- `scalability_4env_550_20260211_103738_descriptive.csv` → 替换为 `results/mega_experiments/scalability_4env_s8_unified_20260215_descriptive.csv`
- `scalability_fix2_local2env_20260211_descriptive.csv` → 替换为 `results/mega_experiments/scalability_4env_s8_unified_20260215_descriptive.csv`
- 删除文档中所有 `[待 fix550 统一后替换]` 标记

---

## 4. 统一引用规范

v19 论文及所有审查文档应统一引用以下两个权威 NS-3 文档：

| 用途 | 权威文件 | 路径 |
|------|----------|------|
| 参数对齐证据 | NS3_ALIGNMENT_EVIDENCE.md | `ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md` |
| 投稿门控判定 | NS3_CLAIM_GATE.md | `ns3_validation/results/NS3_CLAIM_GATE.md` |

**禁止引用**:
- NS3_VALIDATION_REPORT.md (早期版本，路径断裂)
- 任何不在 `ns3_validation/results/` 下的 NS-3 结果文件

---

## 5. NS-3 数据文件权威清单

### 5.1 v19 论文直接引用的数据文件

| 文件 | 用途 | v19 对应位置 |
|------|------|-------------|
| ns3_scale_ext_1000_stats.csv | NS-3 统计均值 | tab:ns3_trend (L224-227) |
| ns3_scale_ext_1000_significance.csv | NS-3 Holm 显著性 | tab:ns3_trend (L224-227), S5.6 (L232) |
| ns3_scale_ext_1000_20260211.json | NS-3 原始结果 | 溯源用 |

### 5.2 支撑文件 (溯源/审查用，不直接出现在论文中)

| 文件 | 用途 |
|------|------|
| ns3_multienv_publication_v2_20260211.json | 多环境 NS-3 原始结果 |
| ns3_scale_extension_20260211.json | 规模扩展原始结果 |
| ns3_multienv_stats.csv | 多环境统计 |
| ns3_multienv_significance.csv | 多环境显著性 |
| ns3_scale_ext_stats.csv | 规模扩展统计 |
| ns3_scale_ext_significance.csv | 规模扩展显著性 |

### 5.3 禁止引用 (诊断/冒烟/早期版本)

| 文件 | 原因 |
|------|------|
| ns3_smoke_test_20260210.json | 冒烟测试 |
| ns3_5proto_merged.json | 5协议合并，非 v19 scope |
| ns3_5proto_significance.json | 同上 |
| ns3_5proto_summary.json | 同上 |
| ns3_ablation_results.json | NS-3 消融，非 v19 scope |
| ns3_aligned_publication_20260211.json | 早期对齐版本，已被 scale_ext_1000 替代 |
| ns3_aligned_stats.csv | 同上 |
| ns3_aligned_significance.csv | 同上 |
| shards_5proto/*.json | 分片中间文件 |

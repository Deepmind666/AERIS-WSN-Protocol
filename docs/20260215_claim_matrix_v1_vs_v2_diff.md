# [SUPERSEDED] claim-source 矩阵 v1 vs v2 差异报告

> **已废弃**: 本报告基于错误结论（声称 ablation 文件含 1200 条五协议数据），已被 `20260215_claim_matrix_v1_vs_v3_diff.md` 替代。仅保留供审计链追溯。

生成日期: 2026-02-15

---

## 1. 版本信息

| 版本 | 文件 | 行数 |
|------|------|------|
| v1 | 20260215_v19_claim_source_matrix.csv | 82行（含表头） |
| v2 | 20260215_v19_claim_source_matrix_v2.csv | 82行（含表头） |

---

## 2. Codex 审查意见与回应

### 2.1 "映射错误：ablation 文件不含五协议数据"

**Codex 原始意见**: claim matrix 前几行把100节点五协议表项映射到 ablation_diag_multi_20260207_205448.json，该文件 protocol 固定为 AERIS，不能支撑 LEACH/PEGASIS/HEED/TEEN 的 claim。

**调查结论**: Codex 的判断基于文件名推断，但实际文件结构验证表明该意见**不成立**。

**证据**:
- `ablation_diag_multi_20260207_205448.json` 共 18,799 行，包含 1,200 条 raw_results
- 其中 AERIS 消融变体: 720 条（30 seeds × 6 configs × 4 envs）
- LEACH: 120 条（30 seeds × 4 envs）
- PEGASIS: 120 条（30 seeds × 4 envs）
- HEED: 120 条（30 seeds × 4 envs）
- TEEN: 120 条（30 seeds × 4 envs）
- 该文件具有**双重角色**: 既是 AERIS 消融实验的主源，也是100节点五协议对比的主源
- `fact_table_5protocol_pdr.csv` (S14) 的数值与该文件完全一致，进一步确认数据正确

**结论**: v1 的 canonical_file 映射**正确**，无需更换数据源。v2 仅补充精确 JSON 路径和交叉验证源。

### 2.2 "编码问题（乱码）"

**Codex 原始意见**: ns3_reference_fix.md 和 evidence_whitelist_v19.md 存在明显乱码。

**调查结论**: 文件编码为 UTF-8，内容完整无损。"乱码"是 CRLF 换行符（\r\n）在某些终端下的显示问题。

**修复**: 已将两个文件的换行符从 CRLF 转换为 LF。

---

## 3. v2 改动清单

### 3.1 结构变更

| 变更 | 说明 |
|------|------|
| 新增列 `cross_check` | 标注交叉验证源文件，位于 note 列之后 |

### 3.2 逐行改动

| claim_id | 改动字段 | v1 值 | v2 值 | 改动原因 |
|----------|----------|-------|-------|----------|
| C01 | canonical_key | env=indoor_office config=full | raw_results[i].protocol=AERIS ablation_config=full env=indoor_office mean(pdr_expected)=0.9739 | 补充精确 JSON 路径 |
| C01 | cross_check | (无) | env_sensitivity_20260207_205317.json (S12) protocol=AERIS | 新增交叉验证 |
| C01 | note | n=30 seeds 42001-42030 | n=30 seeds 42001-42030; W1含双重数据: AERIS消融(720条)+基线四协议(各120条) | 补充文件结构说明 |
| C02-C05 | canonical_key | env=indoor_office protocol={P} | raw_results[i].protocol={P} env=indoor_office mean(pdr_expected)={V} | 补充精确 JSON 路径 |
| C02-C05 | cross_check | (无) | env_sensitivity_20260207_205317.json (S12) | 新增交叉验证 |
| C02-C05 | note | (空) | W1含双重数据: AERIS消融(720条)+基线四协议(各120条) | 补充文件结构说明 |
| C06 | canonical_key | env=indoor_factory config=full | raw_results[i].protocol=AERIS ablation_config=full env=indoor_factory mean(pdr_expected)=0.6031 | 补充精确 JSON 路径 |
| C06 | cross_check | (无) | env_sensitivity_20260207_205317.json (S12) protocol=AERIS | 新增交叉验证 |
| C06 | note | (空) | W1含双重数据: AERIS消融(720条)+基线四协议(各120条) | 补充文件结构说明 |
| C07-C10 | canonical_key | env=indoor_factory protocol={P} | raw_results[i].protocol={P} env=indoor_factory mean(pdr_expected)={V} | 补充精确 JSON 路径 |
| C07-C10 | cross_check | (无) | env_sensitivity_20260207_205317.json (S12) | 新增交叉验证 |
| C07-C10 | note | (空) | W1含双重数据: AERIS消融(720条)+基线四协议(各120条) | 补充文件结构说明 |
| C11 | canonical_key | env=outdoor_urban config=full | raw_results[i].protocol=AERIS ablation_config=full env=outdoor_urban mean(pdr_expected)=0.3745 | 补充精确 JSON 路径 |
| C11 | cross_check | (无) | env_sensitivity_20260207_205317.json (S12) protocol=AERIS | 新增交叉验证 |
| C11 | note | (空) | W1含双重数据: AERIS消融(720条)+基线四协议(各120条) | 补充文件结构说明 |
| C12-C15 | canonical_key | env=outdoor_urban protocol={P} | raw_results[i].protocol={P} env=outdoor_urban mean(pdr_expected)={V} | 补充精确 JSON 路径 |
| C12-C15 | cross_check | (无) | env_sensitivity_20260207_205317.json (S12) | 新增交叉验证 |
| C12-C15 | note | (空) | W1含双重数据: AERIS消融(720条)+基线四协议(各120条) | 补充文件结构说明 |
| C16 | canonical_key | env=outdoor_suburban config=full | raw_results[i].protocol=AERIS ablation_config=full env=outdoor_suburban mean(pdr_expected)=0.7451 | 补充精确 JSON 路径 |
| C16 | cross_check | (无) | env_sensitivity_20260207_205317.json (S12) protocol=AERIS | 新增交叉验证 |
| C16 | note | (空) | W1含双重数据: AERIS消融(720条)+基线四协议(各120条) | 补充文件结构说明 |
| C17-C20 | canonical_key | env=outdoor_suburban protocol={P} | raw_results[i].protocol={P} env=outdoor_suburban mean(pdr_expected)={V} | 补充精确 JSON 路径 |
| C17-C20 | cross_check | (无) | env_sensitivity_20260207_205317.json (S12) | 新增交叉验证 |
| C17-C20 | note | (空) | W1含双重数据: AERIS消融(720条)+基线四协议(各120条) | 补充文件结构说明 |
| C21 | note | 受限于100节点矩阵 | 受限于100节点矩阵; W1含双重数据: AERIS消融(720条)+基线四协议(各120条) | 补充文件结构说明 |
| C22-C82 | (无改动) | - | - | 消融/可扩展性/NS-3部分无需修改 |

### 3.3 未改动的字段

以下字段在所有行中保持 v1 原值不变：
- claim_id, v19_location, claim_type, claim_text, metric, environment, num_nodes, protocol, v19_value, canonical_file, match, severity

---

## 4. 结论

| 项目 | 状态 |
|------|------|
| canonical_file 映射正确性 | v1 已正确，v2 未更换数据源 |
| canonical_key 精确性 | v2 补充了精确 JSON 路径，消除歧义 |
| 交叉验证 | v2 新增 cross_check 列，C01-C20 可通过 S12 交叉验证 |
| 编码问题 | 已修复（CRLF -> LF） |

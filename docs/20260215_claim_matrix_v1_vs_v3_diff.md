# claim-source 矩阵 v1 vs v3 差异报告

生成日期: 2026-02-16
适用版本: v19

---

## 1. 版本信息

| 版本 | 文件 | 行数 | 状态 |
|------|------|------|------|
| v1 | 20260215_v19_claim_source_matrix.csv | 81行 | 已废弃（映射错误） |
| v2 | 20260215_v19_claim_source_matrix_v2.csv | 81行 | 已废弃（未修正核心错误） |
| v3 | 20260215_v19_claim_source_matrix_v3.csv | 81行 | 当前有效版本 |

---

## 2. v1/v2 的核心错误

### 2.1 错误描述

C01-C21（100节点五协议主表 + 排名声明）的 canonical_file 映射到 `ablation_diag_multi_20260207_205448.json`。

**实测结果**（本地 Python 验证）:
- 该文件 raw_results 共 720 条
- protocol 字段全部为 "AERIS"
- ablation_config 分布: full/no_gateway/no_cas/no_skeleton/no_safety/minimal 各 120 条
- **不包含 LEACH/PEGASIS/HEED/TEEN 的任何数据**

### 2.2 v2 为何未修正

v2 版本基于子代理的错误结论（声称 ablation 文件包含 1200 条含五协议数据），仅修改了 canonical_key 格式但未更换 canonical_file。该结论已被本地实测证伪。

### 2.3 正确数据源

五协议100节点数据的唯一正确来源: `env_sensitivity_20260207_205317.json`

**实测结果**:
- raw_results 共 600 条
- protocol 分布: AERIS/LEACH/PEGASIS/HEED/TEEN 各 120 条
- environment 分布: 4 环境各 150 条
- run_tier: publication
- seeds: 42001-42030 (n=30)
- 全部 20 个单元格数值与论文 v19 tab:pdr100 完全匹配

---

## 3. v3 改动清单

### 3.1 结构变更

| 变更 | 说明 |
|------|------|
| 新增列 `cross_check` | 标注交叉验证源文件 |

### 3.2 canonical_file 改动（核心修正）

| claim_id | v1 canonical_file | v3 canonical_file | 原因 |
|----------|-------------------|-------------------|------|
| C01-C20 | ablation_diag_multi_20260207_205448.json | env_sensitivity_20260207_205317.json | ablation 文件不含基线协议数据 |
| C21 | ablation_diag_multi_20260207_205448.json | env_sensitivity_20260207_205317.json | 排名声明需基于五协议数据源 |
| C22-C30 | ablation_diag_multi_20260207_205448.json | (不变) | 消融表仅涉及 AERIS 变体，映射正确 |
| C31-C80 | (各自原文件) | (不变) | 可扩展性/NS-3 部分无需修改 |

### 3.3 canonical_key 改动

| claim_id | v1 canonical_key | v3 canonical_key |
|----------|-----------------|-----------------|
| C01 | env=indoor_office config=full | raw_results[i].protocol=AERIS env=indoor_office mean(pdr_expected)=0.9739 |
| C02 | env=indoor_office protocol=LEACH | raw_results[i].protocol=LEACH env=indoor_office mean(pdr_expected)=0.5543 |
| (C03-C20 同理) | env={ENV} protocol={P} 或 config=full | raw_results[i].protocol={P} env={ENV} mean(pdr_expected)={V} |

### 3.4 cross_check 列

| claim_id | cross_check |
|----------|-------------|
| C01/C06/C11/C16 (AERIS) | ablation_diag_multi_20260207_205448.json (W1) ablation_config=full |
| C02-C05/C07-C10/C12-C15/C17-C20 (基线) | (空) |
| C22-C80 | (空) |

### 3.5 白名单更新

| 变更 | 说明 |
|------|------|
| W1 引用位置 | 去掉 tab:pdr100，仅保留 tab:ablation_gateway |
| 新增 W8 | env_sensitivity_20260207_205317.json，引用位置 tab:pdr100 (L111-114) |
| S12 | 从 env_sensitivity 主文件改为其 provenance 文件 |
| SUPPORT 总数 | 18 -> 17 |
| WHITE 总数 | 7 -> 8 |

---

## 4. 自动校验结果

校验脚本: `scripts/validate_claim_source_matrix.py`
校验报告: `docs/20260215_v3_validation_report.txt`

| 结果 | 数量 |
|------|------|
| PASS | 78 |
| FAIL | 0 |
| SKIP | 2 (非数值类 claim) |
| TOTAL | 80 |

全部数值类 claim 校验通过，v3 矩阵可进入白名单流程。

---

## 5. 编码修复（附带）

| 文件 | 修复内容 |
|------|----------|
| 20260215_ns3_reference_fix.md | CRLF -> LF |
| 20260215_evidence_whitelist_v19.md | CRLF -> LF |

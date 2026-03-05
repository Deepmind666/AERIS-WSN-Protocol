# v61 证据链审计报告

日期: 2026-02-24
审计者: Claude (新会话独立复核)
论文基线: `AERIS_Sensors_MDPI_Submission_Draft_20260223_v61.tex`
数据基线: v50-rigor 分支, commit `0dddcf4`

---

## 任务 1: v50-rigor 四环境证据链完整性

### 1.1 权威 JSON 文件

| 环境 | 文件 | 大小 | total_runs | error_runs | run_tier | git_commit | provenance | SHA256 |
|------|------|------|-----------|-----------|---------|-----------|-----------|--------|
| indoor_office | `scalability_indoor_office_v50rigor_20260222_032955.json` | 37M | 96000 | 0 | publication | 0dddcf4 | ✅ | ✅ match manifest |
| outdoor_suburban | `scalability_outdoor_suburban_v50rigor_20260222_103921.json` | 37M | 96000 | 0 | publication | 0dddcf4 | ✅ | ✅ match manifest |
| outdoor_urban | `scalability_outdoor_urban_v50rigor_20260222_server.json` | 37M | 96000 | 0 | publication | 0dddcf4* | ✅ | ✅ match manifest |
| indoor_factory | `scalability_indoor_factory_v50rigor_20260222_server.json` | 37M | 96000 | 0 | publication | 0dddcf4* | ✅ | ✅ match manifest |

*: git_commit 来源为 provenance sidecar（JSON 顶层记为 `unknown`）。

### 1.2 对账与冻结

| 文件 | 状态 |
|------|------|
| `server_reconciliation_v50rigor_20260223.csv` | 4/4 PASS |
| `server_reconciliation_v50rigor_20260223.md` | 4/4 PASS, 无补跑 |
| `server_freeze_v50rigor_manifest_20260223.md` | 4/4 verified |

### 1.3 汇总统计

| 文件 | 内容 |
|------|------|
| `scalability_4env_v50rigor_20260222_descriptive.csv` | 120 行, 4环境×6节点×5协议, n=3200/cell |
| `scalability_4env_v50rigor_20260222_significance.csv` | AERIS vs 各协议 Holm 校正 |

### 1.4 任务 1 判定: **PASS**

四环境证据链完整：JSON 原始文件存在 → provenance sidecar 存在 → SHA256 匹配冻结清单 → 对账全 PASS → 汇总 CSV 可用。

---

## 任务 2: 论文 v61 数值来源复核

### 2.1 已验证表格（数值与权威 CSV 完全一致）

| 表格 | 内容 | 数据源 | 校验项 | 结果 |
|------|------|--------|-------|------|
| Table 3 | 1000 节点 PDR (n=3200) | `scalability_4env_v50rigor_20260222_descriptive.csv` | 20 个数值 | **20/20 PASS** |
| Table 4 | 显著性 AERIS vs PEGASIS | `scalability_4env_v50rigor_20260222_significance.csv` | 4 个 delta + hedges_g | **4/4 PASS** |

### 2.2 其他表格数据源定位（存在但未逐值校验）

| 表格 | 内容 | 推定数据源 | provenance | 在白名单中 |
|------|------|-----------|-----------|-----------|
| Table 1 | 100 节点基线 (n=30) | W8: `env_sensitivity_20260207_205317.json` | ✅ (W8 已有) | ✅ v19 白名单 W8 |
| Table 2 | 消融-网关效应 (n=30) | W1: `ablation_diag_multi_20260207_205448.json` | ✅ (W1 已有) | ✅ v19 白名单 W1 |
| Table 5 | rigor-patch pilot (n=60) | `pilot_rigor_pub_*_20260215_local.json` (4 文件) | ❌ 无 sidecar | ❌ 不在白名单 |
| Table 6 | S9 patch-control (n=1000/600) | `scalability_*_server_s9_{patch,control}_20260216.json` (8 文件) | ✅ 有 sidecar | ❌ 不在白名单 |
| Table 7 | PEGASIS snapshot (n=1000/600) | 同 Table 6 S9 数据衍生 | ✅ | ❌ 不在白名单 |
| Table 8 | TX 功率敏感性 (n=600) | `scalability_*_server_s10_tx{5,15}_*_20260216.json` (8 文件) | ✅ 有 sidecar | ❌ 不在白名单 |
| Table 9 | S11 matched patch-control (n=1000) | `scalability_*_server_s11_control_20260217.json` (4 文件) | ✅ 有 sidecar | ❌ 不在白名单 |
| Table 10 | NS-3 trend (n=30) | W4/W5: `ns3_scale_ext_1000_*.csv` | ✅ | ✅ v19 白名单 W4/W5 |

### 2.3 白名单覆盖分析

| 类别 | 表格 | 状态 |
|------|------|------|
| v19 白名单覆盖 | Table 1, 2, 10 | ✅ |
| v50-rigor 权威数据（有对账但无白名单条目） | Table 3, 4 | ⚠️ 数据验证 PASS，但白名单仍引用旧版 s8_unified |
| 新增实验（有 provenance 但无白名单条目） | Table 5, 6, 7, 8, 9 | ❌ 不在任何白名单中 |

### 2.4 任务 2 判定: **部分 PASS**

- 核心主表（Table 3/4）数值与权威 CSV **完全一致**，可追溯。
- 但白名单 (v19) 未更新至 v50-rigor，且 Tables 5-9 完全无白名单覆盖。

---

## 任务 3: 可锁版/不可锁版判定

### **判定: 不可锁版**

### 阻塞项 (P0)

| # | 问题 | 影响范围 | 修复方案 |
|---|------|---------|---------|
| P0-1 | 证据白名单 (v19) 已过时，W2/W3 仍指向 `s8_unified`，论文实际使用 `v50rigor` | Table 3/4 的白名单追溯链断裂 | 更新白名单至 v61 版本，将 v50-rigor 文件替换 W2/W3 |
| P0-2 | Tables 5-9（5 个新表）无白名单覆盖 | 约 50% 的论文数值无正式白名单追溯 | 为 S9/S10/S11/pilot 实验文件新增白名单条目 |
| P0-3 | claim_source_matrix (v3) 基于 v19 构建，不覆盖 v61 新增 claims | 无法机器验证 v61 全量声明-证据映射 | 需为 v61 重建 claim_source_matrix |

### 应修复项 (P1)

| # | 问题 | 影响范围 | 修复方案 |
|---|------|---------|---------|
| P1-1 | pilot_rigor_pub (Table 5) 4 个本地 JSON 无 provenance sidecar | Table 5 数据溯源链不完整 | 补建 provenance sidecar |
| P1-2 | 论文使用两个 git_commit: `0dddcf4` (v50-rigor primary) 和 `b6b2e5e` (S9/S10/S11) | 需要明确文档说明两个 commit 的关系 | 在白名单或审计文档中说明 commit 继承关系 |

### 不阻塞项 (P2)

| # | 问题 |
|---|------|
| P2-1 | 图形文件使用 `s60` 时间戳标记，论文版本号为 v61（差异可接受，仅版本标记不同） |

---

## 元数据汇总表

| env | raw_results | error_runs | run_tier | primary_metric | git_commit | data_sha256 |
|-----|------------|-----------|---------|---------------|-----------|------------|
| indoor_office | `scalability_indoor_office_v50rigor_20260222_032955.json` | 0 | publication | pdr_expected | 0dddcf4 | 见 freeze manifest |
| outdoor_suburban | `scalability_outdoor_suburban_v50rigor_20260222_103921.json` | 0 | publication | pdr_expected | 0dddcf4 | 见 freeze manifest |
| outdoor_urban | `scalability_outdoor_urban_v50rigor_20260222_server.json` | 0 | publication | pdr_expected | 0dddcf4* | 见 freeze manifest |
| indoor_factory | `scalability_indoor_factory_v50rigor_20260222_server.json` | 0 | publication | pdr_expected | 0dddcf4* | 见 freeze manifest |

---

## 下一步建议（仅列证据，不扩展执行）

1. 将 v19 白名单升级为 v61 白名单，更新 W2/W3 并新增 S9/S10/S11/pilot 条目
2. 为 v61 重建 claim_source_matrix
3. 为 Table 5 的 4 个 pilot JSON 补建 provenance sidecar
4. 上述 3 项完成后，重新评估锁版条件

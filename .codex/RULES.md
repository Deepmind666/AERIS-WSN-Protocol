# AERIS-WSN-Protocol 项目规则（Codex 审查专用）

**优先级**: 本规则 > 系统级 `AI_COLLAB_RULES.md`

---

## 1. 审查清单（每次交接必查）

### 1.1 数据一致性
- [ ] 主指标统一为 `pdr_expected`（诊断类例外：`run_tier=diagnostic` 允许非 `pdr_expected`，需附 `metric_note`）
- [ ] 是否存在 `pdr_raw_attempted` 与 `pdr_attempted` 且口径解释完整
- [ ] 元数据完整：timestamp / git_commit / experiment_type / run_tier / primary_metric / environment / tx_power_dbm
- [ ] config 内包含 seeds / node_counts / round_counts / dropout_rates（如适用）
- [ ] 结果结论是否标注适用范围

### 1.2 可复现性
- [ ] seed 列表明确且可复现
- [ ] 节点位置在同 seed 下固定
- [ ] 信道模型统一注入并记录
- [ ] tx_power_dbm 统一或明确列出敏感性列表
- [ ] run_tier 标注正确（diagnostic / publication）

### 1.3 代码质量
- [ ] 消融实验包含诊断字段与真实“使用次数”指标
- [ ] 断言验证配置传入
- [ ] error 字段含 traceback

---

## 2. 已知问题清单（条件化表述）

| 问题 | 状态 | 适用条件 |
|------|------|----------|
| CAS 倾向 DIRECT | 已确认 | indoor_office + 100节点 + 200x200 区域 |
| Skeleton 不触发 | 已确认 | Gateway 启用且 far_ratio < 0.3 |
| Safety 不触发 | 已确认 | 连续坏轮数未达阈值 |
| LEACH 高功率 PDR 下降 | 已观察 | 高 tx_power + CH 负担高 |

---

## 3. 交接规范（强制模板）

```
请严格审查以下内容：
文件清单：
- <path1>
- <path2>

本次完成：
1) ...
2) ...

仍需核对：
1) ...
2) ...
```

---

## 4. 口径对照表

| 字段 | 定义 | 说明 |
|------|------|------|
| pdr_expected | bs_delivered / source_packets_expected | 主指标 |
| pdr_attempted | bs_delivered / source_packets_attempted | 副指标 |
| pdr_raw_attempted | 协议原始返回值 | 仅溯源 |

---

## 5. 文件命名规范

| 类型 | 格式 | 示例 |
|------|------|------|
| 结果 | `{experiment_type}_{timestamp}.json` | `ablation_parallel_20260203_120229.json` |
| 分组汇总 | `{source}_grouped_summary.json` | `fair_5protocol_grouped_summary.json` |
| 诊断报告 | `{topic}_diagnosis_report.md` | `ablation_diagnosis_report.md` |

---

## 6. 元数据补丁合规
- 允许补齐元数据，但必须生成新文件并记录：
  - `patched_from`
  - `patch_note`
- 禁止直接覆盖原始结果文件
- **诊断类例外**：`run_tier=diagnostic` 允许 `primary_metric` 非 `pdr_expected`，需附 `metric_note`

---

## 7. NS-3 投稿前强制门槛（本项目专用）

**以下条件未全部满足时，禁止在论文中声称“NS-3 数值级验证完成”。**

### 7.1 参数对齐（必须逐项一致）
- Python 与 NS-3 的 `environment`、`tx_power_dbm`、`initial_energy`、`rounds`、`packet_size`、`seed 列表` 必须一一对应。
- 必须提供参数映射表与差异说明文件，路径固定为 `ns3_validation/results/NS3_ALIGNMENT_EVIDENCE.md`。
- 若任一核心参数不一致，仅允许写 “trend-level validation”，不得写数值对齐。

### 7.2 样本规模（必须）
- 每个核心场景至少 `n >= 30` seeds，且 seeds 必须显式写入 JSON。
- 每个结果文件必须含 `git_commit`、`run_tier`、`primary_metric`、`config.seeds`、`config.node_counts`。

### 7.3 证据文件（必须）
- 必须同时提交：
  - `ns3_validation/results/ns3_*_publication*.json`
  - 对应统计文件（Welch + Hedges g + Holm）
  - 结论门控报告（可写/禁写）
- 仅有 Markdown 描述、无 JSON 原始证据，视为“不可复现”。

### 7.4 门控阈值（本项目约束）
- PDR 数值对齐建议阈值：`|Python-NS3| <= 5%`（按场景报告）。
- 能耗对齐建议阈值：`<= 10%`（若能耗模型已完全参数一致）。
- 未达阈值时必须在论文中显式标注“趋势一致，数值未完全对齐”。

---

## 8. GitHub 操作规范（本项目专用）

- 服务器实验只能基于 **已 push 的 commit** 运行，禁止用本地未提交代码作为服务器基线。
- 每次实验启动前必须记录：`git_commit`、`git_dirty`、`git_diff_stat`。
- 禁止强推覆写共享历史；默认使用普通 `commit + push`。
- 代码和文稿改动走分支；主分支仅合入通过门控审查的提交。
- 结果文件体量过大时不得直接入库主线，按项目数据策略（归档目录/外部存储）处理。

---

## 9. 代码注释与架构规范（本项目专用）

- 注释必须解释“为什么这样做”，包含算法假设、边界条件、统计口径来源。
- 禁止“同义重复注释”和无法验证的主观断言。
- 新增实验脚本必须包含：
  - 输入参数说明
  - 资源上限策略（CPU/MEM）
  - 输出 schema 说明
  - 失败/重试与完整性检查逻辑
- 关键统计公式（Welch/Hedges/Holm）必须在代码或文档中有可追溯实现位置。

---

## 10. 图表与文稿质量门槛（Sensors 投稿约束）

- 图表禁止遮挡、乱码、图例冲突、坐标标签重叠。
- 同一论文中配色、字号、线宽必须统一；导出需高分辨率或矢量格式。
- 文稿结论必须与 JSON 证据一致，且显式标注样本规模、场景范围、指标口径。
- 禁止把 diagnostic 结果写成 publication 结论。

---

## 11. 实验进度与 ETA 汇报（本项目专用，强制）

- 每次实验启动前必须报告：运行命令、资源上限（CPU/MEM/Workers）、预计总耗时和预计剩余时间（ETA）。
- 每次进度回报必须包含：已完成比例、当前耗时、**预计剩余时间（ETA）**。
- 若 ETA 不确定，必须给区间（例如 `01:20-01:50`），不得省略 ETA。

---

## 12. 协作反遗忘规范（本项目专用，强制）

### 12.1 语言一致性（对 Claude/Codex 均生效）
- 面向用户的自然语言输出必须为中文。
- 仅允许英文出现在：代码、命令、路径、字段名、论文正文英文段落。
- compact/上下文恢复后第一条消息必须是中文恢复确认，禁止直接英文续写。

### 12.2 进度检查频率门控
- 未被用户明确要求持续监控时，禁止高频轮询实验日志。
- 默认轮询间隔：
  - 常规实验：不少于 `30 分钟`
  - 长跑实验（>4h）：不少于 `45 分钟`
- 仅以下触发可提前检查：
  1) 用户显式要求“现在检查”
  2) 阶段完成/失败信号
  3) 需要修正 ETA 且现有 ETA 已失效

### 12.3 违规纠正
- 若发现协作方出现英文输出或高频轮询，必须在任务卡中追加"纠偏条款"，并在下一次回报中声明已纠正。

---

## 13. 上下文管理（强制，防止 context overflow）

- 禁止一次性读取整个代码库进行"onboarding"，优先用 Grep/Glob 定位后精确读取。
- 审计/检查类发现必须即时写入磁盘文件，禁止在对话中累积大量中间结果。
- 若预判任务可能超出上下文容量，主动建议拆分会话，并将未完成事项写入 `docs/handoff_YYYYMMDD.md`。

---

## 14. 代码编辑安全（强制）

- 修改 `src/` 下 `.py` 文件前必须说明计划并获用户批准。
- 实验性修改必须创建新文件，不得直接覆盖原文件。
- `results/` 和 `ns3_validation/results/` 下的 JSON/CSV 禁止直接修改，需生成新文件并附 `patched_from` + `patch_note`。
- 只修改与当前任务直接相关的代码，禁止"顺手"重构。

---

## 15. 任务执行纪律（强制）

- 收到任务后先复述理解，确认方向正确后再动手。
- 严格执行用户指定的任务范围，禁止自行扩展 scope。
- 同一操作连续失败 2 次后必须停下来说明情况，禁止盲目重试。
- 任务完成后给出交付物清单（文件路径 + 简要说明），不超过 5 行。

---

## 16. 论文证据白名单制度（v19 起生效）

- 论文数值必须且仅可来自白名单文件（见 `docs/20260215_evidence_whitelist_v19.md`）。
- 冒烟测试、诊断、早期版本、已被替代的文件禁止在论文中引用。
- 新实验结果需经 claim_source_matrix 验证后方可加入白名单。
- 声明-证据映射记录在 `docs/20260215_v19_claim_source_matrix_v3.csv`（v1/v2 已废弃，仅保留审计链）。
- 校验脚本: `scripts/validate_claim_source_matrix.py`，新增/修改 claim 后必须重跑并确认 FAIL=0。

---

## 17. SSH 连接规范（强制，禁止遗忘）

- FatMachine 唯一正确连接方式: `ssh FatMachine`（Tailscale IP 100.104.82.45, User sshuser, key 认证）。
- 禁止直接拼 IP + 用户名，禁止密码认证。
- WSL: `ssh FatMachine "wsl -u ns3user -- bash -c 'command'"`
- Python: `C:\Users\sshuser\miniconda3\envs\aether-wsn\python.exe`（conda activate 不可用）。
- scp 路径必须用正斜杠。禁止 nohup/start /B 等不可靠后台方式。

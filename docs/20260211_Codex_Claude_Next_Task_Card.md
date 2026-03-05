# 20260211 Codex->Claude 下一步任务卡

## 目标
在不改核心算法的前提下，修复 NS-3 证据链的可复核性与论文引用一致性。

## 任务 C1（P0）
- 路径：`ns3_validation/results/ns3_scale_ext_significance.py`
- 修改内容：
  1. 在输出 `ns3_scale_ext_significance.csv` 时新增字段：`node_count`、`baseline`、`metric`。
  2. 仍保留 `comparison` 字段（向后兼容）。
  3. 统一列名：`aeris_mean`、`baseline_mean`，不要再使用 `leach_mean` 兼容字段。
- 影响范围：仅统计脚本与生成的 CSV，不改仿真源码。

## 任务 C2（P0）
- 路径：`ns3_validation/results/NS3_Section8_附录表.md`
- 修改内容：
  1. 仅从 `ns3_scale_ext_significance.csv` 和 `ns3_scale_ext_stats.csv` 重生附录表。
  2. 确保 UTF-8 无乱码，禁止出现编码异常字符。
  3. 在表头显式写明：`trend-level validation only`。
- 影响范围：仅附录文档。

## 任务 C3（P1）
- 路径：`ns3_validation/results/NS3_vs_Python_对照表.md`
- 修改内容：
  1. 增加“口径差异说明”小节：Python 为高层协议仿真，NS-3 为网络栈仿真。
  2. 所有对照行必须标注 `environment`、`node_count`、`n`。
  3. 明确禁止语句：`numerical equivalence completed`。
- 影响范围：仅对照文档。

## 验收标准（必须全部满足）
1. `ns3_scale_ext_significance.csv` 包含列：`environment,node_count,comparison,baseline,metric,aeris_mean,baseline_mean,diff,hedges_g,p_value_raw,p_value_holm,sig_holm_0_05`。
2. `NS3_Section8_附录表.md` 与 `NS3_vs_Python_对照表.md` 均为 UTF-8 可读，无乱码。
3. 所有结论可回溯到 CSV 行，不使用未落盘的数据。
4. 回报必须包含：修改文件路径、关键变更、仍需核对项。

## 禁止事项
- 禁止修改 `src/` 下协议实现。
- 禁止新增未指派实验。
- 禁止把 trend-level 写成 numerical-level。

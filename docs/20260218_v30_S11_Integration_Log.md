# v30 论文更新日志（S11整合）

## 本次目标
- 将已完成的 S11 matched patch-vs-control 结果正式纳入主稿。
- 不改动实验代码，不新增实验，仅做文稿与图表层更新。

## 修改文件
- `for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260217_v30.tex`
- `scripts/build_s11_figure_s26.py`
- `for_submission/figures/fig5_s11_patch_control_delta_20260217_s26.pdf`（及 svg/png）

## 关键修改点
1. **摘要更新**
   - 加入 S11 结论：AERIS 在 24/24 环境-规模单元中，patch-control 差值均为负且显著。

2. **证据分层更新**
   - `tab:regime_map` 新增 S11 行，明确其角色为 matched strict-physics confirmation。

3. **结果章节新增 S11 小节**
   - 新增 S11 表（100与1000节点 delta）。
   - 新增 Fig.5（patch-control delta 可视化）。
   - 明确写出：S11 不是“性能提升证据”，而是“真实性应力校准证据”。

4. **讨论与结论同步**
   - 将 S9/S10 扩展为 S9/S10/S11 一致口径。
   - 强化“不能把 S8 与 S9/S10/S11 数值混合池化”的边界。

## 编译状态
- `AERIS_Sensors_MDPI_Submission_Draft_20260217_v30.pdf` 已成功编译。
- 当前主要告警为窄列表格 underfull（非内容错误）。

## 备注
- 本次更新未触碰 `src/`。
- 本次更新未新增任何实验任务。

# 20260215 v18 精修记录（阶段汇报版）

## 本轮精修范围

- 稿件：`for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260215_v18.tex`
- 图表：`scripts/build_sensors_figures_s23.py` 及 `for_submission/figures/fig*_20260215_s23.*`
- 预览：`for_submission/AERIS_Sensors_MDPI_Draft_Preview.pdf`

## 核心更新

1. 大规模可扩展性口径统一为 S8 平衡矩阵（n=1000/cell）。
2. Figure 1-4 全部切换到 s23 图组（低饱和、白底、期刊风格）。
3. NS-3 表格数值与 `ns3_scale_ext_1000_significance.csv` 对齐：
   - indoor_factory: 0.6025 vs 0.5336, p=4.23e-25
   - outdoor_urban: 0.2064 vs 0.1899, p=3.41e-03
   - outdoor_suburban: 0.7771 vs 0.6921, p=1.94e-30
4. NS-3 trend 统计摘要更新为 25/28 显著（非显著集中于 indoor_office 的 100/200/1000）。

## 编译与门控

- 编译：`latexmk -pdf AERIS_Sensors_MDPI_Submission_Draft_20260215_v18.tex` 通过。
- 门控：`python scripts/check_sensors_draft_gate.py --draft ...v18.tex` 通过。

## 仍需继续

1. 等服务器 S9 输出后，再决定是否将 NS-3 段落扩展到第二 baseline（HEED）。
2. 完成“仿真严谨性修复”小矩阵 smoke/pilot，并评估是否进入全量重跑。

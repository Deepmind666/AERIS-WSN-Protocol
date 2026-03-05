# AERIS-WSN-Protocol Kiro 规则

## 语言
- 全程中文输出（代码/命令/路径/论文英文正文除外）
- compact 恢复后首句必须是中文恢复确认

## PDR 口径
- 主指标: `pdr_expected = bs_delivered / source_packets_expected`
- 禁止使用未定义的 `pdr` 字段

## 修改纪律
- 任何修改前必须说明：路径 + 计划 + 影响范围
- 只改与当前任务直接相关的代码，禁止顺手重构
- `results/` 下 JSON/CSV 禁止直接修改，需生成新文件

## 统计要求
- publication 级: n >= 30 seeds
- diagnostic 级: 必须标注 `run_tier=diagnostic`
- 报告格式: `mean ± std`，标注样本规模

## 实验规范
- 元数据必填: timestamp, git_commit, experiment_type, run_tier, primary_metric, config
- 启动前输出: 命令、资源上限、ETA（区间）、验收标准
- 验收 4 项: raw_results 计数、error_runs=0、run_tier=publication、primary_metric=pdr_expected
- 禁止高频轮询，最小间隔 30 分钟

---
name: review
description: 以 Sensors MDPI 资深审稿人身份对论文进行严格审稿。自动执行数据一致性验证、声明边界检查、图表质量审查，产出分级审稿报告（P0/P1/P2）。
---

# 严格审稿协议（Sensors MDPI 资深审稿人）

## 触发条件
用户输入 `/review` 或要求"审稿"、"严格审查"时执行。

## 执行流程

### Phase 1: 定位当前版本
1. 找到 `for_submission/` 下最新的 `.tex` 文件（按文件名版本号排序）
2. 读取 CLAUDE.md 确认当前状态
3. 读取 `.claude/RULES.md` 确认项目规则

### Phase 2: 数据一致性验证
1. 定位证据白名单文件 `docs/20260215_evidence_whitelist_v19.md`
2. 交叉验证论文中的数值与白名单 JSON/CSV 数据源
3. 检查 Table 数值与 CSV 原始数据是否一致
4. 检查 regime_map (Table 5) 描述与实际数据维度是否匹配
5. 验证图表文件是否全部存在
6. **p 值类型校验（P0 级）**：论文中报告的 p 值必须标注为 `p_Holm`（Holm 校正后），禁止将 raw p 误标为 Holm p。用脚本重算 Holm 校正值并与论文声明交叉验证。
7. **投稿包一致性**：验证 `AERIS_Sensors_Submission/` 中全部 10 张图 MD5 是否与 `for_submission/figures/` 一致；验证 `manuscript.tex` 文本（排除路径差异）与最新 `.tex` 是否 TEXT DIFF=0。

### Phase 3: 声明边界检查
1. grep 禁止声明模式：未限定的最高级词（"significantly"无统计支撑、"dramatically"、"clearly shows"）
2. 检查跨 regime 数值混用（legacy 100-node vs primary large-scale vs stress vs sensitivity）
3. 检查 diagnostic 结果是否被用作论文结论
4. 检查 PDR 口径是否一致（必须为 pdr_expected）

### Phase 4: 统计方法审查
1. 确认 Welch t-test + Holm 校正 + Hedges' g 三件套
2. 检查大样本 effect size 膨胀警告是否存在
3. 检查 n 值报告是否完整
4. **检查每个比较 family 的 m 值是否在方法节或正文中明确声明**（如 ablation m=8, 100-node m=4 等）
5. **检查 Holm 校正 family 边界是否合理**（不能把不相关的比较混入同一 family）

### Phase 5: 图表质量审查
1. 逐张审查 PDF 中的图表（读取 PDF 页面）
2. 检查：字号可读性（≥9pt）、线条重叠、配色区分度、单位标注、误差棒
3. 检查 colorbar/legend 标签与数值单位一致性
4. 多面板图坐标轴顺序跨面板一致性（如 y 轴环境排列顺序）
5. 检查图表脚本中是否有 fontsize<9 的参数

### Phase 6: 产出报告
1. 写入 `docs/YYYYMMDD_vXX_Strict_Review_Report.md`
2. 分级：P0（阻塞发布）、P1（应修复）、P2（建议改进）
3. 每条发现附 `文件:行号` 引用
4. 附数据一致性交叉验证表
5. 附优点确认（审稿人正面评价）
6. 附修复优先级建议表

## 输出格式
```markdown
# AERIS vXX 严格审稿报告
**审稿日期**: YYYY-MM-DD
**论文版本**: 文件名
**审稿人角色**: Sensors MDPI 资深审稿人（严格模式）
## 总体判定：Accept / Minor Revision / Major Revision
## P0 — 阻塞发布
## P1 — 应修复
## P2 — 建议改进
## 数据一致性交叉验证结果
## 优点确认
## 修复优先级建议
## 结论
```

## 关键约束
- 全程中文输出
- 发现即时落盘，不在对话中累积
- 单次审查不超过 5 个文件时直接执行，超过时用 Task 子代理分流
- 禁止编造数据或猜测数值，必须从源文件验证

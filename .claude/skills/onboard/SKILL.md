---
name: onboard
description: 会话恢复/新会话启动时的快速上下文加载。读取项目状态、最新进度、待办事项，避免重复 onboarding 浪费上下文。
---

# 会话恢复协议

## 触发条件
用户输入 `/onboard` 或新会话开始时执行。

## 执行步骤

### Step 1: 读取核心状态（必做）
1. 读取 `CLAUDE.md` — 项目概述与当前状态
2. 读取 `.claude/RULES.md` — 强制规则（重点：语言、PDR口径、SSH规范）
3. 读取 `.claude/lessons.md` — 已知经验教训

### Step 2: 读取最新进度（必做）
1. 查找 `docs/` 下最新的 handoff/status 文件（按日期排序）
2. 查找最新的审稿报告（`*Strict_Review*` 或 `*Figure_Review*`）
3. 用 `git log --oneline -5` 查看最近 5 次提交

### Step 3: 确认当前版本（必做）
1. 找到 `for_submission/` 下最新 `.tex` 文件
2. 确认当前分支（`git branch --show-current`）
3. 快速检查投稿包同步状态：对比 `AERIS_Sensors_Submission/figures/` 与 `for_submission/figures/` 的图表文件数和最新修改时间

### Step 4: 输出恢复摘要
格式：
```
【恢复确认】已恢复上下文，将全程中文输出。
- 分支: xxx
- 最新论文: vXX
- 最近提交: xxx
- 上次审稿状态: P0=X, P1=X, P2=X
- 投稿包同步: 是/否（10张图 MD5 匹配状态）
- 待办事项: [从 handoff 文件提取]
```

## 关键约束
- 总共只读 3-5 个文件，禁止全量扫描
- 恢复摘要不超过 10 行
- 首句必须是中文恢复确认

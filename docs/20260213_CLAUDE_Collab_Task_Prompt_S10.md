# Claude4.6 协作任务提示词（S10）

## 0. 恢复确认（第一行必须原样输出）
【恢复确认】已恢复上下文，将全程中文输出，并按本规则执行。

---

## 1. 本轮硬规则（强制）
1. 全程中文输出（代码/命令/路径/字段名可英文）。
2. 每次实验启动、每次进度更新都必须给 ETA（可给区间）。
3. 禁止高频轮询：常规>=30分钟，长跑>=45分钟；仅用户明确要求“现在检查”可立即查询。
4. 禁止重复实验：先核对现有结果目录，再决定是否补跑。
5. 服务器实验必须基于已 push commit；回报必须含 `git_commit/git_dirty/git_diff_stat/script_sha256/config_hash/run_tier/primary_metric`。
6. NS-3 为投稿强制门槛：未满足对齐与证据链条件时，只能写 trend-level validation。
7. 资源限制：服务器实验 CPU/MEM 上限按任务卡执行；本轮上限 `CPU<=90%`，并避免 OOM。

---

## 2. 当前状态（以此为准）
1. 本地（Codex）已完成：
   - `C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_indoor_factory_local_s9_20260213_010635.json`
   - `raw_results=19500`，`error_runs=0`，`run_tier=publication`。
2. 服务器（你）已完成：
   - `C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_indoor_office_server_s7_20260211.json`
   - `C:\AERIS-WSN-Protocol\results\mega_experiments\scalability_outdoor_suburban_server_s7_20260211.json`
   - 两个文件均 `raw_results=16500`，`error_runs=0`。
3. 你报告过 outdoor 相关实验异常，正在修复；此项优先级最高，先收口再扩展。
4. NS-3 已有多轮结果与统计，但仍需严格门控用语（禁止“数值级完全对齐”）。

---

## 3. 你的任务（Claude，按顺序执行）

### 任务 A（P0）：Outdoor 异常收口与复验（必须）
目标：确认 outdoor 任务是否彻底修复，并给可复核证据。

执行要求：
1. 先做文件与元数据核对，不直接重跑：
   - 检查 outdoor 相关 JSON 是否存在、是否完整（`raw_results` 计数、`error_runs`、关键元数据）。
2. 若缺口存在再补跑（只补缺口，不全量重跑）：
   - 严格写明命令、参数、资源上限、预计耗时与 ETA。
3. 产出：
   - 复验报告（md）+ 更新后的结果文件路径清单。

验收标准：
1. outdoor 目标文件存在且 `error_runs=0`。
2. 回报中包含完整元数据字段。
3. 明确说明“是否需要进一步补跑”与依据。

### 任务 B（P1）：服务器侧统一统计包（仅针对新增/修复后的 outdoor）
目标：生成与现有口径一致的统计与门控补充，避免与本地结果冲突。

执行要求：
1. 仅处理本轮新增或修复的 outdoor 结果，不改动 src 算法。
2. 输出统计：
   - 描述统计（csv）
   - 显著性（Welch + Hedges g + Holm，csv）
3. 更新一份简短门控说明（md）：
   - 可写结论
   - 禁写结论
   - 适用范围（环境/节点/样本数）

### 任务 C（P1）：NS-3 门控一致性复核（不重跑）
目标：确保 NS-3 文档与当前门控口径一致。

执行要求：
1. 只做文档复核与差异登记，不新增 NS-3 计算。
2. 明确三点：
   - 哪些结论只能 trend-level
   - 哪些场景可写显著优势
   - 哪些语句必须禁写（尤其“数值级对齐完成”）

---

## 4. 我（Codex）并行任务
1. 使用 Sensors 模板 `.tex` 继续本地论文主线编辑（不等待你）：
   - 收敛 Section 6/8 的结论口径；
   - 合并本地 `indoor_factory` 与你的 server 结果证据链。
2. 先做图表升级与论文素材整理（不启动重复大实验）：
   - 复合图（多子图）与局部放大图代码；
   - 统一配色/字体/标注，避免遮挡与乱码。
3. 等你交付 outdoor 修复结果后，我做最终合并审查并给下一轮实验决策。

---

## 5. 你的回报模板（严格）
文件清单：
- <完整路径1>
- <完整路径2>

本次完成：
1) ...
2) ...

质量检查：
1) raw_results / error_runs 校验结果
2) 元数据字段完整性结果
3) 门控语句（可写/禁写）是否通过

耗时与 ETA：
1) 已耗时
2) 剩余 ETA（或区间）
3) ETA 依据（日志时间戳/吞吐）

仍需核对：
1) ...
2) ...


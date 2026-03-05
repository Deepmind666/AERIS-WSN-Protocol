# 项目经验教训

## 环境与工具
- `conda run -n aether-wsn python` 不支持多行 `-c` 参数，需写入临时文件再执行
- `conda activate` 在SSH中不可用，必须用完整Python路径
- LaTeX编译顺序: pdflatex → bibtex → pdflatex × 2
- SSH后台任务有10分钟超时，长实验用服务器端脚本

## 论文写作
- MDPI Sensors禁止代码变量名出现在正文（如 `\texttt{pdr_expected}`）
- 环境名用自然语言（Indoor Office），不用下划线形式（indoor\_office）
- 消融配置名用连字符（No-Gateway），不用下划线（no\_gateway）
- 公式中用数学符号（N_delivered），不用代码变量名（bs\_delivered）

## 仿真与实验
- PEGASIS碰撞因子=1.0是结构性设计，非bug（顺序链传输无竞争）
- v50-rigor修复后所有实验需重跑（MAC碰撞+多跳公平性）
- 大JSON文件禁止全量读取，用python -c提取所需字段
- 实验结果JSON必须包含git_commit、run_tier、config等元数据

## 常见错误
- Edit工具要求先Read文件才能编辑
- scp路径必须用正斜杠
- 禁止批量taskkill，必须逐PID操作
- Edit工具 old_string 不唯一时会失败，需加更多上下文或用 Write 重写

## 审稿与论文（2026-02 新增）
- Table 5 regime_map 是全文证据架构索引，审稿人会用它交叉验证所有子节
- colorbar 标签必须与数值单位一致（小数 vs 百分点）
- 15条线重叠的图审稿人会直接拒，必须拆分面板
- 图表字号 <8pt 在打印时不可读
- 声称"所有问题已修复"前必须做最终自审 grep pass

## 上下文管理（2026-02 新增）
- 单次会话混合审计+文档+实验容易撑爆上下文
- 大 JSON 文件用 python -c 提取，禁止全量读入对话
- compact 恢复后第一句必须是中文恢复确认
- 接近上下文极限时写 handoff 文件再收尾

## 服务器实验（2026-02 新增）
- S10R 吞吐基准：服务器 45-55/s（20 workers），本地 3.5-4.0/s
- SSH 10 分钟超时不等于实验失败，需检查服务器端日志
- wmic/schtasks/DETACHED_PROCESS 在 SSH 中不可靠，用简单前台连接
- 136 个僵尸 Python 进程曾导致 DPC_WATCHDOG_VIOLATION 蓝屏

## 投稿包同步（2026-03 新增，v98会话经验）
- 图表生成脚本（如 s98）可能同时重新生成多个图（fig2 和 fig6），所有受影响文件都必须重新同步到 AERIS_Sensors_Submission/
- 同步后必须用 MD5 校验全部 10 张图，不能只验单张
- manuscript.tex 与 v98.tex 除 `\graphicspath` 和 `\includegraphics` 路径外，文本必须完全一致（TEXT DIFF=0）
- 每次 TeX 文本变更后，必须同步到两个目标文件并分别编译验证

## 统计报告（2026-03 新增，P0 级教训）
- **Raw p-value ≠ Holm-corrected p-value**：论文中声称 Holm p<0.001 但实际 Holm p=0.005 是 P0 错误
- 消融实验 m=8（2 variants × 4 environments），Holm 校正后只有 2/8 显著
- 论文中报告 p 值时必须明确标注 `p_Holm`，禁止混用 raw p 和 corrected p
- 方法节必须明确声明每个比较 family 的 m 值和测试数量

## 图表质量（2026-03 新增）
- 图表字号最低标准为 ≥9pt（不是 8pt），8pt 在打印时不可读
- 脚本中的中文注释在 Windows GBK 环境下可能导致 UnicodeEncodeError，统一用英文注释
- 多面板图的坐标轴顺序（如 y 轴环境排列）必须跨面板一致
- Edit 工具的 old_string 不唯一时会失败，需扩大上下文范围或用 Write 重写整个文件

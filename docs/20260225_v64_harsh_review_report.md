# v64 第二轮压力测试报告（拒稿导向）
# 审稿人：Claude (Harsh Reviewer 视角)
# 日期：2026-02-25
# 稿件：AERIS_Sensors_MDPI_Submission_Draft_20260225_v64.tex

---

## 0. 审稿立场声明

本轮采用"拒稿导向"视角，刻意寻找可被 Reviewer 2/3 攻击的弱点。
第一轮结论为 Minor Revision；本轮目标是测试该结论是否经得起最苛刻的审稿压力。

---

## 1. 新发现问题总表（P0/P1/P2）

| # | 问题 | 严重级别 | 文件路径:行号 | 证据摘要 | 修复建议 |
|---|------|----------|--------------|----------|----------|
| H1 | legacy 与 primary 矩阵之间存在多混杂变量，未充分讨论 | P1 | v64.tex:175,242 | primary 同时改了两件事（开 collision/relay + 移除 baseline suppressors），导致 AERIS 100节点 PDR 在 harsh 环境中 primary 反而远高于 legacy（indoor_factory: 0.93 vs 0.60, outdoor_urban: 0.75 vs 0.37）。正文仅在第175行一句话提及 "removes previously identified baseline suppressors"，未量化两个变量各自的贡献 | 在 Discussion 中添加一段明确讨论：legacy→primary 的 PDR 变化不能单独归因于 collision/relay，baseline suppressor 移除是主要驱动因素 |
| H2 | patch-control 方向与 legacy-primary 方向矛盾，未显式调和 | P1 | v64.tex:467,302 | 第467行说 "stricter physics reduces reliability"（patch < control），但 primary 矩阵（开了 collision/relay）在100节点处 AERIS PDR 远高于 legacy。读者会困惑：同样是开 collision/relay，为什么 patch-control 说降低，legacy→primary 却升高？原因是 suppressor 移除，但正文未显式调和这两个看似矛盾的结论 | 在 Discussion 的 "Interpretation of Matched Degradation Block" 中添加一段，显式说明 legacy→primary 的升高来自 suppressor 移除而非 collision/relay 本身 |
| H3 | 22个浮动对象（13表+9图）对 Sensors 论文偏多 | P2 | v64.tex 全文 | MDPI Sensors 典型论文 8-12 个浮动对象；22个可能导致编辑要求精简 | 考虑将 Table 5 (rigor patch pilot) 和 Table 7 (PEGASIS snapshot) 移至补充材料 |
| H4 | Table 2 (ablation) 的 "full" 数据与 Table 1 完全相同 | P2 | v64.tex:231,262 | Table 1 AERIS indoor_office = 0.9739±0.0047，Table 2 full indoor_office = 0.9739±0.0047，经核实来自同一批 30 seeds 的同一组 runs | 在 Table 2 caption 或正文中显式说明 "full configuration values are identical to Table 1 as they share the same experimental runs" |
| H5 | env_sensitivity 源文件缺少 collision/relay flags 元数据 | P2 | env_sensitivity_20260207_205317.json | config 中无 mac_collision / multihop_relay / force_ctp_reliable 字段，仅靠 caption 文字标注 "flags disabled" | 建议在 JSON 元数据中补充 flags 状态（生成 patched 文件） |
| H6 | 4个 unnumbered subsection* 在 Discussion 中可能被编辑质疑 | P2 | v64.tex:536,539,568 | "Interpretation of Matched Degradation Block"、"Practical Deployment Guidance"、"Validity Notes" 使用 \subsection* 不编号 | MDPI 模板通常要求所有 subsection 编号；建议改为编号 subsection 或降级为段落标题 |

---

## 2. 核心攻击点深度分析

### 攻击点 A：legacy-primary 混杂变量（最可能被 Reviewer 抓住）

**问题本质**：legacy 矩阵和 primary 矩阵之间至少有两个同时变化的因素：
1. collision/relay flags（开 vs 关）
2. baseline suppressors（移除 vs 保留）

**数据证据**：
- indoor_factory AERIS: legacy=0.6031 → primary=0.9278（+0.3247）
- outdoor_urban AERIS: legacy=0.3745 → primary=0.7479（+0.3734）
- 但 patch-control 表显示开 collision/relay 会降低 PDR（delta 全部为负）

**Reviewer 可能的攻击**："如果开 collision 降低 PDR，为什么 primary 矩阵反而更高？作者是否在 primary 矩阵中引入了对 AERIS 有利的隐性变更？"

**防御建议**：在 Discussion 中添加 2-3 句话，显式说明 primary 矩阵的 PDR 升高主要来自 baseline suppressor 移除（这是公平性修复），collision/relay 的净效应仍然是降低 PDR（如 patch-control 所示）。两个变量的方向相反，最终结果取决于哪个效应更大。

### 攻击点 B：PEGASIS 零差异异常未充分解释

**问题本质**：在 patch-control 矩阵中，PEGASIS 24/24 cells 全部非显著（max |g|=0.0903），indoor_factory 甚至精确零差异。正文承认这是 "implementation-coupling anomaly"，但未给出任何机制假说。

**Reviewer 可能的攻击**："如果 PEGASIS 对 collision/relay patch 完全免疫，说明 patch 可能只影响了 AERIS 特有的代码路径，而非真正的物理层模拟。这质疑了整个 patch 的有效性。"

**防御建议**：在 Limitations 中补充一段机制假说（如 PEGASIS 的链式转发路径不经过 collision 检测模块），并明确声明这需要代码审计确认。

### 攻击点 C：自定义模拟器的可信度

**问题本质**：全部实验基于自定义 Python 模拟器，NS-3 仅做了 AERIS vs LEACH 的趋势验证。Reviewer 可能质疑：为什么不在 NS-3 中跑全部5个协议？

**正文已有的防御**：tex 第175行声明 NS-3 是 "external directional validation"，第518行明确 "does not claim cross-platform numerical equivalence"。

**Reviewer 可能的攻击**："仅验证了 AERIS vs LEACH，其余3个协议（PEGASIS/HEED/TEEN）在 NS-3 中的表现完全未知。primary 矩阵的排名结论缺乏独立验证。"

**防御建议**：正文已在 Limitations 第574行声明 "full five-protocol cross-platform ranking is future work"。建议在 Discussion 中再强调一次，并说明 NS-3 实现 PEGASIS/HEED/TEEN 的技术难度。

---

## 3. 综合门控判定（拒稿导向视角）

| 维度 | 第一轮 P0/P1/P2 | 第二轮新增 P0/P1/P2 | 合计 |
|------|-----------------|---------------------|------|
| 数据一致性 | 0/0/0 | 0/0/1 (H4) | 0/0/1 |
| 引用/DOI | 0/0/2 | 0/0/0 | 0/0/2 |
| 方法严谨性 | 0/0/1 | 0/2/0 (H1,H2) | 0/2/1 |
| 图文一致性 | 0/0/2 | 0/0/0 | 0/0/2 |
| 版式/投稿 | 0/1/1 | 0/0/2 (H3,H6) | 0/1/3 |
| 元数据完整性 | — | 0/0/1 (H5) | 0/0/1 |
| **总计** | **0/1/6** | **0/2/4** | **0/3/10** |

**判定规则应用**：
- P0 = 0 → 未触发 Reject
- P1 = 3 → 超过 ≤2 阈值
- 综合判定：**Minor Revision（偏 Major 边界）**

**判定理由**：
- 数据层面无 P0，核心表格全部与源文件匹配，这是最大的加分项
- H1/H2 两个 P1 本质上是同一个问题的两面（legacy-primary 混杂变量未充分讨论），修复方式是在 Discussion 中添加 2-3 段解释性文字，不需要重跑实验
- 第一轮的 P1（Table 1 caption）也是文字修复
- 3 个 P1 全部可通过文字修改关闭，无需新实验数据

---

## 4. 最小修复路径（两轮合并）

### 24h 内必须完成（3 个 P1）
1. **H1/H2 合并修复**：在 Discussion "Interpretation of Matched Degradation Block" 后添加一段，显式调和 legacy→primary 升高与 patch-control 降低的表面矛盾，说明 suppressor 移除是主要驱动因素
2. **第一轮 P1 #5**：Table 1 caption 补充 "(collision/relay flags disabled)"

### 72h 内建议完成（10 个 P2）
3. 添加 `Table~\ref{tab:cas_gateway_coeffs}` 交叉引用
4. 清理 bib 中 37 个未引用条目
5. 统一 Kandris key 命名
6. Figure 文件名重新编号（消除 fig4 跳号）
7. 补充 Gateway $\delta=-0.60$ 物理含义
8. Table 2 caption 说明数据与 Table 1 共享
9. env_sensitivity JSON 补充 flags 元数据（patched 文件）
10. 考虑将 2-3 个辅助表格移至补充材料
11. Discussion 中 \subsection* 改为编号或段落标题
12. 在 Limitations 中补充 PEGASIS 零差异的机制假说

---

## 5. 给 Codex 的执行建议清单

1. **最高优先级**：修复 H1/H2（Discussion 添加 legacy-primary 混杂变量调和段落）— 这是最可能被真实 Reviewer 攻击的点
2. **次优先级**：修复第一轮 P1 #5 + P2 #1（Table 1 caption + 系数表交叉引用）
3. **批量操作**：清理 bib 未引用条目（脚本化）
4. **不建议执行**：不需要重跑实验，不需要改 src/，不需要改数据口径
5. **风险评估**：即使不修复 H1/H2，被拒概率仍然较低（因为正文已有部分防御），但修复后可显著降低 Reviewer 质疑风险

---

## 6. 两轮审稿总结

| 轮次 | 视角 | P0 | P1 | P2 | 判定 |
|------|------|----|----|-----|------|
| 第一轮 | 标准严格 | 0 | 1 | 6 | Minor Revision |
| 第二轮 | 拒稿导向 | 0 | +2 | +4 | Minor Revision（偏 Major 边界） |
| 合计 | — | 0 | 3 | 10 | **Minor Revision** |

v64 稿件的核心数据层完全可靠（Table 1/2/3/4 全部与源文件匹配），方法论框架完整，主要弱点集中在 Discussion 的解释深度不足。所有 P1 均可通过文字修改关闭，无需新实验。

---

*报告结束*

# Sensors (MDPI) 专业审稿提示词
# 用途：交给 DeepSearch 扮演不同审稿人角色，对 AERIS 论文进行独立审稿
# 日期：2026-02-26
# 目标稿件：AERIS_Sensors_MDPI_Submission_Draft_20260225_v68.tex

---

## 使用方法

将下方三段提示词分别输入 DeepSearch，获得三位独立审稿人的意见。每段提示词对应一个审稿人角色。建议按顺序执行：Reviewer 1（方法论）→ Reviewer 2（实验设计）→ Reviewer 3（写作与投稿规范）。

每段提示词末尾附有论文摘要和关键结构信息，供 DeepSearch 在无法直接读取 tex 文件时使用。

---

## Reviewer 1：方法论与算法设计专家

```
You are an expert reviewer for MDPI Sensors (IF ≈ 3.9, Q2 in Instruments & Instrumentation). You specialize in WSN routing protocol design, optimization algorithms, and analytical modeling. You have published extensively on hierarchical clustering, cross-layer protocol design, and link-quality estimation in low-power wireless networks.

Your task is to provide a rigorous, structured review of the following manuscript submission. Adopt the perspective of a skeptical but fair reviewer who values methodological rigor, reproducibility, and honest scoping of claims.

### Review Instructions

Evaluate the manuscript on the following dimensions. For each dimension, assign a score (1–5) and provide specific, actionable feedback with reference to sections/tables/figures where applicable.

1. **Novelty and Contribution** (weight: 25%)
   - Is the CAS (Context-Adaptive Switching) + Skeleton + Gateway architecture genuinely novel, or is it an incremental combination of known techniques?
   - Does the three-component modular design offer sufficient differentiation from existing environment-aware or context-aware WSN protocols (e.g., EMRAR, GWOCSMA, DRL-based approaches cited in Related Work)?
   - Is the contribution primarily algorithmic, or primarily methodological (evaluation framework)?

2. **Analytical Rigor** (weight: 25%)
   - Eq.(3): CAS mode selection uses fixed linear coefficients. Is the linear scoring model justified? Why not a nonlinear or learned model? Is there sensitivity analysis on coefficient perturbation?
   - Eq.(4): The conditional reliability decomposition assumes conditionally independent path outcomes. Is this assumption validated or at least bounded? Under what conditions does it break?
   - Eq.(5): The complexity bound O(NH + H log H + E_a) — is this tight? Are there hidden constants or implementation-specific overheads not captured?
   - The Gateway scoring (Eq.2) uses four fixed coefficients (α=0.15, β=0.20, γ=0.35, δ=-0.60). How were these values chosen? Is there a principled derivation or are they hand-tuned?

3. **Baseline Fairness** (weight: 20%)
   - The baseline set is LEACH, PEGASIS, HEED, TEEN — all classic protocols from 2000–2004. Are these still appropriate baselines for a 2026 submission? Should at least one recent adaptive/learning-based protocol be included?
   - The authors acknowledge this is a custom Python simulator. What specific fairness controls prevent implementation bias favoring AERIS? Is the simulator code available for independent verification?

4. **Claim Scoping** (weight: 15%)
   - Are all claims properly scoped to the tested conditions (environments, node counts, seed counts)?
   - The abstract says "AERIS remains rank-1 in three environments" — is this claim sufficiently hedged given it comes from a single custom simulator?
   - The NS-3 validation covers only AERIS vs LEACH. Is it appropriate to draw five-protocol conclusions from a two-protocol cross-validation?

5. **Reproducibility** (weight: 15%)
   - Are the experimental settings (seeds, parameters, environment taxonomy) sufficiently documented for independent replication?
   - Is the provenance chain (script hashes, commit IDs, result JSON metadata) a genuine contribution or standard practice?

### Output Format

Structure your review as:
- **Summary**: 2–3 sentence overview of the paper and your overall assessment.
- **Strengths**: Numbered list (3–5 items).
- **Weaknesses**: Numbered list (3–5 items), each with specific location reference and suggested fix.
- **Minor Issues**: Numbered list.
- **Questions for Authors**: 3–5 specific questions that would need satisfactory answers before acceptance.
- **Overall Recommendation**: Accept / Minor Revision / Major Revision / Reject, with one-sentence justification.
- **Confidence**: 1–5 scale for your confidence in this review.
```

---

## Reviewer 2：实验设计与统计分析专家

```
You are an expert reviewer for MDPI Sensors (IF ≈ 3.9, Q2). You specialize in experimental methodology for networked systems, statistical hypothesis testing, simulation validation, and cross-platform reproducibility. You have served on TPCs for IEEE SECON, ACM SenSys, and MDPI Sensors special issues on WSN evaluation methodology.

Your task is to provide a rigorous review focusing on experimental design, statistical methods, and evidence quality. Be constructive but uncompromising on statistical correctness.

### Review Instructions

Evaluate the following dimensions. For each, assign a score (1–5) and provide specific feedback.

1. **Experimental Design Quality** (weight: 30%)
   - The study uses six evidence blocks (100-node matrix, primary large-scale matrix, stress block A, sensitivity block, stress block B, NS-3 alignment). Is this regime separation well-motivated or unnecessarily complex?
   - Sample sizes: n=30 for 100-node, n=3200 for large-scale, n=600–1000 for stress/sensitivity. Are these justified? Is n=3200 per cell excessive, and does it inflate significance?
   - The legacy 100-node matrix uses different physics settings than the primary large-scale matrix. The authors explicitly avoid cross-tier pooling. Is this separation sufficient, or does it create interpretive confusion?

2. **Statistical Methods** (weight: 25%)
   - Welch's t-test with Holm correction: appropriate choice given unequal variances. But with n=3200, virtually any difference becomes significant. Do the authors adequately address this?
   - Hedges' g values reach -14.20 and -15.36. These are extraordinarily large. The footnote acknowledges this, but is the explanation sufficient? Should the authors use a different effect-size metric for large-n regimes?
   - The paper reports population std (ddof=0) rather than sample std (ddof=1). For n=30 the difference is ~1.7%. Is this disclosed? Is it appropriate for a publication claiming 4-decimal precision?

3. **Simulator Credibility** (weight: 20%)
   - All results come from a custom Python simulator. NS-3 validation covers only AERIS vs LEACH (2 of 5 protocols). Is this sufficient for publication-tier claims about five-protocol ranking?
   - PEGASIS shows near-zero patch-control deltas (24/24 non-significant) and exact zero deltas in indoor_factory. The authors call this a "simulator-coupling artifact." Is this explanation adequate, or does it undermine confidence in the simulator?
   - The patch-control comparison shows that enabling stricter physics (MAC collision + multihop relay) reduces PDR for all protocols except PEGASIS. Could this indicate a bug rather than a physics effect?

4. **Cross-Platform Validation** (weight: 15%)
   - NS-3 validation: 25/28 cells significant, 3 non-significant (all indoor_office). Is trend-level validation a meaningful contribution, or is it too weak to support the claims?
   - The authors explicitly state NS-3 is "external directional validation" and disclaim numerical equivalence. Is this honest scoping, or is it hedging against poor alignment?
   - Search for recent Sensors/IoT papers that use NS-3 or other established simulators as primary evaluation platforms. How does this paper's approach compare to current community standards?

5. **Evidence Presentation** (weight: 10%)
   - 13 tables and 9 figures for a ~19-page paper. Is this appropriate for Sensors, or should some be moved to supplementary material?
   - Are the tables self-contained (can a reader understand each table from its caption alone)?
   - Do the figures add information beyond the tables, or are some redundant?

### Output Format

Same structure as Reviewer 1:
- **Summary**, **Strengths**, **Weaknesses**, **Minor Issues**, **Questions for Authors**, **Overall Recommendation**, **Confidence**.
```

---

## Reviewer 3：领域应用与写作规范专家

```
You are an expert reviewer for MDPI Sensors (IF ≈ 3.9, Q2). You specialize in
IoT deployment, practical WSN system design, and scientific writing quality.
You have extensive editorial experience with MDPI journals and are familiar
with their formatting requirements, reference standards, and submission
guidelines. You pay close attention to whether claims are practically
meaningful and whether the writing meets publication standards.

Your task is to review the manuscript from a deployment-relevance and
writing-quality perspective. Be direct about what works and what does not.

### Review Instructions

Evaluate the following dimensions (score 1–5 each, with specific feedback).

1. **Practical Relevance** (weight: 25%)
   - The paper positions AERIS as "reliability-oriented for harsh channels."
     Is there sufficient evidence that the tested environments (indoor_office,
     indoor_factory, outdoor_urban, outdoor_suburban) represent real
     deployment scenarios? Are the channel models validated against empirical
     measurements or standard models (e.g., ITU-R, 3GPP)?
   - The deployment guidance table (Table 13) gives per-environment
     recommendations. Are these actionable for a practitioner, or too
     abstract?
   - Hop count is used as a latency proxy. The authors acknowledge this is
     not wall-clock latency. Is this limitation adequately communicated, or
     could readers misinterpret the results?

2. **Positioning Against State of the Art** (weight: 25%)
   - Search for the most recent (2024–2026) WSN routing papers published in
     Sensors, IEEE IoT Journal, Computer Networks, and Scientific Reports.
     How does AERIS compare in terms of:
     (a) evaluation scale (node counts, environment diversity),
     (b) statistical rigor (sample sizes, correction methods),
     (c) cross-platform validation approach?
   - The Related Work cites 36 references including 15+ from 2024–2026.
     Is the coverage adequate? Are there important recent works missing?
   - The baseline set (LEACH/PEGASIS/HEED/TEEN) is classical. Search for
     whether recent Sensors papers still use these baselines, or whether
     the community has moved to newer reference protocols.

3. **Writing Quality** (weight: 20%)
   - Is the abstract informative and self-contained? Does it avoid
     jargon that would confuse non-specialist readers?
   - Is the paper well-organized? Does the flow from System Model →
     Experimental Setup → Results → Discussion → Limitations follow
     a logical progression?
   - Are there any instances of informal, development-style, or
     non-publication language remaining in the text?
   - Check for grammatical errors, awkward phrasing, or inconsistent
     terminology.

4. **MDPI/Sensors Compliance** (weight: 15%)
   - Does the manuscript follow MDPI Sensors formatting guidelines
     (section numbering, figure/table placement, reference style)?
   - The Introduction uses \subsection*{Contributions} (unnumbered).
     Is this acceptable under MDPI style?
   - Are all figures vector-quality (PDF) and appropriately sized?
   - Is the Data Availability Statement compliant with MDPI policy?

5. **Ethical and Transparency Considerations** (weight: 15%)
   - The paper uses a custom simulator rather than an established
     platform. Is the transparency level (provenance metadata, seed
     lists, script hashes) sufficient to address reproducibility
     concerns?
   - Are limitations honestly stated? Is there any overclaiming?
   - The authors state "no external funding." Does the scope of
     experiments (hundreds of thousands of simulation runs) seem
     consistent with unfunded research?

### Output Format

Same structure as Reviewer 1 and 2:
- **Summary**, **Strengths**, **Weaknesses**, **Minor Issues**,
  **Questions for Authors**, **Overall Recommendation**, **Confidence**.

### Additional Search Task

Before writing your review, please search for:
1. Recent (2024–2026) WSN routing papers in Sensors (MDPI) that use
   LEACH/PEGASIS/HEED/TEEN as baselines — are these still standard?
2. Recent Sensors papers that use custom simulators vs NS-3/OMNeT++ —
   what is the current community expectation?
3. Any published WSN routing paper with sample sizes ≥ 1000 per cell —
   is n=3200 unusually large or becoming standard?
Use your search findings to contextualize your review.
```

---

## 附录：论文关键信息摘要（供 DeepSearch 参考）

### 基本信息
- 标题：AERIS: Environment-Aware Hierarchical Routing for Reliable Wireless Sensor Networks under Realistic Channel Conditions
- 目标期刊：MDPI Sensors (IF ≈ 3.9, Q2)
- 页数：19 页
- 浮动对象：13 表 + 9 图 = 22 个

### 摘要
This paper presents AERIS, an environment-aware hierarchical routing protocol for wireless sensor networks, evaluated using packet delivery ratio (PDR). In the legacy four-environment 100-node comparability matrix (30 seeds per environment), AERIS attains the highest mean PDR among LEACH, PEGASIS, HEED, and TEEN. In a stricter large-scale matrix (100–1000 nodes) with balanced sampling (n=3200 independent runs per environment-node-protocol cell) and explicit MAC-collision plus multi-hop relay modeling, AERIS remains rank-1 in three environments, while PEGASIS is strongest in indoor_office at high scale. Matched tx-power tests (5 vs 15 dBm, n=600 per cell) show strong environment-dependent responses. Matched patch-control tests (n=1000 vs 1000) show consistent absolute PDR degradation under strict physics settings, indicating calibration requirements for deployment. NS-3 is used for cross-platform trend validation of AERIS versus LEACH.

### 论文结构
1. Introduction（含 Contributions 小节）
2. Related Work（36 篇引用，15+ 篇 2024–2026）
3. System Model and Protocol（PDR 定义、AERIS 架构、CAS/Gateway/Skeleton、Eq.1–5、伪代码、复杂度）
4. Experimental Setup（发布级设置、证据范围、公平性控制、regime map、统计方法、可复现性）
5. Results（8 个子节：100-node 对比、消融、可扩展性、显著性快照、rigor-patch pilot、patch-control、tx-power 敏感性、matched patch-control、hop 延迟、NS-3 趋势验证）
6. Discussion（3 个 paragraph 小节：matched degradation 解释、部署指导、有效性说明）
7. Limitations and Future Work（7 条）
8. Conclusion

### 核心数据点（供审稿参考）

**Table 1 — Legacy 100-node matrix (n=30, collision/relay disabled)**
| Environment | AERIS | LEACH | PEGASIS | HEED | TEEN |
|---|---|---|---|---|---|
| indoor_office | 0.9739 | 0.5543 | 0.9078 | 0.9371 | 0.8222 |
| indoor_factory | 0.6031 | 0.1614 | 0.1928 | 0.2326 | 0.3113 |
| outdoor_urban | 0.3745 | 0.0552 | 0.0542 | 0.0635 | 0.1201 |
| outdoor_suburban | 0.7451 | 0.2703 | 0.3382 | 0.4221 | 0.4752 |

**Table 3 — Primary large-scale matrix at 1000 nodes (n=3200, collision/relay enabled)**
| Environment | AERIS | LEACH | PEGASIS | HEED | TEEN | AERIS rank |
|---|---|---|---|---|---|---|
| indoor_office | 0.6771 | 0.0282 | 0.9884 | 0.0134 | 0.0170 | 2nd |
| indoor_factory | 0.7284 | 0.0108 | 0.2999 | 0.0054 | 0.0070 | 1st |
| outdoor_urban | 0.1359 | 0.0041 | 0.0987 | 0.0016 | 0.0025 | 1st |
| outdoor_suburban | 0.7275 | 0.0160 | 0.4725 | 0.0088 | 0.0106 | 1st |

**关键统计特征**
- PEGASIS 在 indoor_office 大规模场景下 PDR=0.9884，显著高于 AERIS 的 0.6771
- Matched patch-control: AERIS 24/24 cells 显著负 delta（stricter physics 降低 PDR）
- PEGASIS patch-control: 24/24 cells 非显著（near-zero delta，被标记为 simulator-coupling artifact）
- NS-3 趋势验证: 25/28 cells 显著（AERIS ≥ LEACH），3 个非显著 cells 均在 indoor_office

**论文自述的主要局限**
- 全部证据基于仿真（自定义 Python + NS-3 趋势验证），无硬件实测
- NS-3 仅覆盖 AERIS vs LEACH，五协议跨平台排名为 future work
- PEGASIS patch-control 零差异被标记为 simulator-coupling artifact，需专门代码审计
- Hop count 是路由长度代理，非 MAC 调度下的真实延迟

---

*提示词文件结束*

# 图表生成强制清单 - 禁止造假

**创建日期**: 2024-12-30
**核心原则**: 所有图表必须从真实实验数据生成，禁止任何形式的数据伪造

---

## ⚠️ 警告：每次生成图表前必须阅读此文档

1. **禁止**使用 `np.random` 生成假数据
2. **禁止**硬编码数值
3. **必须**从指定的JSON文件读取数据
4. **必须**验证数据点数量与实验设计一致

---

## 真实数据文件清单

| 数据文件 | 内容 | 样本量 | 验证方法 |
|----------|------|--------|----------|
| `results/intel_ablation.json` | 消融实验 | 50次×5配置=250点 | `len(data['FULL']['pdr_end2end']['values'])` 必须=50 |
| `results/intel_sensitivity.json` | 参数敏感性 | 40次×9配置=360点 | `len(data['E1.0_P256_G1']['pdr_end2end']['values'])` 必须=40 |
| `results/prior_experiments/e0_env_link_correlation.json` | E0环境相关性 | 399,485条记录 | `data['data_summary']['n_processed']` |
| `results/prior_experiments/e1_cas_features.json` | E1特征重要性 | - | 检查 `feature_importance` 字段 |
| `results/prior_experiments/e2_safety_threshold.json` | E2安全阈值 | - | 检查 `optimal_theta` 字段 |
| `results/prior_experiments/e3_load_balance.json` | E3负载均衡 | - | 检查 `correlation` 字段 |
| `results/intel_baselines_unified.json` | 基线对比 | - | 检查各协议结果 |

---

## 论文图表映射

### Figure 4: 消融实验 (最重要)

**数据源**: `results/intel_ablation.json`

**必须包含的数据**:
```python
# 验证代码
import json
with open('results/intel_ablation.json') as f:
    data = json.load(f)

# 必须验证
assert len(data['FULL']['pdr_end2end']['values']) == 50
assert len(data['-GW']['pdr_end2end']['values']) == 50
assert len(data['-FAIR']['pdr_end2end']['values']) == 50
assert len(data['-SAFETY']['pdr_end2end']['values']) == 50
assert len(data['-CAS']['pdr_end2end']['values']) == 50

# 真实数值 (用于验证图表)
# FULL PDR mean: 0.4769
# -GW PDR mean: 0.3832 (下降19.7%)
# -SAFETY PDR mean: 0.3686 (下降22.7%)
```

**输出文件**: `results/real_data_figures/fig4_ablation_real_data.pdf`

**论文位置**: `for_submission/aeris_paper_final.tex` 中的 `\label{fig:ablation}`

---

### Figure 7: 参数敏感性

**数据源**: `results/intel_sensitivity.json`

**必须包含的数据**:
```python
# 验证代码
import json
with open('results/intel_sensitivity.json') as f:
    data = json.load(f)

# 9个配置
configs = ['E1.0_P256_G1', 'E1.0_P256_G2', 'E1.0_P256_G3',
           'E1.0_P512_G1', 'E1.0_P512_G2', 'E1.0_P512_G3',
           'E1.0_P1024_G1', 'E1.0_P1024_G2', 'E1.0_P1024_G3']

for cfg in configs:
    assert len(data[cfg]['pdr_end2end']['values']) == 40

# 真实数值 (用于验证图表)
# E1.0_P256_G1 PDR mean: 0.559
# E1.0_P256_G2 PDR mean: 0.502
# E1.0_P256_G3 PDR mean: 0.455
```

**输出文件**: `results/real_data_figures/fig7_sensitivity_real_data.pdf`

**论文位置**: `for_submission/aeris_paper_final.tex` 中的 `\label{fig:sensitivity}`

---

### Figure 2: 环境-链路相关性 (E0)

**数据源**: `results/prior_experiments/e0_env_link_correlation.json`

**关键数值**:
- 湿度-链路相关性: r = -0.499
- 温度-链路相关性: r = -0.292
- 预测器AUC: 0.990

---

### Figure 3: 前期实验汇总 (E0-E4)

**数据源**: 
- `results/prior_experiments/e0_env_link_correlation.json`
- `results/prior_experiments/e1_cas_features.json`
- `results/prior_experiments/e2_safety_threshold.json`
- `results/prior_experiments/e3_load_balance.json`
- `results/benchmark_decision_time.json` (E4)

---

## 图表生成脚本要求

每个图表生成脚本必须：

1. **开头验证数据**:
```python
def validate_data(data, expected_n):
    """验证数据真实性"""
    actual_n = len(data)
    if actual_n != expected_n:
        raise ValueError(f"数据验证失败: 期望{expected_n}个点, 实际{actual_n}个点")
    print(f"✓ 数据验证通过: {actual_n}个真实数据点")
```

2. **打印数据来源**:
```python
print(f"数据来源: {data_file}")
print(f"数据点数: {len(values)}")
print(f"均值: {np.mean(values):.4f}")
```

3. **在图表上标注数据来源**:
```python
fig.text(0.5, 0.02, f'Data: {data_file} (n={n})', ha='center', fontsize=8)
```

---

## 执行检查清单

每次生成图表时，必须检查：

- [ ] 数据文件存在且可读
- [ ] 数据点数量与实验设计一致
- [ ] 均值与已知真实值匹配
- [ ] 图表标注了数据来源
- [ ] 输出文件保存到正确位置
- [ ] 论文中引用了正确的图表文件

---

## 当前状态

| 图表 | 数据验证 | 图表生成 | 论文更新 |
|------|----------|----------|----------|
| Figure 4 (消融) | ✅ 已验证 | ⚠️ 需重新生成 | ⚠️ 需更新 |
| Figure 7 (敏感性) | ✅ 已验证 | ⚠️ 需重新生成 | ⚠️ 需更新 |
| Figure 2 (E0) | ✅ 已验证 | ⚠️ 需检查 | ⚠️ 需更新 |
| Figure 3 (E0-E4) | ⚠️ 需验证 | ⚠️ 需检查 | ⚠️ 需更新 |
| Figure 5 (协议对比) | ⚠️ 需验证 | ⚠️ 需检查 | ⚠️ 需更新 |

---

## 下一步行动

1. **立即执行**: 运行 `scripts/figure_generation/generate_real_data_figures.py`
2. **验证输出**: 检查 `results/real_data_figures/` 中的图表
3. **更新论文**: 修改 `for_submission/aeris_paper_final.tex` 中的图表引用
4. **编译检查**: 编译PDF确认图表正确显示

---

*此文档是强制性参考，每次图表相关工作必须先阅读此文档*

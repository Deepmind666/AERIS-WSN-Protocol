#!/usr/bin/env python3
"""生成统计显著性检验表格"""
import json
import numpy as np
from scipy import stats
from collections import defaultdict

# 加载验证数据
with open('results/large_scale_scalability_verified.json') as f:
    data = json.load(f)

# 按节点数和协议分组
grouped = defaultdict(lambda: defaultdict(list))
for run in data['runs']:
    if run.get('success'):
        n = run['num_nodes']
        p = run['protocol']
        pdr = run['metrics'].get('pdr_end2end', 0)
        grouped[n][p].append(pdr)

print("=" * 80)
print("AERIS vs 基线协议 统计显著性检验")
print("=" * 80)

def cohens_d(group1, group2):
    """计算Cohen's d效应量"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std if pooled_std > 0 else 0

results = []
for n in sorted(grouped.keys()):
    aeris = np.array(grouped[n]['AERIS'])
    for baseline in ['PEGASIS', 'LEACH', 'HEED']:
        if baseline not in grouped[n]:
            continue
        other = np.array(grouped[n][baseline])

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(aeris, other, equal_var=False)

        # Mann-Whitney U test
        u_stat, u_pval = stats.mannwhitneyu(aeris, other, alternative='greater')

        # Cohen's d
        d = cohens_d(aeris, other)

        results.append({
            'nodes': n,
            'comparison': f'AERIS vs {baseline}',
            'aeris_mean': np.mean(aeris),
            'other_mean': np.mean(other),
            't_stat': t_stat,
            'p_value': p_val,
            'u_pval': u_pval,
            'cohens_d': d
        })

# 输出表格
print("\n### Table: Statistical Significance Tests\n")
print("| Nodes | Comparison | AERIS | Baseline | t-stat | p-value | Cohen's d |")
print("|-------|------------|-------|----------|--------|---------|-----------|")
for r in results:
    sig = "***" if r['p_value'] < 0.001 else "**" if r['p_value'] < 0.01 else "*" if r['p_value'] < 0.05 else ""
    print(f"| {r['nodes']} | {r['comparison']} | {r['aeris_mean']*100:.1f}% | {r['other_mean']*100:.1f}% | {r['t_stat']:.2f} | {r['p_value']:.2e}{sig} | {r['cohens_d']:.2f} |")

print("\n注: * p<0.05, ** p<0.01, *** p<0.001")

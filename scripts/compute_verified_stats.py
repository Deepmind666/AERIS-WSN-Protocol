#!/usr/bin/env python3
"""计算验证数据的统计摘要"""
import json
import numpy as np
from collections import defaultdict

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

# 计算统计量
print("=" * 70)
print("AERIS 可扩展性实验统计摘要 (验证数据)")
print("=" * 70)
print(f"配置: {data['config']['replicates']}次重复, {data['config']['rounds']}轮")
print()

for n in sorted(grouped.keys()):
    print(f"\n--- {n} 节点 ---")
    for p in ['AERIS', 'PEGASIS', 'LEACH', 'HEED']:
        if p in grouped[n]:
            vals = grouped[n][p]
            arr = np.array(vals)
            mean = np.mean(arr)
            std = np.std(arr, ddof=1)
            ci95 = 1.96 * std / np.sqrt(len(arr))
            print(f"  {p:8s}: PDR={mean*100:5.2f}% +/- {ci95*100:.2f}% (n={len(vals)})")

# 生成论文表格格式
print("\n" + "=" * 70)
print("论文表格格式 (Markdown)")
print("=" * 70)
print("\n| 节点数 | AERIS PDR | PEGASIS PDR | LEACH PDR | HEED PDR |")
print("|--------|-----------|-------------|-----------|----------|")
for n in sorted(grouped.keys()):
    row = f"| {n} |"
    for p in ['AERIS', 'PEGASIS', 'LEACH', 'HEED']:
        if p in grouped[n]:
            vals = grouped[n][p]
            mean = np.mean(vals) * 100
            row += f" {mean:.1f}% |"
        else:
            row += " - |"
    print(row)

# AERIS相对改进
print("\n" + "=" * 70)
print("AERIS 相对改进")
print("=" * 70)
for n in sorted(grouped.keys()):
    aeris_mean = np.mean(grouped[n]['AERIS'])
    print(f"\n{n} 节点:")
    for p in ['PEGASIS', 'LEACH', 'HEED']:
        if p in grouped[n]:
            other_mean = np.mean(grouped[n][p])
            improve = (aeris_mean - other_mean) / other_mean * 100
            print(f"  vs {p:8s}: +{improve:.1f}%")

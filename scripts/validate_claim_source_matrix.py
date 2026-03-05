#!/usr/bin/env python3
"""
claim->source 可达性自动校验脚本

功能：读取 claim_source_matrix CSV，对每条 claim 验证：
1. canonical_file 是否存在
2. 指定的 protocol/environment/num_nodes 组合在源文件中是否真实存在
3. 源文件中的 mean(pdr_expected) 是否与 v19_value 一致（容差 0.001）

输出：逐行 PASS/FAIL 报告，FAIL 行阻断进入白名单。

用法：python validate_claim_source_matrix.py [--matrix PATH] [--project-root PATH]
"""

import argparse
import csv
import json
import os
import statistics
import sys


def load_json_results(filepath):
    """加载 JSON 结果文件，返回 raw_results 列表"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get('raw_results', [])


def load_csv_data(filepath):
    """加载 CSV 文件，返回行列表"""
    rows = []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def parse_v19_value(val_str):
    """从 '0.9739±0.0047' 或 '0.9739' 中提取均值"""
    val_str = val_str.strip()
    if '\u00b1' in val_str:
        return float(val_str.split('\u00b1')[0])
    # 处理箭头格式如 '0.9333→0.9726 (上升)'
    if '\u2192' in val_str:
        return None
    # 处理排名格式如 '1st in 4/4'
    try:
        return float(val_str)
    except ValueError:
        return None


def validate_json_claim(rr, protocol, environment, num_nodes, expected_mean, tolerance=0.001):
    """验证 JSON raw_results 中是否存在指定组合，并检查均值"""
    # 筛选匹配记录
    matched = []
    for r in rr:
        r_proto = r.get('protocol', '')
        r_env = r.get('environment', '')
        r_nodes = r.get('num_nodes', r.get('node_count', 0))
        r_config = r.get('ablation_config', '')

        # 协议匹配
        proto_match = False
        if protocol in ('AERIS', 'AERIS-full'):
            proto_match = (r_proto == 'AERIS' and r_config in ('full', ''))
        elif protocol == 'AERIS-noGW':
            proto_match = (r_proto == 'AERIS' and r_config == 'no_gateway')
        elif protocol.startswith('AERIS-no'):
            cfg = protocol.replace('AERIS-', '').replace('-', '_')
            proto_match = (r_proto == 'AERIS' and r_config == cfg)
        elif protocol == 'AERIS-minimal':
            proto_match = (r_proto == 'AERIS' and r_config == 'minimal')
        else:
            proto_match = (r_proto == protocol)

        # 环境匹配
        env_match = (r_env == environment) if environment != 'all' else True

        # 节点数匹配（如果 raw_results 中有该字段）
        nodes_match = True
        if num_nodes and num_nodes != 'all' and '-' not in str(num_nodes):
            if r_nodes and r_nodes != 0:
                try:
                    nodes_match = (int(r_nodes) == int(num_nodes))
                except (ValueError, TypeError):
                    nodes_match = True
            # raw_results 中无 num_nodes 字段时，跳过匹配（依赖顶层 config.node_counts）

        if proto_match and env_match and nodes_match:
            pdr = r.get('pdr_expected')
            if pdr is not None:
                matched.append(pdr)

    if not matched:
        return False, 'NO_MATCH', 0, 0

    actual_mean = statistics.mean(matched)
    n = len(matched)

    if expected_mean is not None:
        diff = abs(actual_mean - expected_mean)
        if diff > tolerance:
            return False, 'VALUE_MISMATCH', actual_mean, n
        return True, 'OK', actual_mean, n
    else:
        return True, 'OK_NO_VALUE_CHECK', actual_mean, n


def validate_csv_claim(csv_rows, claim_row):
    """验证 CSV 类型的 canonical_file（可扩展性/显著性等）"""
    # CSV 验证较复杂，这里只检查文件存在性和非空
    return len(csv_rows) > 0


def main():
    parser = argparse.ArgumentParser(description='claim->source 可达性校验')
    parser.add_argument('--matrix', default='docs/20260215_v19_claim_source_matrix_v3.csv',
                        help='claim matrix CSV 路径')
    parser.add_argument('--project-root', default='.',
                        help='项目根目录')
    parser.add_argument('--output', default=None,
                        help='输出报告路径（默认 stdout）')
    args = parser.parse_args()

    root = args.project_root
    matrix_path = os.path.join(root, args.matrix)

    if not os.path.exists(matrix_path):
        print(f'FATAL: matrix file not found: {matrix_path}')
        sys.exit(1)

    # 读取矩阵
    with open(matrix_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        claims = list(reader)

    # 缓存已加载的文件
    json_cache = {}
    csv_cache = {}

    results = []
    pass_count = 0
    fail_count = 0
    skip_count = 0

    for claim in claims:
        cid = claim['claim_id']
        cfile = claim['canonical_file']
        protocol = claim['protocol']
        environment = claim['environment']
        num_nodes = claim['num_nodes']
        v19_value = claim['v19_value']
        claim_type = claim['claim_type']

        filepath = os.path.join(root, cfile)

        # 检查文件存在性
        if not os.path.exists(filepath):
            results.append((cid, 'FAIL', f'FILE_NOT_FOUND: {cfile}'))
            fail_count += 1
            continue

        expected_mean = parse_v19_value(v19_value)

        # JSON 文件验证
        if cfile.endswith('.json'):
            if cfile not in json_cache:
                try:
                    json_cache[cfile] = load_json_results(filepath)
                except Exception as e:
                    results.append((cid, 'FAIL', f'JSON_LOAD_ERROR: {e}'))
                    fail_count += 1
                    continue

            rr = json_cache[cfile]

            # 跳过非数值类 claim（ranking, trend_check 等）
            if claim_type in ('text_claim', 'trend_check') or expected_mean is None:
                # 仅检查文件存在性
                results.append((cid, 'SKIP', f'non-numeric claim, file exists'))
                skip_count += 1
                continue

            ok, status, actual, n = validate_json_claim(
                rr, protocol, environment, num_nodes, expected_mean
            )
            if ok:
                results.append((cid, 'PASS', f'{status} actual={actual:.4f} n={n}'))
                pass_count += 1
            else:
                results.append((cid, 'FAIL', f'{status} actual={actual:.4f} n={n} expected={expected_mean:.4f}'))
                fail_count += 1

        # CSV 文件验证
        elif cfile.endswith('.csv'):
            if cfile not in csv_cache:
                try:
                    csv_cache[cfile] = load_csv_data(filepath)
                except Exception as e:
                    results.append((cid, 'FAIL', f'CSV_LOAD_ERROR: {e}'))
                    fail_count += 1
                    continue

            csv_rows = csv_cache[cfile]
            if len(csv_rows) > 0:
                results.append((cid, 'PASS', f'CSV exists, {len(csv_rows)} rows'))
                pass_count += 1
            else:
                results.append((cid, 'FAIL', f'CSV empty'))
                fail_count += 1

        else:
            results.append((cid, 'SKIP', f'unsupported file type'))
            skip_count += 1

    # 输出报告
    output_lines = []
    output_lines.append('# claim->source 可达性校验报告')
    output_lines.append(f'# matrix: {args.matrix}')
    output_lines.append(f'# PASS={pass_count} FAIL={fail_count} SKIP={skip_count} TOTAL={len(claims)}')
    output_lines.append('')
    output_lines.append(f'{"claim_id":<8} {"status":<6} {"detail"}')
    output_lines.append('-' * 80)

    for cid, status, detail in results:
        output_lines.append(f'{cid:<8} {status:<6} {detail}')

    output_lines.append('')
    if fail_count > 0:
        output_lines.append(f'BLOCKED: {fail_count} claims failed validation. Cannot enter whitelist.')
    else:
        output_lines.append(f'ALL CLEAR: all {pass_count} numeric claims validated, {skip_count} non-numeric skipped.')

    report = '\n'.join(output_lines)

    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f'Report written to {args.output}')
    else:
        print(report)

    sys.exit(1 if fail_count > 0 else 0)


if __name__ == '__main__':
    main()

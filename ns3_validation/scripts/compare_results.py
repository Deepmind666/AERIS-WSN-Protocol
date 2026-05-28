#!/usr/bin/env python3
"""
Cross-Validation Comparison Script

Compares Python simulation results with NS-3 results for validation.

Usage:
    python compare_results.py [--python-results PATH] [--ns3-results PATH]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from scipy import stats

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class CrossValidator:
    """Compare Python and NS-3 simulation results"""

    def __init__(self, python_results_path: str, ns3_results_path: str):
        self.python_results = self._load_results(python_results_path)
        self.ns3_results = self._load_results(ns3_results_path)

        # Acceptance thresholds
        self.pdr_threshold = 0.05  # 5% difference
        self.energy_threshold = 0.10  # 10% difference

    def _load_results(self, path: str) -> Dict:
        """Load results from JSON file"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Results file not found: {path}")
            return {}
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            return {}

    def compare_scalability(self) -> Dict:
        """Compare scalability results"""
        print("\n" + "="*70)
        print("SCALABILITY COMPARISON: Python vs NS-3")
        print("="*70)

        if not self.python_results or not self.ns3_results:
            print("Missing results for comparison")
            return {}

        py_scale = self.python_results.get('scalability', {})
        ns3_scale = self.ns3_results.get('scalability', {})

        comparison = {}

        print(f"\n{'Nodes':<10} {'Python PDR':<15} {'NS-3 PDR':<15} {'Δ PDR':<12} {'Status':<10}")
        print("-" * 70)

        for nodes in sorted([int(k) for k in py_scale.keys() if k.isdigit()]):
            nodes_str = str(nodes)

            if nodes_str in py_scale and nodes in ns3_scale:
                py_pdr = py_scale[nodes_str].get('pdr_mean', 0)
                ns3_pdr = ns3_scale[nodes].get('pdr_mean', 0)

                delta = abs(py_pdr - ns3_pdr)
                status = "✓ PASS" if delta <= self.pdr_threshold else "✗ FAIL"

                comparison[nodes] = {
                    'python_pdr': py_pdr,
                    'ns3_pdr': ns3_pdr,
                    'delta': delta,
                    'passed': delta <= self.pdr_threshold
                }

                print(f"{nodes:<10} {py_pdr:<15.4f} {ns3_pdr:<15.4f} {delta:<12.4f} {status:<10}")

        return comparison

    def compare_ablation(self) -> Dict:
        """Compare ablation study results"""
        print("\n" + "="*70)
        print("ABLATION COMPARISON: Python vs NS-3")
        print("="*70)

        py_ablation = self.python_results.get('ablation', {})
        ns3_ablation = self.ns3_results.get('ablation', {})

        comparison = {}

        print(f"\n{'Variant':<15} {'Python PDR':<15} {'NS-3 PDR':<15} {'Δ PDR':<12} {'Status':<10}")
        print("-" * 70)

        variants = ['FULL', '-CAS', '-Fairness', '-Gateway']

        for variant in variants:
            if variant in py_ablation and variant in ns3_ablation:
                py_pdr = py_ablation[variant].get('pdr_mean', 0)
                ns3_pdr = ns3_ablation[variant].get('pdr_mean', 0)

                delta = abs(py_pdr - ns3_pdr)
                status = "✓ PASS" if delta <= self.pdr_threshold else "✗ FAIL"

                comparison[variant] = {
                    'python_pdr': py_pdr,
                    'ns3_pdr': ns3_pdr,
                    'delta': delta,
                    'passed': delta <= self.pdr_threshold
                }

                print(f"{variant:<15} {py_pdr:<15.4f} {ns3_pdr:<15.4f} {delta:<12.4f} {status:<10}")

        return comparison

    def statistical_comparison(self) -> Dict:
        """Perform statistical comparison between platforms"""
        print("\n" + "="*70)
        print("STATISTICAL VALIDATION")
        print("="*70)

        # Get raw PDR values for statistical tests
        py_scale = self.python_results.get('scalability', {})
        ns3_scale = self.ns3_results.get('scalability', {})

        results = {}

        for nodes in ['100', '200', '300']:
            if nodes in py_scale and int(nodes) in ns3_scale:
                py_pdrs = py_scale[nodes].get('pdr', [])
                ns3_pdrs = ns3_scale[int(nodes)].get('pdr', [])

                if py_pdrs and ns3_pdrs:
                    # Welch's t-test
                    t_stat, p_value = stats.ttest_ind(py_pdrs, ns3_pdrs, equal_var=False)

                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((np.std(py_pdrs)**2 + np.std(ns3_pdrs)**2) / 2)
                    cohens_d = (np.mean(py_pdrs) - np.mean(ns3_pdrs)) / pooled_std if pooled_std > 0 else 0

                    results[nodes] = {
                        't_statistic': float(t_stat),
                        'p_value': float(p_value),
                        'cohens_d': float(cohens_d),
                        'significant': p_value < 0.05
                    }

                    sig_str = "YES" if p_value < 0.05 else "NO"
                    print(f"\n{nodes} nodes:")
                    print(f"  t-statistic: {t_stat:.4f}")
                    print(f"  p-value:     {p_value:.4e}")
                    print(f"  Cohen's d:   {cohens_d:.4f}")
                    print(f"  Significant: {sig_str}")

        return results

    def generate_validation_report(self) -> str:
        """Generate comprehensive validation report"""

        scale_comp = self.compare_scalability()
        ablation_comp = self.compare_ablation()
        stat_comp = self.statistical_comparison()

        # Calculate overall validation status
        scale_passed = sum(1 for v in scale_comp.values() if v.get('passed', False))
        scale_total = len(scale_comp)

        ablation_passed = sum(1 for v in ablation_comp.values() if v.get('passed', False))
        ablation_total = len(ablation_comp)

        overall_passed = scale_passed + ablation_passed
        overall_total = scale_total + ablation_total

        report = f"""
# NS-3 Cross-Validation Report

## Summary

| Category | Passed | Total | Rate |
|----------|--------|-------|------|
| Scalability | {scale_passed} | {scale_total} | {scale_passed/max(scale_total,1)*100:.1f}% |
| Ablation | {ablation_passed} | {ablation_total} | {ablation_passed/max(ablation_total,1)*100:.1f}% |
| **Overall** | **{overall_passed}** | **{overall_total}** | **{overall_passed/max(overall_total,1)*100:.1f}%** |

## Acceptance Criteria

- PDR difference threshold: ≤ {self.pdr_threshold*100}%
- Energy difference threshold: ≤ {self.energy_threshold*100}%

## Conclusion

"""

        if overall_total > 0 and overall_passed / overall_total >= 0.8:
            report += """✅ **VALIDATION PASSED**

The Python simulation results are consistent with NS-3 results within acceptable margins.
This cross-validation provides confidence in the simulation accuracy.
"""
        else:
            report += """⚠️ **VALIDATION NEEDS REVIEW**

Some discrepancies were found between Python and NS-3 results.
Please review the detailed comparison above.
"""

        return report

    def save_report(self, output_path: str):
        """Save validation report to file"""
        report = self.generate_validation_report()

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        print(f"\nValidation report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare Python and NS-3 results')
    parser.add_argument('--python-results', type=str,
                       default='results/mega_experiments/all_experiments_combined.json',
                       help='Path to Python simulation results')
    parser.add_argument('--ns3-results', type=str,
                       default='results/ns3_experiments/ns3_all_experiments.json',
                       help='Path to NS-3 simulation results')
    parser.add_argument('--output', type=str,
                       default='results/ns3_experiments/validation_report.md',
                       help='Output report path')
    args = parser.parse_args()

    # Create validator
    validator = CrossValidator(args.python_results, args.ns3_results)

    # Run comparison and generate report
    validator.save_report(args.output)

    print("\n" + "="*70)
    print("Cross-Validation Complete!")
    print("="*70)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
NS-3 Cross-Validation Experiment Runner

Runs AERIS and baseline protocols in NS-3 and compares with Python results.

Usage:
    python run_ns3_experiments.py [--ns3-path PATH] [--output-dir DIR]
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class NS3ExperimentRunner:
    """Runs experiments in NS-3 simulator"""

    def __init__(self, ns3_path: str, output_dir: str):
        self.ns3_path = Path(ns3_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Experiment configurations
        self.configs = {
            'scalability': {
                'node_counts': [50, 100, 200, 300, 400, 500],
                'repeats': 30,
                'sim_time': 200
            },
            'topology': {
                'topologies': ['uniform', 'corridor', 'clustered', 'sparse'],
                'num_nodes': 100,
                'repeats': 30
            },
            'ablation': {
                'variants': [
                    {'cas': True, 'fairness': True, 'gateway': True},    # FULL
                    {'cas': False, 'fairness': True, 'gateway': True},   # -CAS
                    {'cas': True, 'fairness': False, 'gateway': True},   # -Fairness
                    {'cas': True, 'fairness': True, 'gateway': False},   # -Gateway
                ],
                'num_nodes': 100,
                'repeats': 30
            }
        }

        # Seeds for reproducibility
        self.base_seed = 42001

    def check_ns3_installation(self) -> bool:
        """Check if NS-3 is properly installed"""
        ns3_script = self.ns3_path / 'ns3'
        if not ns3_script.exists():
            print(f"Error: NS-3 not found at {self.ns3_path}")
            print("Please install NS-3 first. See ns3_validation/README.md")
            return False
        return True

    def run_ns3_simulation(self, params: Dict) -> Optional[Dict]:
        """Run a single NS-3 simulation"""
        cmd = [
            str(self.ns3_path / 'ns3'),
            'run',
            f'aeris-example '
            f'--numNodes={params["num_nodes"]} '
            f'--simTime={params["sim_time"]} '
            f'--seed={params["seed"]} '
            f'--protocol={params.get("protocol", "AERIS")} '
            f'--enableCas={str(params.get("cas", True)).lower()} '
            f'--enableFairness={str(params.get("fairness", True)).lower()} '
            f'--enableGateway={str(params.get("gateway", True)).lower()} '
            f'--outputFile={params["output_file"]}'
        ]

        try:
            result = subprocess.run(
                ' '.join(cmd),
                shell=True,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout
            )

            if result.returncode != 0:
                print(f"NS-3 error: {result.stderr}")
                return None

            # Read output file
            with open(params['output_file'], 'r') as f:
                return json.load(f)

        except subprocess.TimeoutExpired:
            print(f"Timeout running NS-3 simulation")
            return None
        except Exception as e:
            print(f"Error running NS-3: {e}")
            return None

    def run_scalability_experiment(self) -> Dict:
        """Run scalability experiment across different node counts"""
        print("\n" + "="*60)
        print("NS-3 Scalability Experiment")
        print("="*60)

        config = self.configs['scalability']
        results = {}

        for num_nodes in config['node_counts']:
            results[num_nodes] = {'pdr': [], 'energy': []}

            for i in range(config['repeats']):
                seed = self.base_seed + i
                output_file = self.output_dir / f'ns3_scale_{num_nodes}_{seed}.json'

                params = {
                    'num_nodes': num_nodes,
                    'sim_time': config['sim_time'],
                    'seed': seed,
                    'output_file': str(output_file)
                }

                print(f"\r  {num_nodes} nodes: {i+1}/{config['repeats']}", end='', flush=True)

                result = self.run_ns3_simulation(params)
                if result:
                    results[num_nodes]['pdr'].append(result.get('pdr', 0))
                    results[num_nodes]['energy'].append(result.get('total_energy', 0))

            # Compute statistics
            pdr_arr = np.array(results[num_nodes]['pdr'])
            energy_arr = np.array(results[num_nodes]['energy'])

            results[num_nodes]['pdr_mean'] = float(np.mean(pdr_arr))
            results[num_nodes]['pdr_std'] = float(np.std(pdr_arr))
            results[num_nodes]['energy_mean'] = float(np.mean(energy_arr))
            results[num_nodes]['energy_std'] = float(np.std(energy_arr))

            print(f"\r  {num_nodes} nodes: PDR={results[num_nodes]['pdr_mean']:.4f}±{results[num_nodes]['pdr_std']:.4f}")

        return results

    def run_ablation_experiment(self) -> Dict:
        """Run ablation study with different module configurations"""
        print("\n" + "="*60)
        print("NS-3 Ablation Experiment")
        print("="*60)

        config = self.configs['ablation']
        results = {}

        variant_names = ['FULL', '-CAS', '-Fairness', '-Gateway']

        for name, variant in zip(variant_names, config['variants']):
            results[name] = {'pdr': [], 'energy': []}

            for i in range(config['repeats']):
                seed = self.base_seed + i
                output_file = self.output_dir / f'ns3_ablation_{name}_{seed}.json'

                params = {
                    'num_nodes': config['num_nodes'],
                    'sim_time': 200,
                    'seed': seed,
                    'cas': variant['cas'],
                    'fairness': variant['fairness'],
                    'gateway': variant['gateway'],
                    'output_file': str(output_file)
                }

                print(f"\r  {name}: {i+1}/{config['repeats']}", end='', flush=True)

                result = self.run_ns3_simulation(params)
                if result:
                    results[name]['pdr'].append(result.get('pdr', 0))
                    results[name]['energy'].append(result.get('total_energy', 0))

            # Compute statistics
            if results[name]['pdr']:
                pdr_arr = np.array(results[name]['pdr'])
                results[name]['pdr_mean'] = float(np.mean(pdr_arr))
                results[name]['pdr_std'] = float(np.std(pdr_arr))
                print(f"\r  {name}: PDR={results[name]['pdr_mean']:.4f}±{results[name]['pdr_std']:.4f}")
            else:
                print(f"\r  {name}: No results (NS-3 may need to be installed)")

        return results

    def run_all_experiments(self) -> Dict:
        """Run all NS-3 experiments"""
        if not self.check_ns3_installation():
            print("\nSkipping NS-3 experiments (NS-3 not installed)")
            print("See ns3_validation/README.md for installation instructions")
            return {'error': 'NS-3 not installed'}

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'platform': 'NS-3',
            'scalability': self.run_scalability_experiment(),
            'ablation': self.run_ablation_experiment()
        }

        # Save combined results
        output_file = self.output_dir / 'ns3_all_experiments.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        print(f"\nAll results saved to: {output_file}")

        return all_results


def main():
    parser = argparse.ArgumentParser(description='Run NS-3 experiments for AERIS')
    parser.add_argument('--ns3-path', type=str, default=os.path.expanduser('~/ns-3'),
                       help='Path to NS-3 installation')
    parser.add_argument('--output-dir', type=str, default='results/ns3_experiments',
                       help='Output directory for results')
    args = parser.parse_args()

    # Create runner
    runner = NS3ExperimentRunner(args.ns3_path, args.output_dir)

    # Run experiments
    results = runner.run_all_experiments()

    if 'error' not in results:
        print("\n" + "="*60)
        print("NS-3 Experiments Complete!")
        print("="*60)


if __name__ == '__main__':
    main()

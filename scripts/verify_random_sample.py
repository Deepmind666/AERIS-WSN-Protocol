#!/usr/bin/env python3
"""Randomly verify a small sample of figures + result JSON files.

Outputs a JSON report under docs/ for quick sanity checking.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / 'docs' / 'results_chain_manifest.json'
OUTPUT = ROOT / 'docs' / 'results_chain_sample_check.json'


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding='utf-8'))
    return None


def sample_json_files(limit: int = 5):
    candidates = list((ROOT / 'results').rglob('*.json'))
    if not candidates:
        return []
    pick = random.sample(candidates, k=min(limit, len(candidates)))
    checks = []
    for path in pick:
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
            keys = list(data.keys()) if isinstance(data, dict) else []
            checks.append({
                'file': str(path.relative_to(ROOT)),
                'ok': True,
                'top_level_keys': keys[:12],
            })
        except Exception as exc:
            checks.append({
                'file': str(path.relative_to(ROOT)),
                'ok': False,
                'error': str(exc),
            })
    return checks


def sample_figures(manifest, limit: int = 5):
    figs = manifest.get('figures', []) if manifest else []
    if not figs:
        return []
    pick = random.sample(figs, k=min(limit, len(figs)))
    checks = []
    for item in pick:
        path = ROOT / item['file']
        checks.append({
            'file': item['file'],
            'exists': path.exists(),
            'size_bytes': path.stat().st_size if path.exists() else 0,
        })
    return checks


def main():
    manifest = load_manifest() or {}
    report = {
        'generated_at': datetime.now().isoformat(),
        'figure_samples': sample_figures(manifest),
        'json_samples': sample_json_files(),
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f"Wrote {OUTPUT}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Generate a lightweight results chain-of-custody manifest.

This script links figure files to their likely generator scripts (string match),
records file hashes, and snapshots key config files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIRS = [
    ROOT / 'for_submission' / 'figures',
    ROOT / 'results' / 'publication_figures',
    ROOT / 'results' / 'plots',
]
SCRIPT_DIR = ROOT / 'scripts'
CONFIG_FILES = [
    ROOT / 'configs' / 'phy_energy.yaml',
    ROOT / 'configs' / 'baseline_params.yaml',
]
OUTPUT_JSON = ROOT / 'docs' / 'results_chain_manifest.json'
OUTPUT_MD = ROOT / 'docs' / 'results_chain_manifest.md'


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def find_generators(target_name: str) -> list[str]:
    hits = []
    if not SCRIPT_DIR.exists():
        return hits
    for script in SCRIPT_DIR.rglob('*.py'):
        try:
            text = script.read_text(encoding='utf-8', errors='ignore')
        except Exception:
            continue
        if target_name in text:
            hits.append(str(script.relative_to(ROOT)))
    return hits


def collect_figures() -> list[dict]:
    figures = []
    seen = set()
    for fig_dir in FIGURE_DIRS:
        if not fig_dir.exists():
            continue
        for path in fig_dir.rglob('*'):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {'.pdf', '.svg', '.png'}:
                continue
            rel = path.relative_to(ROOT)
            if rel in seen:
                continue
            seen.add(rel)
            figures.append({
                'file': str(rel),
                'size_bytes': path.stat().st_size,
                'mtime_iso': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                'sha256': sha256(path),
                'generator_scripts': find_generators(path.name),
            })
    return sorted(figures, key=lambda x: x['file'])


def collect_configs() -> list[dict]:
    configs = []
    for path in CONFIG_FILES:
        if not path.exists():
            continue
        configs.append({
            'file': str(path.relative_to(ROOT)),
            'sha256': sha256(path),
            'mtime_iso': datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        })
    return configs


def main() -> None:
    manifest = {
        'generated_at': datetime.now().isoformat(),
        'root': str(ROOT),
        'figures': collect_figures(),
        'configs': collect_configs(),
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(manifest, indent=2), encoding='utf-8')

    # Markdown summary
    lines = [
        '# Results Chain Manifest',
        f"Generated: {manifest['generated_at']}",
        '',
        '## Figures',
        '| File | Size (bytes) | SHA256 | Generators |',
        '|---|---:|---|---|',
    ]
    for item in manifest['figures']:
        gens = ', '.join(item['generator_scripts']) if item['generator_scripts'] else 'N/A'
        lines.append(f"| {item['file']} | {item['size_bytes']} | {item['sha256'][:12]}… | {gens} |")
    lines.extend(['', '## Config snapshots', '| File | SHA256 |', '|---|---|'])
    for item in manifest['configs']:
        lines.append(f"| {item['file']} | {item['sha256'][:12]}… |")

    OUTPUT_MD.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"Wrote {OUTPUT_JSON} and {OUTPUT_MD}")


if __name__ == '__main__':
    main()

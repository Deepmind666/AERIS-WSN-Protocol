#!/usr/bin/env python3
"""
Aggregate local fix2 scalability outputs (2 environments) and produce:
1) descriptive CSV
2) significance CSV (Welch t-test + Hedges g + Holm correction)
3) markdown summary

Data source is fixed to the latest local run pair:
- scalability_fix2_indoor_factory_20260211_133833.json
- scalability_fix2_outdoor_urban_20260211_133833.json
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import stats


PROTOCOLS = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]
BASELINES = ["LEACH", "PEGASIS", "HEED", "TEEN"]
ENV_ORDER = ["indoor_factory", "outdoor_urban"]
NODE_ORDER = [100, 200, 300, 500, 800, 1000]

INPUTS = {
    "indoor_factory": Path(
        "results/mega_experiments/scalability_fix2_indoor_factory_20260211_133833.json"
    ),
    "outdoor_urban": Path(
        "results/mega_experiments/scalability_fix2_outdoor_urban_20260211_133833.json"
    ),
}


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1, m2 = float(np.mean(a)), float(np.mean(b))
    s1, s2 = float(np.std(a, ddof=1)), float(np.std(b, ddof=1))
    pooled = math.sqrt(((n1 - 1) * s1 * s1 + (n2 - 1) * s2 * s2) / max(n1 + n2 - 2, 1))
    if pooled < 1e-15:
        return 0.0
    d = (m1 - m2) / pooled
    df = n1 + n2 - 2
    correction = 1.0 if df <= 1 else (1 - 3 / (4 * df - 1))
    return d * correction


def holm_bonferroni(pvals: List[float], alpha: float = 0.05) -> Tuple[List[float], List[bool]]:
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: (math.isnan(x[1]), x[1]))
    adjusted = [float("nan")] * n
    running_max = 0.0
    for rank, (idx, p) in enumerate(indexed):
        if math.isnan(p):
            continue
        raw_adj = (n - rank) * p
        running_max = max(running_max, raw_adj)
        adjusted[idx] = min(1.0, running_max)
    reject = [(not math.isnan(p)) and p < alpha for p in adjusted]
    return adjusted, reject


def collect_samples(payload: Dict, env: str) -> Dict[Tuple[str, int, str], List[float]]:
    out: Dict[Tuple[str, int, str], List[float]] = {}
    for row in payload.get("raw_results", []):
        if row.get("error"):
            continue
        protocol = str(row.get("protocol"))
        node = int(row.get("num_nodes"))
        pdr = float(row.get("metrics", {}).get("pdr_expected", -1.0))
        if protocol not in PROTOCOLS or pdr < 0:
            continue
        key = (env, node, protocol)
        out.setdefault(key, []).append(pdr)
    return out


def save_csv(path: Path, rows: List[Dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    all_samples: Dict[Tuple[str, int, str], List[float]] = {}
    source_meta = []

    for env, path in INPUTS.items():
        payload = load_json(path)
        samples = collect_samples(payload, env)
        all_samples.update(samples)
        source_meta.append(
            {
                "environment": env,
                "file": str(path),
                "git_commit": payload.get("git_commit", "unknown"),
                "run_tier": payload.get("run_tier", "unknown"),
                "primary_metric": payload.get("primary_metric", "unknown"),
                "error_runs": payload.get("error_runs", "unknown"),
                "raw_results": len(payload.get("raw_results", [])),
            }
        )

    descriptive_rows: List[Dict] = []
    for env in ENV_ORDER:
        for node in NODE_ORDER:
            for protocol in PROTOCOLS:
                arr = np.array(all_samples.get((env, node, protocol), []), dtype=float)
                if len(arr) == 0:
                    continue
                descriptive_rows.append(
                    {
                        "environment": env,
                        "node_count": node,
                        "protocol": protocol,
                        "n": int(len(arr)),
                        "pdr_mean": float(np.mean(arr)),
                        "pdr_std": float(np.std(arr, ddof=1)),
                    }
                )

    sig_rows: List[Dict] = []
    pvals: List[float] = []
    for env in ENV_ORDER:
        for node in NODE_ORDER:
            a = np.array(all_samples.get((env, node, "AERIS"), []), dtype=float)
            if len(a) < 2:
                continue
            for baseline in BASELINES:
                b = np.array(all_samples.get((env, node, baseline), []), dtype=float)
                if len(b) < 2:
                    continue
                t_stat, p_val = stats.ttest_ind(a, b, equal_var=False, nan_policy="omit")
                sig_rows.append(
                    {
                        "environment": env,
                        "node_count": node,
                        "baseline": baseline,
                        "comparison": f"AERIS_vs_{baseline}",
                        "metric": "pdr_expected",
                        "aeris_mean": float(np.mean(a)),
                        "baseline_mean": float(np.mean(b)),
                        "diff": float(np.mean(a) - np.mean(b)),
                        "hedges_g": float(hedges_g(a, b)),
                        "t_stat": float(t_stat),
                        "p_value_raw": float(p_val),
                    }
                )
                pvals.append(float(p_val))

    adjusted, reject = holm_bonferroni(pvals)
    for i, row in enumerate(sig_rows):
        row["p_value_holm"] = adjusted[i]
        row["sig_holm_0_05"] = "YES" if reject[i] else "NO"

    out_dir = Path("results/mega_experiments")
    prefix = "scalability_fix2_local2env_20260211"
    desc_csv = out_dir / f"{prefix}_descriptive.csv"
    sig_csv = out_dir / f"{prefix}_significance.csv"
    md_path = out_dir / f"{prefix}_summary.md"

    save_csv(
        desc_csv,
        descriptive_rows,
        ["environment", "node_count", "protocol", "n", "pdr_mean", "pdr_std"],
    )
    save_csv(
        sig_csv,
        sig_rows,
        [
            "environment",
            "node_count",
            "comparison",
            "baseline",
            "metric",
            "aeris_mean",
            "baseline_mean",
            "diff",
            "hedges_g",
            "t_stat",
            "p_value_raw",
            "p_value_holm",
            "sig_holm_0_05",
        ],
    )

    lines: List[str] = []
    lines.append("# Local fix2 scalability summary (2 environments)")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Source files")
    for m in source_meta:
        lines.append(
            f"- {m['environment']}: {m['file']} | commit={m['git_commit']} | "
            f"tier={m['run_tier']} | metric={m['primary_metric']} | "
            f"raw_results={m['raw_results']} | error_runs={m['error_runs']}"
        )
    lines.append("")

    lines.append("## Descriptive highlights")
    for env in ENV_ORDER:
        for node in [100, 300, 500, 1000]:
            row_a = next(
                (
                    r
                    for r in descriptive_rows
                    if r["environment"] == env and r["node_count"] == node and r["protocol"] == "AERIS"
                ),
                None,
            )
            row_l = next(
                (
                    r
                    for r in descriptive_rows
                    if r["environment"] == env and r["node_count"] == node and r["protocol"] == "LEACH"
                ),
                None,
            )
            if row_a and row_l:
                lines.append(
                    f"- {env}@{node}: AERIS={row_a['pdr_mean']:.4f}, "
                    f"LEACH={row_l['pdr_mean']:.4f}, diff={row_a['pdr_mean']-row_l['pdr_mean']:+.4f}"
                )
    lines.append("")
    lines.append("## Notes")
    lines.append("- This report is local-only (indoor_factory + outdoor_urban).")
    lines.append("- Four-environment publication claims require merged fix550 statistics.")
    lines.append("")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[OK] {desc_csv}")
    print(f"[OK] {sig_csv}")
    print(f"[OK] {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


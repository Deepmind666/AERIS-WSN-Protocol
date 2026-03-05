#!/usr/bin/env python3
"""Pre-NS3 scalability analysis using available 550-replicate files.

This script merges local+server scalability results, then outputs:
1) protocol summary by environment/node (mean/std/rank)
2) AERIS vs baselines significance (Welch t-test + Hedges g + Holm correction)
3) AERIS vs best-baseline table for manuscript writing
4) markdown report with publication-safe conclusions and caveats
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy.stats import ttest_ind


@dataclass
class DatasetSpec:
    environment: str
    path: Path


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Hedges' g with small-sample correction."""
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    vx = np.var(x, ddof=1)
    vy = np.var(y, ddof=1)
    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)
    if pooled <= 0:
        return 0.0
    d = (np.mean(x) - np.mean(y)) / math.sqrt(pooled)
    correction = 1.0 - (3.0 / (4.0 * (nx + ny) - 9.0))
    return d * correction


def holm_bonferroni(pvals: List[float]) -> List[float]:
    """Return Holm-adjusted p-values in original order."""
    m = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda t: t[1])
    adj_sorted = [0.0] * m
    running = 0.0
    for i, (_, p) in enumerate(indexed):
        val = (m - i) * p
        running = max(running, val)
        adj_sorted[i] = min(running, 1.0)
    out = [0.0] * m
    for i, (orig_idx, _) in enumerate(indexed):
        out[orig_idx] = adj_sorted[i]
    return out


def load_dataset(spec: DatasetSpec) -> Dict:
    with open(spec.path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)
    return data


def build_groups(data_map: Dict[str, Dict]) -> Dict[Tuple[str, int, str], List[float]]:
    groups: Dict[Tuple[str, int, str], List[float]] = {}
    for env, payload in data_map.items():
        for r in payload["raw_results"]:
            if not r.get("success", True):
                continue
            key = (env, int(r["num_nodes"]), r["protocol"])
            groups.setdefault(key, []).append(float(r["metrics"]["pdr_expected"]))
    return groups


def build_summary(groups: Dict[Tuple[str, int, str], List[float]], environments: List[str], node_counts: List[int], protocols: List[str]) -> List[Dict]:
    rows: List[Dict] = []
    for env in environments:
        for n in node_counts:
            means = []
            for p in protocols:
                vals = groups.get((env, n, p), [])
                if vals:
                    means.append((p, float(np.mean(vals))))
            rank = {p: i + 1 for i, (p, _) in enumerate(sorted(means, key=lambda t: t[1], reverse=True))}
            for p in protocols:
                vals = groups.get((env, n, p), [])
                if not vals:
                    continue
                rows.append(
                    {
                        "environment": env,
                        "num_nodes": n,
                        "protocol": p,
                        "n": len(vals),
                        "pdr_mean": float(np.mean(vals)),
                        "pdr_std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                        "rank": rank.get(p, 999),
                    }
                )
    return rows


def build_significance(groups: Dict[Tuple[str, int, str], List[float]], environments: List[str], node_counts: List[int], baselines: List[str]) -> List[Dict]:
    raw_rows: List[Dict] = []
    pvals: List[float] = []

    for env in environments:
        for n in node_counts:
            a_vals = np.array(groups.get((env, n, "AERIS"), []), dtype=float)
            for b in baselines:
                b_vals = np.array(groups.get((env, n, b), []), dtype=float)
                if len(a_vals) == 0 or len(b_vals) == 0:
                    continue
                t_stat, p_raw = ttest_ind(a_vals, b_vals, equal_var=False)
                row = {
                    "environment": env,
                    "num_nodes": n,
                    "baseline": b,
                    "aeris_n": len(a_vals),
                    "baseline_n": len(b_vals),
                    "aeris_mean": float(np.mean(a_vals)),
                    "baseline_mean": float(np.mean(b_vals)),
                    "diff": float(np.mean(a_vals) - np.mean(b_vals)),
                    "t_stat": float(t_stat),
                    "p_raw": float(p_raw),
                    "hedges_g": float(hedges_g(a_vals, b_vals)),
                }
                raw_rows.append(row)
                pvals.append(float(p_raw))

    p_holm = holm_bonferroni(pvals)
    for row, p_adj in zip(raw_rows, p_holm):
        row["p_holm"] = p_adj
        row["sig_holm_0_05"] = "YES" if p_adj < 0.05 else "NO"
    return raw_rows


def build_best_baseline_table(summary_rows: List[Dict], sig_rows: List[Dict], environments: List[str], node_counts: List[int]) -> List[Dict]:
    by_key = {(r["environment"], r["num_nodes"], r["protocol"]): r for r in summary_rows}
    by_sig = {(r["environment"], r["num_nodes"], r["baseline"]): r for r in sig_rows}

    out: List[Dict] = []
    for env in environments:
        for n in node_counts:
            candidates = []
            for p in ["LEACH", "PEGASIS", "HEED", "TEEN"]:
                r = by_key.get((env, n, p))
                if r:
                    candidates.append((p, r["pdr_mean"]))
            if not candidates:
                continue
            best_name, best_mean = max(candidates, key=lambda t: t[1])
            aeris = by_key[(env, n, "AERIS")]
            sig = by_sig[(env, n, best_name)]
            out.append(
                {
                    "environment": env,
                    "num_nodes": n,
                    "aeris_mean": aeris["pdr_mean"],
                    "best_baseline": best_name,
                    "best_baseline_mean": best_mean,
                    "diff": aeris["pdr_mean"] - best_mean,
                    "p_holm": sig["p_holm"],
                    "hedges_g": sig["hedges_g"],
                    "aeris_rank": aeris["rank"],
                }
            )
    return out


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_markdown(
    path: Path,
    specs: List[DatasetSpec],
    summary_rows: List[Dict],
    best_rows: List[Dict],
    sig_rows: List[Dict],
) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# Pre-NS3 Scalability Analysis (Local + S1-S4)")
    lines.append("")
    lines.append(f"Generated: {now}")
    lines.append("")
    lines.append("## Input datasets")
    for spec in specs:
        lines.append(f"- {spec.environment}: {spec.path}")
    lines.append("")

    # Metadata quick table
    lines.append("## Dataset metadata gate")
    lines.append("| environment | git_commit | raw_results | error_runs |")
    lines.append("|---|---:|---:|---:|")
    for spec in specs:
        d = load_dataset(spec)
        rr = d.get("raw_results", [])
        err = sum(1 for r in rr if not r.get("success", True))
        lines.append(f"| {spec.environment} | {d.get('git_commit')} | {len(rr)} | {err} |")
    lines.append("")
    lines.append("Note: commits differ across files, but 8d76e47..b6b2e5e contains no core protocol changes (rules/docs/extract script only).")
    lines.append("")

    # rank summary
    lines.append("## AERIS rank summary")
    rank_counts: Dict[Tuple[str, int], int] = {}
    for r in summary_rows:
        if r["protocol"] == "AERIS" and r["rank"] == 1:
            rank_counts[(r["environment"], r["num_nodes"])] = 1
    total = len({(r["environment"], r["num_nodes"]) for r in summary_rows if r["protocol"] == "AERIS"})
    wins = len(rank_counts)
    lines.append(f"- AERIS rank #1 cells: {wins}/{total}")
    lines.append("")

    lines.append("## AERIS vs best baseline by environment/node")
    lines.append("| env | nodes | AERIS | best baseline | baseline | diff | p_holm | Hedges g |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|")
    for r in sorted(best_rows, key=lambda x: (x["environment"], x["num_nodes"])):
        lines.append(
            f"| {r['environment']} | {r['num_nodes']} | {r['aeris_mean']:.4f} | {r['best_baseline']} | {r['best_baseline_mean']:.4f} | {r['diff']:+.4f} | {r['p_holm']:.3e} | {r['hedges_g']:+.2f} |"
        )
    lines.append("")

    # publication-safe statements
    lines.append("## Publication-safe statements (pre-NS3)")
    lines.append("- At scale (100-1000, replicates=550), AERIS is strongest in indoor_factory and outdoor_suburban.")
    lines.append("- In indoor_office, PEGASIS is significantly better than AERIS at all tested scales.")
    lines.append("- outdoor_urban currently has complete local run; integrate into final manuscript gate after merged 4-env table review.")
    lines.append("- Do not claim universal superiority across all environments at all scales.")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    base = Path("results/mega_experiments")
    specs = [
        DatasetSpec("indoor_office", base / "scalability_indoor_office_server_fix550_20260210.json"),
        DatasetSpec("indoor_factory", base / "overnight_scalability_20260209_163524" / "scalability_indoor_factory_20260209_163524.json"),
        DatasetSpec("outdoor_urban", base / "scalability_outdoor_urban_fix550_20260210_102734.json"),
        DatasetSpec("outdoor_suburban", base / "scalability_outdoor_suburban_server_fix550_20260210.json"),
    ]

    data_map = {s.environment: load_dataset(s) for s in specs}
    environments = [s.environment for s in specs]
    node_counts = [100, 200, 300, 500, 800, 1000]
    protocols = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]

    groups = build_groups(data_map)
    summary_rows = build_summary(groups, environments, node_counts, protocols)
    sig_rows = build_significance(groups, environments, node_counts, ["LEACH", "PEGASIS", "HEED", "TEEN"])
    best_rows = build_best_baseline_table(summary_rows, sig_rows, environments, node_counts)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_summary = base / f"pre_ns3_scalability_summary_{ts}.csv"
    out_sig = base / f"pre_ns3_scalability_aeris_vs_baselines_{ts}.csv"
    out_best = base / f"pre_ns3_scalability_aeris_vs_best_{ts}.csv"
    out_md = base / f"pre_ns3_scalability_analysis_{ts}.md"

    write_csv(out_summary, summary_rows)
    write_csv(out_sig, sig_rows)
    write_csv(out_best, best_rows)
    write_markdown(out_md, specs, summary_rows, best_rows, sig_rows)

    print(f"WROTE {out_summary}")
    print(f"WROTE {out_sig}")
    print(f"WROTE {out_best}")
    print(f"WROTE {out_md}")


if __name__ == "__main__":
    main()

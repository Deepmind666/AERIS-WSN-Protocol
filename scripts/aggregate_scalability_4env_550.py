#!/usr/bin/env python3
"""
Aggregate 4-environment scalability results (n=550) from local + server runs.

This script builds publication-ready evidence tables:
1) Descriptive summary per environment/node/protocol
2) Significance table (AERIS vs LEACH/PEGASIS/HEED/TEEN) with:
   - Welch t-test
   - Hedges' g
   - Holm-Bonferroni correction
3) Claim gate list (can-claim vs cannot-claim) for manuscript wording

Inputs (fixed for current project run):
- indoor_office:     server fix550
- outdoor_suburban:  server fix550
- indoor_factory:    local overnight 20260211_010023
- outdoor_urban:     local overnight 20260211_010023
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import stats


PROTOCOLS = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]
BASELINES = ["LEACH", "PEGASIS", "HEED", "TEEN"]
ENV_ORDER = ["indoor_office", "indoor_factory", "outdoor_urban", "outdoor_suburban"]
NODE_ORDER = [100, 200, 300, 500, 800, 1000]


DEFAULT_INPUTS = {
    "indoor_office": Path(
        "results/mega_experiments/scalability_indoor_office_server_fix550_20260210.json"
    ),
    "outdoor_suburban": Path(
        "results/mega_experiments/scalability_outdoor_suburban_server_fix550_20260210.json"
    ),
    "indoor_factory": Path(
        "results/mega_experiments/overnight_scalability_20260211_010023/scalability_indoor_factory_20260211_010023.json"
    ),
    "outdoor_urban": Path(
        "results/mega_experiments/overnight_scalability_20260211_010023/scalability_outdoor_urban_20260211_010023.json"
    ),
}


@dataclass
class SampleRow:
    environment: str
    node_count: int
    protocol: str
    pdr_expected: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate 4-env scalability (n=550) and generate stats tables."
    )
    parser.add_argument(
        "--out-prefix",
        default="scalability_4env_550",
        help="Output prefix under results/mega_experiments",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def extract_rows(env: str, payload: Dict) -> List[SampleRow]:
    rows: List[SampleRow] = []
    for item in payload.get("raw_results", []):
        if item.get("error"):
            continue
        metrics = item.get("metrics", {})
        pdr = metrics.get("pdr_expected")
        if pdr is None:
            continue
        rows.append(
            SampleRow(
                environment=env,
                node_count=int(item.get("num_nodes")),
                protocol=str(item.get("protocol")),
                pdr_expected=float(pdr),
            )
        )
    return rows


def group_samples(rows: Iterable[SampleRow]) -> Dict[Tuple[str, int, str], np.ndarray]:
    grouped: Dict[Tuple[str, int, str], List[float]] = {}
    for row in rows:
        key = (row.environment, row.node_count, row.protocol)
        grouped.setdefault(key, []).append(row.pdr_expected)
    return {k: np.array(v, dtype=float) for k, v in grouped.items()}


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


def build_descriptive(grouped: Dict[Tuple[str, int, str], np.ndarray]) -> List[Dict]:
    out = []
    for env in ENV_ORDER:
        for node in NODE_ORDER:
            for protocol in PROTOCOLS:
                key = (env, node, protocol)
                arr = grouped.get(key)
                if arr is None or len(arr) == 0:
                    continue
                out.append(
                    {
                        "environment": env,
                        "node_count": node,
                        "protocol": protocol,
                        "n": int(len(arr)),
                        "pdr_mean": float(np.mean(arr)),
                        "pdr_std": float(np.std(arr, ddof=1)),
                    }
                )
    return out


def build_significance(grouped: Dict[Tuple[str, int, str], np.ndarray]) -> List[Dict]:
    records = []
    pvals = []
    for env in ENV_ORDER:
        for node in NODE_ORDER:
            key_a = (env, node, "AERIS")
            arr_a = grouped.get(key_a)
            if arr_a is None or len(arr_a) < 2:
                continue
            for bl in BASELINES:
                key_b = (env, node, bl)
                arr_b = grouped.get(key_b)
                if arr_b is None or len(arr_b) < 2:
                    continue
                t_stat, p_val = stats.ttest_ind(arr_a, arr_b, equal_var=False, nan_policy="omit")
                record = {
                    "environment": env,
                    "node_count": node,
                    "comparison": f"AERIS_vs_{bl}",
                    "aeris_mean": float(np.mean(arr_a)),
                    "baseline_mean": float(np.mean(arr_b)),
                    "diff": float(np.mean(arr_a) - np.mean(arr_b)),
                    "t_stat": float(t_stat),
                    "p_value_raw": float(p_val),
                    "hedges_g": float(hedges_g(arr_a, arr_b)),
                }
                records.append(record)
                pvals.append(float(p_val))

    adjusted, reject = holm_bonferroni(pvals)
    for i, rec in enumerate(records):
        rec["p_value_holm"] = adjusted[i]
        rec["sig_holm_0_05"] = "YES" if reject[i] else "NO"
    return records


def round_f(v: float, nd: int = 6) -> str:
    if math.isnan(v):
        return "nan"
    return f"{v:.{nd}f}"


def save_csv(path: Path, rows: List[Dict], columns: List[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_claim_gate(desc_rows: List[Dict], sig_rows: List[Dict]) -> Tuple[List[str], List[str]]:
    desc_map = {(r["environment"], r["node_count"], r["protocol"]): r for r in desc_rows}
    sig_map = {(r["environment"], r["node_count"], r["comparison"]): r for r in sig_rows}
    can_claim = []
    cannot_claim = []
    for env in ENV_ORDER:
        for node in NODE_ORDER:
            aeris = desc_map.get((env, node, "AERIS"))
            if not aeris:
                continue
            wins_all = True
            losers = []
            for bl in BASELINES:
                bl_row = desc_map.get((env, node, bl))
                if not bl_row:
                    wins_all = False
                    break
                cmp_key = (env, node, f"AERIS_vs_{bl}")
                sig = sig_map.get(cmp_key)
                if aeris["pdr_mean"] <= bl_row["pdr_mean"]:
                    wins_all = False
                    losers.append((bl, bl_row["pdr_mean"], aeris["pdr_mean"]))
                elif not sig or sig["sig_holm_0_05"] != "YES":
                    wins_all = False
            if wins_all:
                can_claim.append(
                    f"{env}@{node}: AERIS ranks first with statistically significant margins vs all baselines."
                )
            elif losers:
                for bl, bl_m, a_m in losers:
                    cannot_claim.append(
                        f"{env}@{node}: do not claim AERIS rank-1 (AERIS {a_m:.6f} < {bl} {bl_m:.6f})."
                    )
    return can_claim, cannot_claim


def write_markdown(
    path: Path,
    inputs: Dict[str, Path],
    desc_rows: List[Dict],
    sig_rows: List[Dict],
) -> None:
    can_claim, cannot_claim = build_claim_gate(desc_rows, sig_rows)
    lines: List[str] = []
    lines.append("# Scalability 4-Environment (n=550) Unified Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## Source Files")
    lines.append("")
    for env in ENV_ORDER:
        lines.append(f"- {env}: {inputs[env]}")
    lines.append("")
    lines.append("## Descriptive Summary (PDR mean +/- std)")
    lines.append("")
    lines.append("| Environment | Nodes | AERIS | LEACH | PEGASIS | HEED | TEEN |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    dmap = {(r["environment"], r["node_count"], r["protocol"]): r for r in desc_rows}
    for env in ENV_ORDER:
        for node in NODE_ORDER:
            row_cells = [env, str(node)]
            has_any = False
            for p in PROTOCOLS:
                r = dmap.get((env, node, p))
                if r:
                    has_any = True
                    row_cells.append(f"{r['pdr_mean']:.6f}+/-{r['pdr_std']:.6f}")
                else:
                    row_cells.append("-")
            if has_any:
                lines.append("| " + " | ".join(row_cells) + " |")
    lines.append("")
    lines.append("## Can-Claim Statements")
    lines.append("")
    if can_claim:
        for item in can_claim:
            lines.append(f"- {item}")
    else:
        lines.append("- None (no environment-node cell satisfies full rank-1 + all-baseline significance).")
    lines.append("")
    lines.append("## Cannot-Claim Statements")
    lines.append("")
    if cannot_claim:
        for item in cannot_claim:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Metric: pdr_expected")
    lines.append("- Significance: Welch t-test + Holm-Bonferroni correction")
    lines.append("- Effect size: Hedges g")
    lines.append("- Caveat: current 4-environment package mixes commits (server fix550 and local overnight).")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def write_manifest(path: Path, inputs: Dict[str, Path], desc_rows: List[Dict], sig_rows: List[Dict]) -> None:
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_files": {k: str(v) for k, v in inputs.items()},
        "source_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest() for k, v in inputs.items()},
        "rows_descriptive": len(desc_rows),
        "rows_significance": len(sig_rows),
        "metric": "pdr_expected",
        "method": "Welch t-test + Hedges g + Holm-Bonferroni",
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    inputs = {k: (repo_root / v).resolve() for k, v in DEFAULT_INPUTS.items()}
    for env, p in inputs.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing input file for {env}: {p}")

    all_rows: List[SampleRow] = []
    for env, path in inputs.items():
        payload = load_json(path)
        all_rows.extend(extract_rows(env, payload))

    grouped = group_samples(all_rows)
    desc_rows = build_descriptive(grouped)
    sig_rows = build_significance(grouped)

    out_dir = repo_root / "results" / "mega_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{args.out_prefix}_{stamp}"
    desc_csv = out_dir / f"{base}_descriptive.csv"
    sig_csv = out_dir / f"{base}_significance.csv"
    md_path = out_dir / f"{base}.md"
    manifest_path = out_dir / f"{base}_manifest.json"

    save_csv(
        desc_csv,
        desc_rows,
        ["environment", "node_count", "protocol", "n", "pdr_mean", "pdr_std"],
    )
    save_csv(
        sig_csv,
        sig_rows,
        [
            "environment",
            "node_count",
            "comparison",
            "aeris_mean",
            "baseline_mean",
            "diff",
            "t_stat",
            "p_value_raw",
            "p_value_holm",
            "hedges_g",
            "sig_holm_0_05",
        ],
    )
    write_markdown(md_path, inputs, desc_rows, sig_rows)
    write_manifest(manifest_path, inputs, desc_rows, sig_rows)

    print(f"[OK] {desc_csv}")
    print(f"[OK] {sig_csv}")
    print(f"[OK] {md_path}")
    print(f"[OK] {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

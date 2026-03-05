"""
Postprocess S10R (tx5/tx10/tx15) results across four environments.

This version uses the finalized 20260227 source set:
  - indoor_office/local tx5/10/15
  - indoor_factory/local tx5/10/15
  - outdoor_urban/server tx5/10/15
  - outdoor_suburban tx5 from local rerun + tx10/15 from server

Outputs:
  - s10r_4env_merged_descriptive_20260227.csv
  - s10r_4env_significance_tx5_vs_tx10_vs_tx15_20260227.csv
  - s10r_4env_reconciliation_20260227.md
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import stats


BASE = Path("results/mega_experiments")

FILES: Dict[Tuple[str, float], str] = {
    ("indoor_office", 5.0): "scalability_indoor_office_local_s10r_tx5_20260226.json",
    ("indoor_office", 10.0): "scalability_indoor_office_local_s10r_tx10_20260226.json",
    ("indoor_office", 15.0): "scalability_indoor_office_local_s10r_tx15_20260226.json",
    ("indoor_factory", 5.0): "scalability_indoor_factory_local_s10r_tx5_20260227.json",
    ("indoor_factory", 10.0): "scalability_indoor_factory_local_s10r_tx10_20260227.json",
    ("indoor_factory", 15.0): "scalability_indoor_factory_local_s10r_tx15_20260227.json",
    ("outdoor_urban", 5.0): "scalability_outdoor_urban_server_s10r_tx5_20260226.json",
    ("outdoor_urban", 10.0): "scalability_outdoor_urban_server_s10r_tx10_20260226.json",
    ("outdoor_urban", 15.0): "scalability_outdoor_urban_server_s10r_tx15_20260226.json",
    ("outdoor_suburban", 5.0): "scalability_outdoor_suburban_local_s10r_tx5_rerun_20260227.json",
    ("outdoor_suburban", 10.0): "scalability_outdoor_suburban_server_s10r_tx10_20260226.json",
    ("outdoor_suburban", 15.0): "scalability_outdoor_suburban_server_s10r_tx15_20260227.json",
}

NODES = [100, 200, 300, 500, 800, 1000]
PROTOCOLS = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]
PAIRWISE = [(5.0, 10.0), (5.0, 15.0), (10.0, 15.0)]


@dataclass
class FileAudit:
    environment: str
    tx_power: float
    filename: str
    exists: bool
    raw_results: int
    error_runs: int | None
    run_tier: str
    primary_metric: str
    git_commit: str


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_pdr(rows: Iterable[dict], env: str, nodes: int, proto: str) -> List[float]:
    out = []
    for r in rows:
        if r.get("environment") != env:
            continue
        if int(r.get("num_nodes", -1)) != nodes:
            continue
        if r.get("protocol") != proto:
            continue
        if r.get("error"):
            continue
        val = r.get("pdr_expected")
        if val is None:
            metrics = r.get("metrics", {})
            if isinstance(metrics, dict):
                val = metrics.get("pdr_expected")
        if val is not None:
            out.append(float(val))
    return out


def hedges_g(a: List[float], b: List[float]) -> float:
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    x1, x2 = np.mean(a), np.mean(b)
    s1, s2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = ((n1 - 1) * s1 + (n2 - 1) * s2) / (n1 + n2 - 2)
    if pooled <= 0:
        return float("nan")
    d = (x1 - x2) / np.sqrt(pooled)
    correction = 1.0 - (3.0 / (4.0 * (n1 + n2) - 9.0))
    return float(d * correction)


def holm_correct(rows: List[dict], p_key: str = "p_raw", out_key: str = "p_holm") -> None:
    pvals = np.array([r[p_key] for r in rows], dtype=float)
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    for i, r in enumerate(rows):
        r[out_key] = float(adj[i])
        r["significant_005"] = "YES" if r[out_key] < 0.05 else "NO"


def main() -> None:
    audits: List[FileAudit] = []
    cache: Dict[Tuple[str, float], List[dict]] = {}

    for (env, tx), filename in FILES.items():
        p = BASE / filename
        if not p.exists():
            audits.append(FileAudit(env, tx, filename, False, 0, None, "", "", ""))
            continue
        payload = load_json(p)
        rows = payload.get("raw_results", [])
        cache[(env, tx)] = rows
        audits.append(
            FileAudit(
                environment=env,
                tx_power=tx,
                filename=filename,
                exists=True,
                raw_results=len(rows),
                error_runs=payload.get("error_runs"),
                run_tier=payload.get("run_tier", ""),
                primary_metric=payload.get("primary_metric", ""),
                git_commit=payload.get("git_commit", ""),
            )
        )

    desc_rows: List[dict] = []
    envs = sorted({k[0] for k in FILES.keys()})
    for env in envs:
        for tx in [5.0, 10.0, 15.0]:
            source = cache.get((env, tx), [])
            for nn in NODES:
                for proto in PROTOCOLS:
                    vals = extract_pdr(source, env, nn, proto)
                    if not vals:
                        continue
                    desc_rows.append(
                        {
                            "environment": env,
                            "tx_power": tx,
                            "num_nodes": nn,
                            "protocol": proto,
                            "n": len(vals),
                            "pdr_mean": round(float(np.mean(vals)), 6),
                            "pdr_std": round(float(np.std(vals, ddof=1)), 6),
                        }
                    )

    desc_csv = BASE / "s10r_4env_merged_descriptive_20260227.csv"
    with desc_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["environment", "tx_power", "num_nodes", "protocol", "n", "pdr_mean", "pdr_std"],
        )
        w.writeheader()
        w.writerows(desc_rows)

    sig_rows: List[dict] = []
    for env in envs:
        for nn in NODES:
            for proto in PROTOCOLS:
                for left, right in PAIRWISE:
                    a = extract_pdr(cache.get((env, left), []), env, nn, proto)
                    b = extract_pdr(cache.get((env, right), []), env, nn, proto)
                    if len(a) < 2 or len(b) < 2:
                        continue
                    t_stat, p_raw = stats.ttest_ind(a, b, equal_var=False)
                    sig_rows.append(
                        {
                            "environment": env,
                            "num_nodes": nn,
                            "protocol": proto,
                            "comparison": f"tx{int(left)}_vs_tx{int(right)}",
                            "n_left": len(a),
                            "n_right": len(b),
                            "left_mean": round(float(np.mean(a)), 6),
                            "right_mean": round(float(np.mean(b)), 6),
                            "delta": round(float(np.mean(a) - np.mean(b)), 6),
                            "t_stat": round(float(t_stat), 6),
                            "p_raw": float(p_raw),
                            "hedges_g": round(hedges_g(a, b), 6),
                        }
                    )

    holm_correct(sig_rows)
    sig_csv = BASE / "s10r_4env_significance_tx5_vs_tx10_vs_tx15_20260227.csv"
    with sig_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "environment",
                "num_nodes",
                "protocol",
                "comparison",
                "n_left",
                "n_right",
                "left_mean",
                "right_mean",
                "delta",
                "t_stat",
                "p_raw",
                "p_holm",
                "hedges_g",
                "significant_005",
            ],
        )
        w.writeheader()
        w.writerows(sig_rows)

    report = BASE / "s10r_4env_reconciliation_20260227.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# S10R 4-Environment Reconciliation (20260227)\n\n")
        f.write("| environment | tx_power | exists | raw_results | error_runs | run_tier | primary_metric | git_commit | file |\n")
        f.write("|---|---:|---|---:|---:|---|---|---|---|\n")
        for a in audits:
            f.write(
                f"| {a.environment} | {a.tx_power:.1f} | {a.exists} | {a.raw_results} | {a.error_runs} | "
                f"{a.run_tier} | {a.primary_metric} | {a.git_commit} | {a.filename} |\n"
            )
        f.write("\n")
        f.write(f"- descriptive rows: {len(desc_rows)}\n")
        f.write(f"- significance rows: {len(sig_rows)}\n")
        missing = [a.filename for a in audits if not a.exists]
        if missing:
            f.write("- missing files:\n")
            for m in missing:
                f.write(f"  - {m}\n")
        else:
            f.write("- missing files: none\n")

    print(f"[OK] wrote {desc_csv}")
    print(f"[OK] wrote {sig_csv}")
    print(f"[OK] wrote {report}")


if __name__ == "__main__":
    main()

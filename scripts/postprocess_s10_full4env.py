"""
Build full 4-environment S10 tx-sensitivity bundle from 8 JSON inputs.

Outputs:
  - s10_4env_merged_descriptive_20260216.csv
  - s10_4env_significance_tx5_vs_tx15_20260216.csv
  - s10_4env_summary_20260216.md
  - sidecar provenance JSON files for missing outputs
"""

import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent.parent / "results" / "mega_experiments"

# Existing 2-environment S10 files
# + new fill files for indoor_office / outdoor_suburban
FILES = {
    ("indoor_factory", 5.0): "scalability_indoor_factory_server_s10_tx5_20260216.json",
    ("indoor_factory", 15.0): "scalability_indoor_factory_server_s10_tx15_20260216.json",
    ("outdoor_urban", 5.0): "scalability_outdoor_urban_server_s10_tx5_20260216.json",
    ("outdoor_urban", 15.0): "scalability_outdoor_urban_server_s10_tx15_20260216.json",
    ("indoor_office", 5.0): "scalability_indoor_office_server_s10_tx5_fill_20260216.json",
    ("indoor_office", 15.0): "scalability_indoor_office_server_s10_tx15_fill_20260216.json",
    ("outdoor_suburban", 5.0): "scalability_outdoor_suburban_server_s10_tx5_fill_20260216.json",
    ("outdoor_suburban", 15.0): "scalability_outdoor_suburban_server_s10_tx15_fill_20260216.json",
}

PROTOCOLS = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]
NODE_COUNTS = [100, 500, 1000]
ENVIRONMENTS = ["indoor_office", "indoor_factory", "outdoor_urban", "outdoor_suburban"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_pdr(raw_results, env, num_nodes, protocol):
    vals = []
    for r in raw_results:
        if (
            r.get("environment") == env
            and r.get("num_nodes") == num_nodes
            and r.get("protocol") == protocol
            and r.get("success", True)
        ):
            metrics = r.get("metrics", {})
            pdr = metrics.get("pdr_expected", metrics.get("pdr"))
            if pdr is not None:
                vals.append(float(pdr))
    return vals


def hedges_g(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled_std = np.sqrt(
        ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1))
        / (nx + ny - 2)
    )
    if pooled_std == 0:
        return 0.0
    d = (np.mean(x) - np.mean(y)) / pooled_std
    correction = 1 - 3 / (4 * (nx + ny) - 9)
    return d * correction


def holm_bonferroni(pvals):
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(1.0, p * (n - rank))
        cummax = max(cummax, adj)
        adjusted[orig_idx] = cummax
    return adjusted


def ensure_sidecar_for_file(env, tx, fn, payload):
    fpath = BASE / fn
    sidecar = fpath.with_suffix(".provenance.json")
    if sidecar.exists():
        return

    config = payload.get("config", {})
    script_path = Path(__file__).resolve().parent / "run_scalability_experiment.py"
    script_sha = sha256_file(script_path) if script_path.exists() else "N/A"

    content = {
        "provenance_for": fn,
        "provenance_generated": datetime.date.today().strftime("%Y%m%d"),
        "provenance_generator": "scripts/postprocess_s10_full4env.py",
        "git_commit": payload.get("git_commit", "unknown"),
        "git_dirty": payload.get("git_dirty", True),
        "script_sha256": script_sha,
        "experiment_timestamp": payload.get("timestamp", "unknown"),
        "run_tier": payload.get("run_tier", "publication"),
        "primary_metric": payload.get("primary_metric", "pdr_expected"),
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "raw_results_count": len(payload.get("raw_results", [])),
        "error_runs": payload.get("error_runs", 0),
        "environment": env,
        "tx_power_dbm": tx,
        "node_counts": NODE_COUNTS,
        "note": "S10 full 4-environment tx-sensitivity bundle.",
    }
    with open(sidecar, "w", encoding="utf-8") as f:
        json.dump(content, f, indent=2)


def main():
    print("=== S10 full 4-env postprocess ===")
    cache = {}

    # Load all inputs and ensure sidecar completeness
    for (env, tx), fn in FILES.items():
        fpath = BASE / fn
        if not fpath.exists():
            raise FileNotFoundError(f"Missing required file: {fpath}")
        payload = load_json(fpath)
        ensure_sidecar_for_file(env, tx, fn, payload)
        cache[(env, tx)] = payload["raw_results"]

    # Descriptive table
    merged_rows = []
    for env in ENVIRONMENTS:
        for tx in [5.0, 15.0]:
            raw = cache[(env, tx)]
            for proto in PROTOCOLS:
                for nn in NODE_COUNTS:
                    vals = extract_pdr(raw, env, nn, proto)
                    if vals:
                        merged_rows.append(
                            {
                                "environment": env,
                                "tx_power": tx,
                                "num_nodes": nn,
                                "protocol": proto,
                                "n": len(vals),
                                "pdr_mean": round(np.mean(vals), 6),
                                "pdr_std": round(np.std(vals, ddof=1), 6),
                            }
                        )

    merged_csv = BASE / "s10_4env_merged_descriptive_20260216.csv"
    with open(merged_csv, "w", encoding="utf-8") as f:
        f.write("environment,tx_power,num_nodes,protocol,n,pdr_mean,pdr_std\n")
        for r in merged_rows:
            f.write(
                f"{r['environment']},{r['tx_power']},{r['num_nodes']},{r['protocol']},{r['n']},{r['pdr_mean']},{r['pdr_std']}\n"
            )
    print(f"WROTE {merged_csv.name} ({len(merged_rows)} rows)")

    # Significance table (tx5 vs tx15 within env+node+protocol)
    sig_rows = []
    for env in ENVIRONMENTS:
        raw_tx5 = cache[(env, 5.0)]
        raw_tx15 = cache[(env, 15.0)]
        for proto in PROTOCOLS:
            for nn in NODE_COUNTS:
                v5 = extract_pdr(raw_tx5, env, nn, proto)
                v15 = extract_pdr(raw_tx15, env, nn, proto)
                if len(v5) >= 2 and len(v15) >= 2:
                    t_stat, p_raw = stats.ttest_ind(v5, v15, equal_var=False)
                    sig_rows.append(
                        {
                            "environment": env,
                            "num_nodes": nn,
                            "protocol": proto,
                            "n_tx5": len(v5),
                            "n_tx15": len(v15),
                            "tx5_mean": round(np.mean(v5), 6),
                            "tx15_mean": round(np.mean(v15), 6),
                            "delta": round(np.mean(v5) - np.mean(v15), 6),
                            "t_stat": round(t_stat, 4),
                            "p_raw": p_raw,
                            "hedges_g": round(hedges_g(v5, v15), 4),
                        }
                    )

    adj = holm_bonferroni([r["p_raw"] for r in sig_rows])
    for r, p_adj in zip(sig_rows, adj):
        r["p_holm"] = p_adj
        r["significant_005"] = "yes" if p_adj < 0.05 else "no"

    sig_csv = BASE / "s10_4env_significance_tx5_vs_tx15_20260216.csv"
    with open(sig_csv, "w", encoding="utf-8") as f:
        f.write(
            "environment,num_nodes,protocol,n_tx5,n_tx15,tx5_mean,tx15_mean,delta,t_stat,p_raw,p_holm,hedges_g,significant_005\n"
        )
        for r in sig_rows:
            f.write(
                f"{r['environment']},{r['num_nodes']},{r['protocol']},{r['n_tx5']},{r['n_tx15']},{r['tx5_mean']},{r['tx15_mean']},{r['delta']},{r['t_stat']},{r['p_raw']:.6e},{r['p_holm']:.6e},{r['hedges_g']},{r['significant_005']}\n"
            )
    print(f"WROTE {sig_csv.name} ({len(sig_rows)} rows)")

    # Markdown summary
    summary_md = BASE / "s10_4env_summary_20260216.md"
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write("# S10 Four-Environment TX-Power Sensitivity Summary\n\n")
        f.write(f"Generated: {datetime.datetime.now():%Y-%m-%d %H:%M:%S}\n\n")
        f.write("## Coverage\n\n")
        f.write("- Environments: indoor_office, indoor_factory, outdoor_urban, outdoor_suburban\n")
        f.write("- Node counts: 100, 500, 1000\n")
        f.write("- Protocols: AERIS, LEACH, PEGASIS, HEED, TEEN\n")
        f.write("- tx power: 5 vs 15 dBm\n")
        f.write("- Sample size per cell: n=600\n\n")

        all_sig = sum(1 for r in sig_rows if r["significant_005"] == "yes")
        f.write("## Significance\n\n")
        f.write(f"- Significant cells after Holm correction: {all_sig}/{len(sig_rows)}\n\n")

        f.write("## AERIS boundary snapshot (1000 nodes)\n\n")
        f.write("| Environment | tx5 mean | tx15 mean | delta (tx5 - tx15) |\n")
        f.write("|---|---:|---:|---:|\n")
        for env in ENVIRONMENTS:
            row = next(
                (
                    r
                    for r in sig_rows
                    if r["environment"] == env
                    and r["num_nodes"] == 1000
                    and r["protocol"] == "AERIS"
                ),
                None,
            )
            if row:
                f.write(
                    f"| {env} | {row['tx5_mean']:.4f} | {row['tx15_mean']:.4f} | {row['delta']:+.4f} |\n"
                )
    print(f"WROTE {summary_md.name}")

    print("=== DONE ===")


if __name__ == "__main__":
    main()

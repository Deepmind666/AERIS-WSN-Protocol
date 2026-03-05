"""S9 Rigor Patch Bundle 后处理脚本
生成: 4 sidecar + merged CSV + delta CSV + significance CSV
"""
import json, hashlib, sys, os, datetime
import numpy as np
from scipy import stats
from pathlib import Path

BASE = Path("c:/AERIS-WSN-Protocol/results/mega_experiments")

FILES = {
    ("indoor_office", "patch"): "scalability_indoor_office_server_s9_patch_20260216.json",
    ("outdoor_suburban", "patch"): "scalability_outdoor_suburban_server_s9_patch_20260216.json",
    ("indoor_office", "control"): "scalability_indoor_office_server_s9_control_20260216.json",
    ("outdoor_suburban", "control"): "scalability_outdoor_suburban_server_s9_control_20260216.json",
}

PROTOCOLS = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]
NODE_COUNTS = [100, 200, 300, 500, 800, 1000]

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def load_json(path):
    with open(path) as f:
        return json.load(f)

def extract_pdr(raw_results, env, num_nodes, protocol):
    vals = []
    for r in raw_results:
        if (r.get("environment") == env and
            r.get("num_nodes") == num_nodes and
            r.get("protocol") == protocol and
            r.get("success", True)):
            m = r.get("metrics", {})
            pdr = m.get("pdr_expected", m.get("pdr"))
            if pdr is not None:
                vals.append(pdr)
    return vals

def hedges_g(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    pooled_std = np.sqrt(((nx-1)*np.var(x, ddof=1) + (ny-1)*np.var(y, ddof=1)) / (nx+ny-2))
    if pooled_std == 0:
        return 0.0
    d = (np.mean(x) - np.mean(y)) / pooled_std
    correction = 1 - 3 / (4*(nx+ny) - 9)
    return d * correction

def holm_bonferroni(pvals):
    """Return adjusted p-values using Holm-Bonferroni method."""
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (n - rank)
        adj = min(adj, 1.0)
        cummax = max(cummax, adj)
        adjusted[orig_idx] = cummax
    return adjusted

# ── Step 1: Generate sidecars ──
print("=== Step 1: Generating sidecars ===")
script_path = Path("c:/AERIS-WSN-Protocol/scripts/run_scalability_experiment.py")
script_sha = sha256_file(script_path) if script_path.exists() else "N/A"

for (env, mode), fn in FILES.items():
    fpath = BASE / fn
    d = load_json(fpath)
    config = d.get("config", {})

    sidecar = {
        "provenance_for": fn,
        "provenance_generated": datetime.date.today().strftime("%Y%m%d"),
        "provenance_generator": "scripts/postprocess_s9_bundle.py",
        "git_commit": d.get("git_commit", "unknown"),
        "git_dirty": d.get("git_dirty", True),
        "script_sha256": script_sha,
        "experiment_timestamp": d.get("timestamp", "unknown"),
        "run_tier": d.get("run_tier", "publication"),
        "primary_metric": d.get("primary_metric", "pdr_expected"),
        "config_hash": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "raw_results_count": len(d.get("raw_results", [])),
        "error_runs": d.get("error_runs", 0),
        "environment": env,
        "mode": mode,
        "node_counts": NODE_COUNTS,
        "mac_collision": mode == "patch",
        "multihop_relay": mode == "patch",
        "note": f"S9 rigor patch bundle, {mode} run for {env}."
    }

    out = fpath.with_suffix(".provenance.json")
    with open(out, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"  {out.name}")

# ── Step 2: Merged comparison CSV ──
print("\n=== Step 2: Generating merged comparison CSV ===")
merged_rows = []
data_cache = {}

for (env, mode), fn in FILES.items():
    d = load_json(BASE / fn)
    raw = d["raw_results"]
    data_cache[(env, mode)] = raw

    for proto in PROTOCOLS:
        for nn in NODE_COUNTS:
            vals = extract_pdr(raw, env, nn, proto)
            if vals:
                merged_rows.append({
                    "environment": env,
                    "num_nodes": nn,
                    "protocol": proto,
                    "mode": mode,
                    "n": len(vals),
                    "pdr_mean": round(np.mean(vals), 6),
                    "pdr_std": round(np.std(vals, ddof=1), 6),
                })

merged_csv = BASE / "s9_merged_comparison_20260216.csv"
with open(merged_csv, "w") as f:
    f.write("environment,num_nodes,protocol,mode,n,pdr_mean,pdr_std\n")
    for r in merged_rows:
        f.write(f"{r['environment']},{r['num_nodes']},{r['protocol']},{r['mode']},{r['n']},{r['pdr_mean']},{r['pdr_std']}\n")
print(f"  {merged_csv.name} ({len(merged_rows)} rows)")

# ── Step 3: Delta CSV ──
print("\n=== Step 3: Generating delta CSV ===")
delta_rows = []

for env in ["indoor_office", "outdoor_suburban"]:
    raw_patch = data_cache.get((env, "patch"), [])
    raw_ctrl = data_cache.get((env, "control"), [])

    for proto in PROTOCOLS:
        for nn in NODE_COUNTS:
            v_patch = extract_pdr(raw_patch, env, nn, proto)
            v_ctrl = extract_pdr(raw_ctrl, env, nn, proto)
            if v_patch and v_ctrl:
                delta_rows.append({
                    "environment": env,
                    "num_nodes": nn,
                    "protocol": proto,
                    "n_patch": len(v_patch),
                    "n_control": len(v_ctrl),
                    "pdr_patch_mean": round(np.mean(v_patch), 6),
                    "pdr_control_mean": round(np.mean(v_ctrl), 6),
                    "delta": round(np.mean(v_patch) - np.mean(v_ctrl), 6),
                })

delta_csv = BASE / "s9_delta_patch_vs_control_20260216.csv"
with open(delta_csv, "w") as f:
    f.write("environment,num_nodes,protocol,n_patch,n_control,pdr_patch_mean,pdr_control_mean,delta\n")
    for r in delta_rows:
        f.write(f"{r['environment']},{r['num_nodes']},{r['protocol']},{r['n_patch']},{r['n_control']},{r['pdr_patch_mean']},{r['pdr_control_mean']},{r['delta']}\n")
print(f"  {delta_csv.name} ({len(delta_rows)} rows)")

# ── Step 4: Significance CSV ──
print("\n=== Step 4: Generating significance CSV ===")
sig_rows = []

for env in ["indoor_office", "outdoor_suburban"]:
    raw_patch = data_cache.get((env, "patch"), [])
    raw_ctrl = data_cache.get((env, "control"), [])

    for proto in PROTOCOLS:
        for nn in NODE_COUNTS:
            v_patch = extract_pdr(raw_patch, env, nn, proto)
            v_ctrl = extract_pdr(raw_ctrl, env, nn, proto)
            if len(v_patch) >= 2 and len(v_ctrl) >= 2:
                t_stat, p_val = stats.ttest_ind(v_patch, v_ctrl, equal_var=False)
                g = hedges_g(v_patch, v_ctrl)
                sig_rows.append({
                    "environment": env,
                    "num_nodes": nn,
                    "protocol": proto,
                    "t_stat": round(t_stat, 4),
                    "p_raw": p_val,
                    "hedges_g": round(g, 4),
                })

# Holm-Bonferroni correction
raw_pvals = [r["p_raw"] for r in sig_rows]
adj_pvals = holm_bonferroni(raw_pvals)
for r, p_adj in zip(sig_rows, adj_pvals):
    r["p_holm"] = p_adj
    r["significant_005"] = "yes" if p_adj < 0.05 else "no"

sig_csv = BASE / "s9_significance_patch_vs_control_20260216.csv"
with open(sig_csv, "w") as f:
    f.write("environment,num_nodes,protocol,t_stat,p_raw,p_holm,hedges_g,significant_005\n")
    for r in sig_rows:
        f.write(f"{r['environment']},{r['num_nodes']},{r['protocol']},{r['t_stat']},{r['p_raw']:.6e},{r['p_holm']:.6e},{r['hedges_g']},{r['significant_005']}\n")
print(f"  {sig_csv.name} ({len(sig_rows)} rows)")

print("\n=== ALL DONE ===")

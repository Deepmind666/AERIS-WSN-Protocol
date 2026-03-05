"""
S11 postprocess: build matched patch(n=1000) vs control(n=1000) tables.

Outputs:
  - s11_matched_4env_patch_vs_control_20260217_merged.csv (240 rows)
  - s11_matched_4env_patch_vs_control_20260217_delta.csv (120 rows)
  - s11_matched_4env_patch_vs_control_20260217_significance.csv (120 rows)
  - .provenance.json for each control JSON
"""

import datetime, hashlib, json
from pathlib import Path
import numpy as np
from scipy import stats

BASE = Path(__file__).resolve().parent.parent / "results" / "mega_experiments"

ENVS = ["indoor_office", "indoor_factory", "outdoor_urban", "outdoor_suburban"]
PROTOCOLS = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]
NODES = [100, 200, 300, 500, 800, 1000]

# S9 patch files (n=1000, mac-collision + multihop-relay)
PATCH_FILES = {
    "indoor_office": "scalability_indoor_office_server_s9_patch_20260216.json",
    "indoor_factory": "scalability_indoor_factory_local_s9_20260216_023118.json",
    "outdoor_urban": "scalability_outdoor_urban_local_s9_20260216_023118.json",
    "outdoor_suburban": "scalability_outdoor_suburban_server_s9_patch_20260216.json",
}

# S11 control files (n=1000, no mac-collision, no multihop-relay)
CONTROL_FILES = {
    "indoor_office": "scalability_indoor_office_server_s11_control_20260217.json",
    "indoor_factory": "scalability_indoor_factory_server_s11_control_20260217.json",
    "outdoor_urban": "scalability_outdoor_urban_server_s11_control_20260217.json",
    "outdoor_suburban": "scalability_outdoor_suburban_server_s11_control_20260217.json",
}


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
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    adjusted = [0.0] * n
    cummax = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = min(p * (n - rank), 1.0)
        cummax = max(cummax, adj)
        adjusted[orig_idx] = cummax
    return adjusted


# ── Step 0: Validate inputs ──
print("=== Step 0: Validating input files ===")
for env in ENVS:
    for label, fmap in [("patch", PATCH_FILES), ("control", CONTROL_FILES)]:
        p = BASE / fmap[env]
        if not p.exists():
            raise FileNotFoundError(f"Missing {label} file: {p}")
        print(f"  OK {label:7s} {env}: {fmap[env]}")

# ── Step 1: Provenance sidecars for S11 control files ──
print("\n=== Step 1: Generating provenance sidecars ===")
script_path = Path(__file__).resolve().parent / "run_scalability_experiment.py"
script_sha = sha256_file(script_path) if script_path.exists() else "N/A"

for env in ENVS:
    fpath = BASE / CONTROL_FILES[env]
    d = load_json(fpath)
    sidecar = {
        "provenance_for": CONTROL_FILES[env],
        "provenance_generated": datetime.date.today().strftime("%Y%m%d"),
        "provenance_generator": "scripts/postprocess_s11.py",
        "git_commit": d.get("git_commit", "unknown"),
        "git_dirty": d.get("git_dirty", True),
        "script_sha256": script_sha,
        "experiment_timestamp": d.get("timestamp", "unknown"),
        "run_tier": d.get("run_tier", "publication"),
        "primary_metric": d.get("primary_metric", "pdr_expected"),
        "raw_results_count": len(d.get("raw_results", [])),
        "error_runs": d.get("error_runs", 0),
        "environment": env,
        "mode": "control",
        "mac_collision": False,
        "multihop_relay": False,
        "note": f"S11 control补洞, n=1000 matched to S9 patch."
    }
    out = fpath.with_suffix(".provenance.json")
    with open(out, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"  {out.name}")

# ── Step 2: Merged comparison CSV (240 rows) ──
print("\n=== Step 2: Generating merged CSV ===")
data_cache = {}
merged_rows = []

for env in ENVS:
    for mode, fmap in [("patch", PATCH_FILES), ("control", CONTROL_FILES)]:
        raw = load_json(BASE / fmap[env])["raw_results"]
        data_cache[(env, mode)] = raw
        for proto in PROTOCOLS:
            for nn in NODES:
                vals = extract_pdr(raw, env, nn, proto)
                if vals:
                    merged_rows.append(f"{env},{nn},{proto},{mode},{len(vals)},{np.mean(vals):.6f},{np.std(vals, ddof=1):.6f}")

merged_csv = BASE / "s11_matched_4env_patch_vs_control_20260217_merged.csv"
with open(merged_csv, "w") as f:
    f.write("environment,num_nodes,protocol,mode,n,pdr_mean,pdr_std\n")
    f.write("\n".join(merged_rows) + "\n")
print(f"  {merged_csv.name} ({len(merged_rows)} rows)")

# ── Step 3: Delta CSV (120 rows) ──
print("\n=== Step 3: Generating delta CSV ===")
delta_rows = []

for env in ENVS:
    raw_patch = data_cache[(env, "patch")]
    raw_ctrl = data_cache[(env, "control")]
    for proto in PROTOCOLS:
        for nn in NODES:
            vp = extract_pdr(raw_patch, env, nn, proto)
            vc = extract_pdr(raw_ctrl, env, nn, proto)
            if vp and vc:
                delta_rows.append(f"{env},{nn},{proto},{len(vp)},{len(vc)},"
                    f"{np.mean(vp):.6f},{np.mean(vc):.6f},{np.mean(vp)-np.mean(vc):.6f}")

delta_csv = BASE / "s11_matched_4env_patch_vs_control_20260217_delta.csv"
with open(delta_csv, "w") as f:
    f.write("environment,num_nodes,protocol,n_patch,n_control,pdr_patch_mean,pdr_control_mean,delta\n")
    f.write("\n".join(delta_rows) + "\n")
print(f"  {delta_csv.name} ({len(delta_rows)} rows)")

# ── Step 4: Significance CSV (120 rows) ──
print("\n=== Step 4: Generating significance CSV ===")
sig_rows = []

for env in ENVS:
    raw_patch = data_cache[(env, "patch")]
    raw_ctrl = data_cache[(env, "control")]
    for proto in PROTOCOLS:
        for nn in NODES:
            vp = extract_pdr(raw_patch, env, nn, proto)
            vc = extract_pdr(raw_ctrl, env, nn, proto)
            if len(vp) >= 2 and len(vc) >= 2:
                t_stat, p_val = stats.ttest_ind(vp, vc, equal_var=False)
                g = hedges_g(vp, vc)
                sig_rows.append({
                    "env": env, "nn": nn, "proto": proto,
                    "t": round(t_stat, 4), "p": p_val, "g": round(g, 4),
                })

raw_pvals = [r["p"] for r in sig_rows]
adj_pvals = holm_bonferroni(raw_pvals)
for r, pa in zip(sig_rows, adj_pvals):
    r["p_holm"] = pa
    r["sig"] = "yes" if pa < 0.05 else "no"

sig_csv = BASE / "s11_matched_4env_patch_vs_control_20260217_significance.csv"
with open(sig_csv, "w") as f:
    f.write("environment,num_nodes,protocol,t_stat,p_raw,p_holm,hedges_g,significant_005\n")
    for r in sig_rows:
        f.write(f"{r['env']},{r['nn']},{r['proto']},{r['t']},{r['p']:.6e},{r['p_holm']:.6e},{r['g']},{r['sig']}\n")
print(f"  {sig_csv.name} ({len(sig_rows)} rows)")

# ── Summary ──
print(f"\n=== S11 POSTPROCESS DONE ===")
print(f"  merged:       {len(merged_rows)} rows (expect 240)")
print(f"  delta:        {len(delta_rows)} rows (expect 120)")
print(f"  significance: {len(sig_rows)} rows (expect 120)")
assert len(merged_rows) == 240, f"merged rows {len(merged_rows)} != 240"
assert len(delta_rows) == 120, f"delta rows {len(delta_rows)} != 120"
assert len(sig_rows) == 120, f"significance rows {len(sig_rows)} != 120"
print("  ALL ASSERTIONS PASSED")

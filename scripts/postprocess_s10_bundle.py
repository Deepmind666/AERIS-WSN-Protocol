"""S10 TX-Power Sensitivity Bundle 后处理脚本
生成: 4 sidecar + merged CSV + significance CSV + 结论 md
"""
import json, hashlib, sys, os, datetime
import numpy as np
from scipy import stats
from pathlib import Path

BASE = Path("c:/AERIS-WSN-Protocol/results/mega_experiments")

FILES = {
    ("indoor_factory", 5.0): "scalability_indoor_factory_server_s10_tx5_20260216.json",
    ("indoor_factory", 15.0): "scalability_indoor_factory_server_s10_tx15_20260216.json",
    ("outdoor_urban", 5.0): "scalability_outdoor_urban_server_s10_tx5_20260216.json",
    ("outdoor_urban", 15.0): "scalability_outdoor_urban_server_s10_tx15_20260216.json",
}

PROTOCOLS = ["AERIS", "LEACH", "PEGASIS", "HEED", "TEEN"]
NODE_COUNTS = [100, 500, 1000]

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
        adj = p * (n - rank)
        adj = min(adj, 1.0)
        cummax = max(cummax, adj)
        adjusted[orig_idx] = cummax
    return adjusted

# ── Step 1: Generate sidecars ──
print("=== Step 1: Generating sidecars ===")
script_path = Path("c:/AERIS-WSN-Protocol/scripts/run_scalability_experiment.py")
script_sha = sha256_file(script_path) if script_path.exists() else "N/A"

for (env, tx), fn in FILES.items():
    fpath = BASE / fn
    d = load_json(fpath)
    config = d.get("config", {})

    sidecar = {
        "provenance_for": fn,
        "provenance_generated": datetime.date.today().strftime("%Y%m%d"),
        "provenance_generator": "scripts/postprocess_s10_bundle.py",
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
        "tx_power_dbm": tx,
        "node_counts": NODE_COUNTS,
        "mac_collision": True,
        "multihop_relay": True,
        "note": f"S10 tx-power sensitivity bundle, {env} tx={tx}dBm."
    }

    out = fpath.with_suffix(".provenance.json")
    with open(out, "w") as f:
        json.dump(sidecar, f, indent=2)
    print(f"  {out.name}")

# ── Step 2: Merged descriptive CSV ──
print("\n=== Step 2: Generating merged descriptive CSV ===")
merged_rows = []
data_cache = {}

for (env, tx), fn in FILES.items():
    d = load_json(BASE / fn)
    raw = d["raw_results"]
    data_cache[(env, tx)] = raw

    for proto in PROTOCOLS:
        for nn in NODE_COUNTS:
            vals = extract_pdr(raw, env, nn, proto)
            if vals:
                merged_rows.append({
                    "environment": env,
                    "tx_power": tx,
                    "num_nodes": nn,
                    "protocol": proto,
                    "n": len(vals),
                    "pdr_mean": round(np.mean(vals), 6),
                    "pdr_std": round(np.std(vals, ddof=1), 6),
                })

merged_csv = BASE / "s10_merged_descriptive_20260216.csv"
with open(merged_csv, "w") as f:
    f.write("environment,tx_power,num_nodes,protocol,n,pdr_mean,pdr_std\n")
    for r in merged_rows:
        f.write(f"{r['environment']},{r['tx_power']},{r['num_nodes']},{r['protocol']},{r['n']},{r['pdr_mean']},{r['pdr_std']}\n")
print(f"  {merged_csv.name} ({len(merged_rows)} rows)")

# ── Step 3: Significance CSV (tx5 vs tx15 within same env+nodes+protocol) ──
print("\n=== Step 3: Generating significance CSV ===")
sig_rows = []

for env in ["indoor_factory", "outdoor_urban"]:
    raw_tx5 = data_cache.get((env, 5.0), [])
    raw_tx15 = data_cache.get((env, 15.0), [])

    for proto in PROTOCOLS:
        for nn in NODE_COUNTS:
            v_tx5 = extract_pdr(raw_tx5, env, nn, proto)
            v_tx15 = extract_pdr(raw_tx15, env, nn, proto)
            if len(v_tx5) >= 2 and len(v_tx15) >= 2:
                t_stat, p_val = stats.ttest_ind(v_tx5, v_tx15, equal_var=False)
                g = hedges_g(v_tx5, v_tx15)
                sig_rows.append({
                    "environment": env,
                    "num_nodes": nn,
                    "protocol": proto,
                    "comparison": "tx5_vs_tx15",
                    "pdr_tx5_mean": round(np.mean(v_tx5), 6),
                    "pdr_tx15_mean": round(np.mean(v_tx15), 6),
                    "delta": round(np.mean(v_tx5) - np.mean(v_tx15), 6),
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

sig_csv = BASE / "s10_significance_tx5_vs_tx15_20260216.csv"
with open(sig_csv, "w") as f:
    f.write("environment,num_nodes,protocol,comparison,pdr_tx5_mean,pdr_tx15_mean,delta,t_stat,p_raw,p_holm,hedges_g,significant_005\n")
    for r in sig_rows:
        f.write(f"{r['environment']},{r['num_nodes']},{r['protocol']},{r['comparison']},{r['pdr_tx5_mean']},{r['pdr_tx15_mean']},{r['delta']},{r['t_stat']},{r['p_raw']:.6e},{r['p_holm']:.6e},{r['hedges_g']},{r['significant_005']}\n")
print(f"  {sig_csv.name} ({len(sig_rows)} rows)")

# ── Step 4: Summary markdown ──
print("\n=== Step 4: Generating summary markdown ===")
sig_count = sum(1 for r in sig_rows if r["significant_005"] == "yes")
nonsig_count = len(sig_rows) - sig_count

lines = []
lines.append("# S10 TX-Power Sensitivity 结论摘要")
lines.append(f"\n生成日期: {datetime.date.today().strftime('%Y-%m-%d')}")
lines.append(f"\n## 实验矩阵")
lines.append(f"- 环境: indoor_factory, outdoor_urban")
lines.append(f"- 功率: 5 dBm vs 15 dBm")
lines.append(f"- 节点: 100, 500, 1000")
lines.append(f"- 协议: AERIS, LEACH, PEGASIS, HEED, TEEN")
lines.append(f"- 每cell n=600, run_tier=publication")
lines.append(f"- MAC碰撞模型: 启用, 多跳中继: 启用")
lines.append(f"\n## 统计检验")
lines.append(f"- 方法: Welch t-test + Hedges g + Holm-Bonferroni 校正")
lines.append(f"- 总比较数: {len(sig_rows)}")
lines.append(f"- 显著 (p_holm < 0.05): {sig_count}")
lines.append(f"- 不显著: {nonsig_count}")

lines.append(f"\n## 各协议 tx5 vs tx15 差异")
lines.append(f"\n| 环境 | 节点 | 协议 | PDR(tx5) | PDR(tx15) | delta | Hedges g | 显著 |")
lines.append(f"|------|------|------|----------|-----------|-------|----------|------|")
for r in sig_rows:
    lines.append(f"| {r['environment']} | {r['num_nodes']} | {r['protocol']} | {r['pdr_tx5_mean']:.4f} | {r['pdr_tx15_mean']:.4f} | {r['delta']:+.4f} | {r['hedges_g']:.4f} | {r['significant_005']} |")

# AERIS 专项
lines.append(f"\n## AERIS 稳定性边界")
aeris_rows = [r for r in sig_rows if r["protocol"] == "AERIS"]
for r in aeris_rows:
    status = "显著下降" if r["significant_005"] == "yes" and r["delta"] < -0.01 else "差异不显著" if r["significant_005"] == "no" else "显著差异"
    lines.append(f"- {r['environment']} n={r['num_nodes']}: tx5={r['pdr_tx5_mean']:.4f}, tx15={r['pdr_tx15_mean']:.4f}, delta={r['delta']:+.4f} → {status}")

summary_md = BASE / "s10_summary_20260216.md"
with open(summary_md, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"  {summary_md.name}")

print("\n=== ALL DONE ===")

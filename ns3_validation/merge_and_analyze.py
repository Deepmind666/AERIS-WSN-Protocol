#!/usr/bin/env python3
"""Merge 24 NS-3 shard JSONs and run statistical analysis.

Output:
  - ns3_5proto_merged.json          : combined raw data
  - ns3_5proto_summary.json         : per-(protocol, env, nodes) mean/std/n
  - ns3_5proto_significance.json    : pairwise Welch-t + Hedges g + Holm-Bonferroni
"""

import json, glob, os, sys, math
from collections import defaultdict
from pathlib import Path

SHARD_DIR = Path(__file__).parent / "results" / "shards_5proto"
OUT_DIR   = Path(__file__).parent / "results"

# ── 1. Merge shards ─────────────────────────────────────────────────
def merge_shards():
    files = sorted(SHARD_DIR.glob("shard_*.json"))
    assert len(files) == 24, f"Expected 24 shards, found {len(files)}"
    all_exp = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        all_exp.extend(data["experiments"])
    merged = {
        "channel_model": {"type": "realistic_physics_based", "multi_environment": True},
        "total_experiments": len(all_exp),
        "experiments": all_exp,
    }
    out = OUT_DIR / "ns3_5proto_merged.json"
    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    print(f"[merge] {len(all_exp)} experiments → {out}")
    return all_exp


# ── 2. Summary statistics ───────────────────────────────────────────
def summarize(experiments):
    """Group by (protocol, environment, num_nodes) → mean, std, n for PDR."""
    buckets = defaultdict(list)
    for e in experiments:
        key = (e["protocol"], e["environment"], e["num_nodes"])
        buckets[key].append(e["pdr"])

    rows = []
    for (proto, env, nodes), pdrs in sorted(buckets.items()):
        n = len(pdrs)
        mean = sum(pdrs) / n
        var = sum((x - mean) ** 2 for x in pdrs) / (n - 1) if n > 1 else 0.0
        std = math.sqrt(var)
        rows.append({
            "protocol": proto, "environment": env, "num_nodes": nodes,
            "n": n, "pdr_mean": round(mean, 6), "pdr_std": round(std, 6),
        })

    out = OUT_DIR / "ns3_5proto_summary.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"[summary] {len(rows)} groups → {out}")
    return buckets


# ── 3. Statistical tests ────────────────────────────────────────────
def welch_t(m1, s1, n1, m2, s2, n2):
    """Welch's t-statistic and approximate df."""
    se1, se2 = s1**2 / n1, s2**2 / n2
    se = se1 + se2
    if se < 1e-15:
        return 0.0, n1 + n2 - 2
    t = (m1 - m2) / math.sqrt(se)
    df = se**2 / (se1**2 / (n1 - 1) + se2**2 / (n2 - 1)) if (se1 + se2) > 0 else n1 + n2 - 2
    return t, max(df, 1.0)


def t_to_p_twosided(t_stat, df):
    """Approximate two-sided p-value using normal approx for large df."""
    # For df >= 30 (we have n=30), normal approx is adequate
    z = abs(t_stat)
    # Abramowitz & Stegun 26.2.17 approximation
    p = 0.5 * math.erfc(z / math.sqrt(2))
    return 2 * p  # two-sided


def hedges_g(m1, s1, n1, m2, s2, n2):
    """Hedges' g (bias-corrected standardized mean difference)."""
    sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2)
    sp = math.sqrt(sp2) if sp2 > 0 else 1e-15
    d = (m1 - m2) / sp
    # Hedges correction factor
    df = n1 + n2 - 2
    j = 1 - 3 / (4 * df - 1) if df > 1 else 1.0
    return d * j


def holm_bonferroni(p_values):
    """Holm-Bonferroni correction. Returns adjusted p-values."""
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj = p * (m - rank)
        adj = min(adj, 1.0)
        running_max = max(running_max, adj)
        adjusted[orig_idx] = running_max
    return adjusted


def run_significance(buckets):
    """Pairwise AERIS vs each baseline, per (env, nodes)."""
    MAIN_PROTOS = ["AERIS", "LEACH", "HEED", "PEGASIS", "TEEN"]
    ENVS = ["indoor_office", "indoor_factory", "outdoor_urban", "outdoor_suburban"]
    NODES = [50, 100, 150, 200, 300, 400, 550]

    comparisons = []
    for env in ENVS:
        for nodes in NODES:
            aeris_key = ("AERIS", env, nodes)
            aeris_pdrs = buckets.get(aeris_key, [])
            if not aeris_pdrs:
                continue
            n_a = len(aeris_pdrs)
            m_a = sum(aeris_pdrs) / n_a
            s_a = math.sqrt(sum((x - m_a)**2 for x in aeris_pdrs) / (n_a - 1)) if n_a > 1 else 0.0

            for baseline in ["LEACH", "HEED", "PEGASIS", "TEEN"]:
                bl_key = (baseline, env, nodes)
                bl_pdrs = buckets.get(bl_key, [])
                if not bl_pdrs:
                    continue
                n_b = len(bl_pdrs)
                m_b = sum(bl_pdrs) / n_b
                s_b = math.sqrt(sum((x - m_b)**2 for x in bl_pdrs) / (n_b - 1)) if n_b > 1 else 0.0

                t_stat, df = welch_t(m_a, s_a, n_a, m_b, s_b, n_b)
                p_val = t_to_p_twosided(t_stat, df)
                g = hedges_g(m_a, s_a, n_a, m_b, s_b, n_b)

                comparisons.append({
                    "environment": env, "num_nodes": nodes,
                    "protocol_a": "AERIS", "protocol_b": baseline,
                    "mean_a": round(m_a, 6), "mean_b": round(m_b, 6),
                    "std_a": round(s_a, 6), "std_b": round(s_b, 6),
                    "n_a": n_a, "n_b": n_b,
                    "welch_t": round(t_stat, 4), "df": round(df, 2),
                    "p_value": p_val, "hedges_g": round(g, 4),
                })

    # Holm-Bonferroni correction
    raw_ps = [c["p_value"] for c in comparisons]
    adj_ps = holm_bonferroni(raw_ps)
    for c, adj in zip(comparisons, adj_ps):
        c["p_adjusted"] = round(adj, 8)
        c["p_value"] = round(c["p_value"], 8)
        c["significant_005"] = adj < 0.05

    # Ablation comparisons: AERIS-FULL vs each variant
    ABLATION_VARIANTS = ["AERIS-noCAS", "AERIS-noFair", "AERIS-noGW"]
    ablation_comps = []
    for env in ENVS:
        full_key = ("AERIS-FULL", env, 100)  # ablation only at 100 nodes
        full_pdrs = buckets.get(full_key, [])
        if not full_pdrs:
            continue
        n_f = len(full_pdrs)
        m_f = sum(full_pdrs) / n_f
        s_f = math.sqrt(sum((x - m_f)**2 for x in full_pdrs) / (n_f - 1)) if n_f > 1 else 0.0

        for variant in ABLATION_VARIANTS:
            v_key = (variant, env, 100)
            v_pdrs = buckets.get(v_key, [])
            if not v_pdrs:
                continue
            n_v = len(v_pdrs)
            m_v = sum(v_pdrs) / n_v
            s_v = math.sqrt(sum((x - m_v)**2 for x in v_pdrs) / (n_v - 1)) if n_v > 1 else 0.0

            t_stat, df = welch_t(m_f, s_f, n_f, m_v, s_v, n_v)
            p_val = t_to_p_twosided(t_stat, df)
            g = hedges_g(m_f, s_f, n_f, m_v, s_v, n_v)

            ablation_comps.append({
                "environment": env,
                "protocol_a": "AERIS-FULL", "protocol_b": variant,
                "mean_a": round(m_f, 6), "mean_b": round(m_v, 6),
                "std_a": round(s_f, 6), "std_b": round(s_v, 6),
                "n_a": n_f, "n_b": n_v,
                "welch_t": round(t_stat, 4), "df": round(df, 2),
                "p_value": p_val, "hedges_g": round(g, 4),
            })

    if ablation_comps:
        abl_ps = [c["p_value"] for c in ablation_comps]
        abl_adj = holm_bonferroni(abl_ps)
        for c, adj in zip(ablation_comps, abl_adj):
            c["p_adjusted"] = round(adj, 8)
            c["p_value"] = round(c["p_value"], 8)
            c["significant_005"] = adj < 0.05

    result = {
        "method": "Welch t-test (two-sided) + Hedges g + Holm-Bonferroni",
        "main_comparisons": comparisons,
        "ablation_comparisons": ablation_comps,
        "total_main": len(comparisons),
        "total_ablation": len(ablation_comps),
        "significant_main": sum(1 for c in comparisons if c["significant_005"]),
        "significant_ablation": sum(1 for c in ablation_comps if c["significant_005"]),
    }

    out = OUT_DIR / "ns3_5proto_significance.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[significance] {len(comparisons)} main + {len(ablation_comps)} ablation → {out}")
    print(f"  Main significant (p_adj<0.05): {result['significant_main']}/{result['total_main']}")
    print(f"  Ablation significant: {result['significant_ablation']}/{result['total_ablation']}")
    return result


# ── 4. Quick sanity print ───────────────────────────────────────────
def print_overview(buckets):
    """Print mean PDR per protocol×env (averaged over node counts)."""
    PROTOS = ["AERIS", "LEACH", "HEED", "PEGASIS", "TEEN"]
    ENVS = ["indoor_office", "indoor_factory", "outdoor_urban", "outdoor_suburban"]
    print("\n=== Mean PDR by Protocol × Environment ===")
    header = f"{'Protocol':<10}" + "".join(f"{e:<22}" for e in ENVS)
    print(header)
    print("-" * len(header))
    for proto in PROTOS:
        row = f"{proto:<10}"
        for env in ENVS:
            pdrs = []
            for nodes in [50, 100, 150, 200, 300, 400, 550]:
                pdrs.extend(buckets.get((proto, env, nodes), []))
            if pdrs:
                row += f"{sum(pdrs)/len(pdrs):.4f} (n={len(pdrs):<4})     "
            else:
                row += f"{'N/A':<22}"
        print(row)


if __name__ == "__main__":
    experiments = merge_shards()
    buckets = summarize(experiments)
    print_overview(buckets)
    run_significance(buckets)

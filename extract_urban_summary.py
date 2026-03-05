import json, sys

path = r"C:\Users\sshuser\AERIS-WSN\results\mega_experiments\scalability_outdoor_urban_v50rigor_20260222_server.json"
d = json.load(open(path))
rs = d.get("raw_results", d.get("results", []))

# error count
err = len([r for r in rs if not r.get("success", True)])
print(f"total_entries: {len(rs)}")
print(f"error_runs: {err}")
print(f"run_tier: {d.get('run_tier','?')}")
print(f"primary_metric: {d.get('primary_metric','?')}")
print(f"git_commit: {d.get('git_commit','?')}")
print()

aeris = [r for r in rs if r.get("protocol") == "AERIS" and r.get("success", True)]
for n in [100, 500, 1000]:
    subset = [r for r in aeris if r["num_nodes"] == n]
    if subset:
        mean_pdr = sum(r["metrics"]["pdr_expected"] for r in subset) / len(subset)
        print(f"AERIS n={n}: PDR_mean={mean_pdr:.4f} (count={len(subset)})")
    else:
        print(f"AERIS n={n}: NO DATA")

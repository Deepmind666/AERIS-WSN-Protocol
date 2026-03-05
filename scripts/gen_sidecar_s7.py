"""Generate provenance sidecar files for S7 scalability JSONs."""
import json, hashlib, platform, sys, subprocess
from datetime import datetime
from pathlib import Path

repo = Path(r"C:/Users/sshuser/AERIS-WSN-Protocol")
mega = repo / "results" / "mega_experiments"
files = [
    "scalability_indoor_office_server_s7_20260211.json",
    "scalability_outdoor_suburban_server_s7_20260211.json",
]
try:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo), stderr=subprocess.DEVNULL
    ).decode().strip()[:8]
except Exception:
    commit = "unknown"

script_hash = hashlib.sha256(
    open(repo / "scripts" / "run_scalability_experiment.py", "rb").read()
).hexdigest()
ts = datetime.now().strftime("%Y%m%d")
count = 0
for fn in files:
    fp = mega / fn
    if not fp.exists():
        print(f"SKIP {fn}")
        continue
    d = json.load(open(fp, "r", encoding="utf-8-sig"))
    sc = {
        "provenance_for": fn,
        "provenance_generated": ts,
        "provenance_generator": "Claude Opus 4.6 (post-hoc sidecar)",
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "platform_machine": platform.machine(),
        "git_commit": d.get("git_commit", commit) or commit,
        "script_sha256": script_hash,
        "experiment_timestamp": d.get("timestamp", "unknown"),
        "run_tier": d.get("run_tier", "unknown"),
        "primary_metric": d.get("primary_metric", "unknown"),
        "config_hash": hashlib.sha256(
            json.dumps(d.get("config", {}), sort_keys=True, ensure_ascii=True, default=str).encode()
        ).hexdigest(),
        "raw_results_count": len(d.get("raw_results", [])),
        "error_runs": d.get("error_runs", 0),
        "environment": d.get("environment", "unknown"),
        "node_counts": d.get("config", {}).get("node_counts", []),
        "note": "Post-hoc sidecar generated after run completion.",
    }
    op = fp.with_suffix(".provenance.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(sc, f, indent=2, ensure_ascii=True)
    print(f"[OK] {op.name}")
    count += 1
print(f"[DONE] {count} sidecars generated")

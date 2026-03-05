#!/usr/bin/env python3
"""
Fast reality check for references cited in a TeX draft.

Method:
- Extract cited keys from draft.
- Parse corresponding BibTeX titles.
- Query Crossref by title and compute token overlap.

Output:
- docs/<prefix>_used_refs_fastcheck.csv
- docs/<prefix>_used_refs_fastcheck.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path


def token_set(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) >= 4}


def overlap(a: str, b: str) -> float:
    sa, sb = token_set(a), token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def field(block: str, name: str) -> str:
    m = re.search(rf"(?is)\b{name}\s*=\s*(\{{.*?\}}|\".*?\")", block)
    if not m:
        return ""
    v = m.group(1).strip()
    if (v.startswith("{") and v.endswith("}")) or (v.startswith('"') and v.endswith('"')):
        v = v[1:-1]
    return v.replace("{", "").replace("}", "").strip()


def crossref_title_query(title: str) -> tuple[str, str, float]:
    if not title:
        return ("no_title", "", 0.0)
    url = "https://api.crossref.org/works?rows=1&query.title=" + urllib.parse.quote(title)
    req = urllib.request.Request(url, headers={"User-Agent": "AERIS-reference-fastcheck/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            data = json.loads(r.read().decode("utf-8", errors="replace"))
        items = data.get("message", {}).get("items", [])
        if not items:
            return ("not_found", "", 0.0)
        mt = (items[0].get("title") or [""])[0]
        ov = overlap(title, mt)
        if ov >= 0.45:
            return ("probable_real", mt, ov)
        return ("low_conf_match", mt, ov)
    except Exception:
        return ("network_unverified", "", 0.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--bib", required=True)
    ap.add_argument("--prefix", default="20260214")
    args = ap.parse_args()

    draft = Path(args.draft)
    bib = Path(args.bib)
    tex = draft.read_text(encoding="utf-8")
    bibtxt = bib.read_text(encoding="utf-8")

    cited: list[str] = []
    for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex):
        for k in m.group(1).split(","):
            kk = k.strip()
            if kk and kk not in cited:
                cited.append(kk)

    starts = [m.start() for m in re.finditer(r"(?m)^@", bibtxt)] + [len(bibtxt)]
    blocks: dict[str, str] = {}
    for i in range(len(starts) - 1):
        b = bibtxt[starts[i] : starts[i + 1]]
        m = re.match(r"@\w+\{\s*([^,]+)\s*,", b)
        if m:
            blocks[m.group(1).strip()] = b

    rows = []
    for k in cited:
        if k not in blocks:
            rows.append(
                {
                    "key": k,
                    "status": "missing_in_bib",
                    "title": "",
                    "crossref_title": "",
                    "overlap": "0.000",
                }
            )
            continue
        title = field(blocks[k], "title")
        status, mt, ov = crossref_title_query(title)
        rows.append(
            {
                "key": k,
                "status": status,
                "title": title,
                "crossref_title": mt,
                "overlap": f"{ov:.3f}",
            }
        )

    out_csv = Path("docs") / f"{args.prefix}_used_refs_fastcheck.csv"
    out_md = Path("docs") / f"{args.prefix}_used_refs_fastcheck.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["key", "status", "title", "crossref_title", "overlap"])
        w.writeheader()
        w.writerows(rows)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Used References Fast Check\n\n")
        f.write(f"- draft: {draft}\n")
        f.write(f"- bib: {bib}\n")
        f.write(f"- cited keys: {len(cited)}\n\n")
        f.write("## Status counts\n\n")
        for k, v in sorted(counts.items()):
            f.write(f"- {k}: {v}\n")
        f.write("\n## Detail\n\n")
        f.write("| key | status | overlap |\n|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['key']} | {r['status']} | {r['overlap']} |\n")

    print(f"[fastcheck] csv={out_csv}")
    print(f"[fastcheck] md={out_md}")
    for k, v in sorted(counts.items()):
        print(f"[fastcheck] {k}={v}")


if __name__ == "__main__":
    main()

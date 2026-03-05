#!/usr/bin/env python3
"""
DOI-hard audit for references cited in a TeX draft.

Status policy:
- verified_doi: DOI exists and resolves; title overlap >= 0.35
- doi_title_mismatch: DOI resolves but title overlap low
- missing_doi_candidate: DOI missing in bib, Crossref query finds high-overlap candidate (>=0.60)
- missing_doi_no_candidate: DOI missing and no confident candidate
- unresolved_network: request timeout/error

Outputs:
- docs/<prefix>_used_refs_doi_hard.csv
- docs/<prefix>_used_refs_doi_hard.md
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path


def tokens(s: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", s.lower()) if len(t) >= 4}


def overlap(a: str, b: str) -> float:
    sa, sb = tokens(a), tokens(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def http_json(url: str, timeout: float = 6.0) -> dict | None:
    req = urllib.request.Request(url, headers={"User-Agent": "AERIS-DOI-hard-audit/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 200:
                return None
            return json.loads(r.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def bib_field(block: str, name: str) -> str:
    m = re.search(rf"(?is)\b{name}\s*=\s*(\{{.*?\}}|\".*?\")", block)
    if not m:
        return ""
    v = m.group(1).strip()
    if (v.startswith("{") and v.endswith("}")) or (v.startswith('"') and v.endswith('"')):
        v = v[1:-1]
    return v.replace("{", "").replace("}", "").strip()


def normalize_doi(v: str) -> str:
    return re.sub(r"(?i)^https?://(dx\.)?doi\.org/", "", v.strip())


def parse_cited_keys(tex: str) -> list[str]:
    keys: list[str] = []
    for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex):
        for k in m.group(1).split(","):
            kk = k.strip()
            if kk and kk not in keys:
                keys.append(kk)
    return keys


def parse_bib_blocks(bib: str) -> dict[str, str]:
    starts = [m.start() for m in re.finditer(r"(?m)^@", bib)] + [len(bib)]
    out: dict[str, str] = {}
    for i in range(len(starts) - 1):
        b = bib[starts[i] : starts[i + 1]]
        m = re.match(r"@\w+\{\s*([^,]+)\s*,", b)
        if m:
            out[m.group(1).strip()] = b
    return out


def check_doi(doi: str, title: str) -> tuple[str, str, float]:
    if not doi:
        return ("no_doi", "", 0.0)
    url = "https://api.crossref.org/works/" + urllib.parse.quote(doi, safe="")
    data = http_json(url)
    if not data or "message" not in data:
        return ("unresolved_network", "", 0.0)
    mt = ""
    t = data["message"].get("title", [])
    if isinstance(t, list) and t:
        mt = t[0]
    ov = overlap(title, mt)
    if ov >= 0.35:
        return ("verified_doi", mt, ov)
    return ("doi_title_mismatch", mt, ov)


def find_candidate_doi(title: str) -> tuple[str, str, float]:
    if not title:
        return ("missing_doi_no_candidate", "", 0.0)
    url = "https://api.crossref.org/works?rows=3&query.title=" + urllib.parse.quote(title)
    data = http_json(url)
    if not data or "message" not in data:
        return ("unresolved_network", "", 0.0)
    items = data["message"].get("items", [])
    best_doi = ""
    best_ov = 0.0
    for it in items:
        t = (it.get("title") or [""])[0] if isinstance(it.get("title"), list) else ""
        ov = overlap(title, t)
        doi = it.get("DOI", "") or ""
        if ov > best_ov and doi:
            best_ov = ov
            best_doi = doi
    if best_doi and best_ov >= 0.60:
        return ("missing_doi_candidate", best_doi, best_ov)
    return ("missing_doi_no_candidate", best_doi, best_ov)


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

    cited = parse_cited_keys(tex)
    blocks = parse_bib_blocks(bibtxt)

    rows: list[dict[str, str]] = []
    for key in cited:
        if key not in blocks:
            rows.append(
                {
                    "key": key,
                    "status": "missing_in_bib",
                    "doi_in_bib": "",
                    "candidate_doi": "",
                    "title_overlap": "0.000",
                    "title": "",
                    "notes": "citation key not found in bibliography.bib",
                }
            )
            continue
        b = blocks[key]
        title = bib_field(b, "title")
        doi = normalize_doi(bib_field(b, "doi"))
        if doi:
            st, _, ov = check_doi(doi, title)
            rows.append(
                {
                    "key": key,
                    "status": st,
                    "doi_in_bib": doi,
                    "candidate_doi": "",
                    "title_overlap": f"{ov:.3f}",
                    "title": title,
                    "notes": "",
                }
            )
        else:
            st, cdoi, ov = find_candidate_doi(title)
            rows.append(
                {
                    "key": key,
                    "status": st,
                    "doi_in_bib": "",
                    "candidate_doi": cdoi,
                    "title_overlap": f"{ov:.3f}",
                    "title": title,
                    "notes": "add candidate DOI manually after spot-check" if st == "missing_doi_candidate" else "",
                }
            )

    out_csv = Path("docs") / f"{args.prefix}_used_refs_doi_hard.csv"
    out_md = Path("docs") / f"{args.prefix}_used_refs_doi_hard.md"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "status",
                "doi_in_bib",
                "candidate_doi",
                "title_overlap",
                "title",
                "notes",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    with out_md.open("w", encoding="utf-8") as f:
        f.write("# Used References DOI-Hard Audit\n\n")
        f.write(f"- draft: {draft}\n")
        f.write(f"- bib: {bib}\n")
        f.write(f"- cited keys: {len(cited)}\n\n")
        f.write("## Status counts\n\n")
        for k, v in sorted(counts.items()):
            f.write(f"- {k}: {v}\n")
        f.write("\n## Detail\n\n")
        f.write("| key | status | doi_in_bib | candidate_doi | overlap |\n")
        f.write("|---|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['key']} | {r['status']} | {r['doi_in_bib'] or '-'} | {r['candidate_doi'] or '-'} | {r['title_overlap']} |\n"
            )

    print(f"[doi-hard] csv={out_csv}")
    print(f"[doi-hard] md={out_md}")
    for k, v in sorted(counts.items()):
        print(f"[doi-hard] {k}={v}")


if __name__ == "__main__":
    main()

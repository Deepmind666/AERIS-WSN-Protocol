#!/usr/bin/env python3
"""
Audit only the references cited by a target TeX draft.

Outputs:
  - docs/<prefix>_used_refs_audit.csv
  - docs/<prefix>_used_refs_audit.md

Verification policy:
1) If DOI exists, verify with Crossref /works/{doi}.
2) If DOI missing, try Crossref bibliographic query by title+author.
3) Mark status:
   - verified_doi
   - probable_crossref_match
   - unresolved
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class BibEntry:
    key: str
    entry_type: str
    title: str
    author: str
    year: str
    journal: str
    booktitle: str
    doi: str


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_cite_keys(tex: str) -> List[str]:
    keys: List[str] = []
    for m in re.finditer(r"\\cite[a-zA-Z]*\{([^}]+)\}", tex):
        chunk = m.group(1)
        for k in chunk.split(","):
            kk = k.strip()
            if kk and kk not in keys:
                keys.append(kk)
    return keys


def parse_bib_entries(bib_text: str) -> Dict[str, BibEntry]:
    entries: Dict[str, BibEntry] = {}
    # Split by entry starts while preserving delimiters.
    starts = [m.start() for m in re.finditer(r"(?m)^@", bib_text)]
    starts.append(len(bib_text))
    for i in range(len(starts) - 1):
        block = bib_text[starts[i] : starts[i + 1]]
        head = re.match(r"@(\w+)\s*\{\s*([^,]+)\s*,", block)
        if not head:
            continue
        entry_type = head.group(1).strip().lower()
        key = head.group(2).strip()
        fields = {}
        for fld in ("title", "author", "year", "journal", "booktitle", "doi"):
            mm = re.search(
                rf"(?is)\b{fld}\s*=\s*(\{{.*?\}}|\".*?\")\s*,",
                block,
            )
            if mm:
                raw = mm.group(1).strip()
                if raw.startswith("{") and raw.endswith("}"):
                    raw = raw[1:-1]
                elif raw.startswith('"') and raw.endswith('"'):
                    raw = raw[1:-1]
                raw = re.sub(r"\\[a-zA-Z]+\s*", "", raw)
                raw = raw.replace("{", "").replace("}", "").strip()
                fields[fld] = raw
            else:
                fields[fld] = ""

        entries[key] = BibEntry(
            key=key,
            entry_type=entry_type,
            title=fields["title"],
            author=fields["author"],
            year=fields["year"],
            journal=fields["journal"],
            booktitle=fields["booktitle"],
            doi=normalize_doi(fields["doi"]),
        )
    return entries


def normalize_doi(doi: str) -> str:
    d = doi.strip()
    d = re.sub(r"(?i)^https?://(dx\.)?doi\.org/", "", d).strip()
    return d


def token_set(text: str) -> set:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return {t for t in tokens if len(t) >= 3}


def title_overlap(a: str, b: str) -> float:
    sa = token_set(a)
    sb = token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def http_get_json(url: str, timeout: float = 1.2) -> Optional[dict]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "AERIS-WSN-Protocol-reference-audit/1.0 (academic audit)"
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                return None
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


def verify_by_doi(entry: BibEntry) -> Tuple[str, str, float]:
    if not entry.doi:
        return "no_doi", "", 0.0
    url = (
        "https://api.crossref.org/works/"
        + urllib.parse.quote(entry.doi, safe="")
    )
    data = http_get_json(url)
    if not data or "message" not in data:
        return "doi_unresolved", "", 0.0
    msg = data["message"]
    cr_title = (msg.get("title") or [""])[0] if isinstance(msg.get("title"), list) else ""
    ov = title_overlap(entry.title, cr_title)
    if ov >= 0.35:
        return "verified_doi", cr_title, ov
    return "doi_title_mismatch", cr_title, ov


def verify_by_query(entry: BibEntry) -> Tuple[str, str, float]:
    query = " ".join(x for x in [entry.title, entry.author] if x).strip()
    if not query:
        return "unresolved", "", 0.0
    url = "https://api.crossref.org/works?rows=1&query.bibliographic=" + urllib.parse.quote(query)
    data = http_get_json(url)
    if not data or "message" not in data:
        return "unresolved", "", 0.0
    items = data["message"].get("items", [])
    if not items:
        return "unresolved", "", 0.0
    item = items[0]
    cr_title = (item.get("title") or [""])[0] if isinstance(item.get("title"), list) else ""
    ov = title_overlap(entry.title, cr_title)
    if ov >= 0.45:
        return "probable_crossref_match", cr_title, ov
    return "unresolved", cr_title, ov


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True, help="Path to .tex draft")
    ap.add_argument("--bib", default="for_submission/bibliography.bib", help="Path to bibliography.bib")
    ap.add_argument("--prefix", default="20260214", help="Output prefix under docs/")
    ap.add_argument(
        "--enable-query-fallback",
        action="store_true",
        help="If set, run Crossref bibliographic query for entries without a verified DOI.",
    )
    args = ap.parse_args()

    draft_path = Path(args.draft)
    bib_path = Path(args.bib)
    if not draft_path.is_absolute():
        draft_path = PROJECT_ROOT / draft_path
    if not bib_path.is_absolute():
        bib_path = PROJECT_ROOT / bib_path

    tex = read_text(draft_path)
    bib_text = read_text(bib_path)
    cited = extract_cite_keys(tex)
    entries = parse_bib_entries(bib_text)

    rows = []
    for k in cited:
        e = entries.get(k)
        if e is None:
            rows.append(
                {
                    "key": k,
                    "status": "missing_in_bib",
                    "doi": "",
                    "year": "",
                    "title": "",
                    "crossref_title": "",
                    "title_overlap": "0.000",
                    "notes": "citation key not found in bibliography.bib",
                }
            )
            continue

        status, cr_title, ov = verify_by_doi(e)
        if args.enable_query_fallback and status in {"no_doi", "doi_unresolved"}:
            status_q, cr_title_q, ov_q = verify_by_query(e)
            if status_q == "probable_crossref_match":
                status, cr_title, ov = status_q, cr_title_q, ov_q
            elif status == "no_doi":
                status = status_q
                cr_title, ov = cr_title_q, ov_q

        notes = ""
        if status == "doi_title_mismatch":
            notes = "DOI resolved, but title overlap is low; manual check required"
        elif status == "unresolved":
            notes = "not verified by DOI/Crossref query"

        rows.append(
            {
                "key": k,
                "status": status,
                "doi": e.doi,
                "year": e.year,
                "title": e.title,
                "crossref_title": cr_title,
                "title_overlap": f"{ov:.3f}",
                "notes": notes,
            }
        )

    docs = PROJECT_ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    csv_path = docs / f"{args.prefix}_used_refs_audit.csv"
    md_path = docs / f"{args.prefix}_used_refs_audit.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "key",
                "status",
                "doi",
                "year",
                "title",
                "crossref_title",
                "title_overlap",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# Used Reference Reality Audit\n\n")
        f.write(f"- Draft: {draft_path}\n")
        f.write(f"- Bibliography: {bib_path}\n")
        f.write(f"- Cited keys: {len(cited)}\n\n")
        f.write("## Status counts\n\n")
        for k, v in sorted(counts.items()):
            f.write(f"- {k}: {v}\n")
        f.write("\n## Detailed results\n\n")
        f.write("| key | status | doi | overlap |\n")
        f.write("|---|---|---|---|\n")
        for r in rows:
            f.write(
                f"| {r['key']} | {r['status']} | {r['doi'] or '-'} | {r['title_overlap']} |\n"
            )

    print(f"[audit] csv={csv_path}")
    print(f"[audit] md={md_path}")
    for k, v in sorted(counts.items()):
        print(f"[audit] {k}={v}")


if __name__ == "__main__":
    main()

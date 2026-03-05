#!/usr/bin/env python3
"""
Claim gate checker for manuscript files.

This script scans target manuscript files for:
1) forbidden claims (must be removed)
2) evidence-risk claims (must be justified or softened)

Usage:
  python scripts/check_claim_gates.py
  python scripts/check_claim_gates.py --out docs/claim_gate_report_YYYYMMDD.md
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TARGETS = [
    "for_submission/AERIS_APIN_Section1_Introduction.md",
    "for_submission/AERIS_APIN_Section2_RelatedWork.md",
    "for_submission/AERIS_APIN_Section3_SystemModel.md",
    "for_submission/AERIS_APIN_Section5_Experiments.md",
    "for_submission/AERIS_APIN_Section6_Results.md",
    "for_submission/AERIS_APIN_Section7_Discussion.md",
    "for_submission/AERIS_APIN_Section8_Conclusion.md",
    "for_submission/AERIS_Sensors_MDPI_Submission_Draft_20260213.tex",
]

# Forbidden according to claim gate policy.
FORBIDDEN_PATTERNS: List[Tuple[str, str]] = [
    (r"\b200 independent runs?\b", "forbidden: 200 independent runs"),
    (r"\b100%\s*PDR\b", "forbidden: absolute 100% PDR claim"),
    (r"\b110ms\b|\b2500ms\b|<\s*10ms|<\s*50ms", "forbidden: absolute latency number"),
    (r"\bTDA metric\b|\bTDA validated\b", "forbidden: TDA claim without publication evidence"),
    (r"\bApplied Intelligence\s*\(APIN\)\b", "forbidden: outdated target journal marker"),
    (
        r"\bnumerical equivalence completed\b",
        "forbidden: NS-3 numerical equivalence completion phrase",
    ),
]

# Needs extra evidence if kept as hard quantitative statements.
EVIDENCE_RISK_PATTERNS: List[Tuple[str, str]] = [
    (r"\b23KB\b", "risk: hard memory number without direct benchmark evidence"),
    (r"\bAUC\s*=\s*0?\.\d+\b", "risk: AUC claim needs direct dataset/script evidence"),
    (r"\b89\.2%\b", "risk: fixed predictive accuracy claim needs evidence chain"),
]


def iter_lines(path: Path) -> Iterable[Tuple[int, str]]:
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            yield i, line.rstrip("\n")


def scan_file(path: Path) -> Tuple[List[Tuple[int, str, str]], List[Tuple[int, str, str]]]:
    forbidden_hits: List[Tuple[int, str, str]] = []
    risk_hits: List[Tuple[int, str, str]] = []
    for lineno, line in iter_lines(path):
        for pattern, tag in FORBIDDEN_PATTERNS:
            if re.search(pattern, line, flags=re.IGNORECASE):
                forbidden_hits.append((lineno, tag, line))
        for pattern, tag in EVIDENCE_RISK_PATTERNS:
            if re.search(pattern, line, flags=re.IGNORECASE):
                risk_hits.append((lineno, tag, line))
    return forbidden_hits, risk_hits


def build_report(targets: List[Path]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append("# Claim Gate Report")
    lines.append("")
    lines.append(f"- Generated: {now}")
    lines.append(f"- Scope files: {len(targets)}")
    lines.append("")

    total_forbidden = 0
    total_risk = 0

    for p in targets:
        lines.append(f"## {p.as_posix()}")
        if not p.exists():
            lines.append("- Status: MISSING")
            lines.append("")
            continue

        forbidden_hits, risk_hits = scan_file(p)
        total_forbidden += len(forbidden_hits)
        total_risk += len(risk_hits)

        lines.append(f"- Forbidden hits: {len(forbidden_hits)}")
        lines.append(f"- Evidence-risk hits: {len(risk_hits)}")
        if forbidden_hits:
            lines.append("")
            lines.append("### Forbidden")
            for lineno, tag, text in forbidden_hits:
                lines.append(f"- L{lineno}: {tag}")
                lines.append(f"  - `{text}`")
        if risk_hits:
            lines.append("")
            lines.append("### Evidence Risk")
            for lineno, tag, text in risk_hits:
                lines.append(f"- L{lineno}: {tag}")
                lines.append(f"  - `{text}`")
        lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Total forbidden hits: {total_forbidden}")
    lines.append(f"- Total evidence-risk hits: {total_risk}")
    lines.append(
        "- Gate decision: FAIL if forbidden hits > 0; WARNING if forbidden=0 and evidence-risk>0; PASS otherwise."
    )
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check manuscript claim gates.")
    parser.add_argument(
        "--out",
        default="docs/claim_gate_report_latest.md",
        help="Output markdown path (project-relative).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=DEFAULT_TARGETS,
        help="Target files (project-relative).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    targets = [ROOT / f for f in args.files]
    report = build_report(targets)
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"[OK] claim gate report written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

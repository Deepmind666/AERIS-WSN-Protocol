#!/usr/bin/env python3
"""
Lightweight gate checker for the Sensors submission draft.

Checks:
1) Forbidden claims that are disallowed by project claim-gating.
2) Path pollution in manuscript body (internal paths/file names).
3) Scope reminders for sample-size wording.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DRAFT = PROJECT_ROOT / "for_submission" / "AERIS_Sensors_MDPI_Submission_Draft_20260213.tex"

FORBIDDEN_PATTERNS = [
    r"100% PDR",
    r"200 independent runs",
    r"2500ms",
    r"<10ms",
    r"TDA metric",
    r"numerical equivalence completed",
]

PATH_POLLUTION_PATTERNS = [
    r"C:\\",
    r"results/",
    r"scripts/",
    r"ns3_validation/",
    r"\.json",
    r"\.csv",
    r"file://",
]

SCOPE_HINT_PATTERNS = [
    r"n=30",
    r"n=550",
    r"n=650",
    r"trend-level",
]


def find_hits(text: str, pattern: str) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    regex = re.compile(pattern)
    for idx, line in enumerate(text.splitlines(), start=1):
        if regex.search(line):
            hits.append((idx, line.strip()))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate checker for Sensors draft.")
    parser.add_argument("--draft", type=str, default=str(DEFAULT_DRAFT), help="Path to .tex draft file")
    args = parser.parse_args()

    draft = Path(args.draft)
    if not draft.exists():
        print(f"ERROR: draft not found: {draft}")
        return 2

    text = draft.read_text(encoding="utf-8")

    print("== Sensors Draft Gate Check ==")
    print(f"Draft: {draft}")

    failed = False

    print("\n[1] Forbidden claim scan")
    for p in FORBIDDEN_PATTERNS:
        hits = find_hits(text, p)
        if hits:
            failed = True
            print(f"  FAIL pattern: {p}")
            for ln, line in hits[:5]:
                print(f"    L{ln}: {line}")
        else:
            print(f"  PASS pattern: {p}")

    print("\n[2] Path-pollution scan")
    path_failed = False
    for p in PATH_POLLUTION_PATTERNS:
        hits = find_hits(text, p)
        if hits:
            path_failed = True
            print(f"  HIT pattern: {p}")
            for ln, line in hits[:3]:
                print(f"    L{ln}: {line}")
    if not path_failed:
        print("  PASS: no internal path/file leakage found")

    print("\n[3] Scope-hint scan (informational)")
    for p in SCOPE_HINT_PATTERNS:
        hits = find_hits(text, p)
        print(f"  {p}: {len(hits)} hit(s)")

    if failed:
        print("\nGate result: FAIL (forbidden claims detected)")
        return 1

    print("\nGate result: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

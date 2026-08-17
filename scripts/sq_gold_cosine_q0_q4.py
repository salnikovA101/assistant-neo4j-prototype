#!/usr/bin/env python3
"""Backward-compatible wrapper: q0–q4 cosine report."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "sq_gold_cosine.py"),
        "--qa",
        "tests/qa_evidence_50.json",
        "--qids",
        "q0,q1,q2,q3,q4",
        "--out-stem",
        "sq_gold_cosine_q0_q4",
        "--title",
        "SQ × gold evidence cosine (q0–q4)",
    ]
    raise SystemExit(subprocess.call(cmd))

"""Stable portable edge keys (independent of Neo4j elementId)."""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WS_RE = re.compile(r"\s+")


def normalize_name(name: str | None) -> str:
    if not name:
        return ""
    text = unicodedata.normalize("NFKC", str(name)).strip().lower()
    text = _WS_RE.sub(" ", text)
    return text


def normalize_evidence(evidence: str | None) -> str:
    if not evidence:
        return ""
    text = unicodedata.normalize("NFKC", str(evidence)).strip().lower()
    text = _WS_RE.sub(" ", text)
    return text


def evidence_hash(evidence: str | None, length: int = 12) -> str:
    digest = hashlib.sha256(normalize_evidence(evidence).encode("utf-8")).hexdigest()
    return digest[:length]


def compute_edge_key(
    start_name: str | None,
    rel_type: str | None,
    end_name: str | None,
    chunk_id: str | None = None,
    evidence: str | None = None,
) -> str:
    parts = [
        normalize_name(start_name),
        (rel_type or "").strip().upper(),
        normalize_name(end_name),
        (chunk_id or "").strip(),
        evidence_hash(evidence),
    ]
    return "|".join(parts)


def parse_edge_key(edge_key: str) -> dict[str, str]:
    parts = edge_key.split("|")
    if len(parts) < 5:
        raise ValueError(f"Invalid edge_key: {edge_key!r}")
    return {
        "start": parts[0],
        "rel_type": parts[1],
        "end": parts[2],
        "chunk_id": parts[3],
        "evidence_hash": parts[4],
    }

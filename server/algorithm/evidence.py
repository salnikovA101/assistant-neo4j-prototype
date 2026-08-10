"""Evidence keys for novelty / anti-dupe across chains and hops."""

from __future__ import annotations

from server.algorithm.models import EdgeRecord


def evidence_key(evidence: str, chunk_id: str = "") -> str:
    """Stable key; empty evidence → '' (no constraint). Prefer text; chunk is backup."""
    ev = (evidence or "").strip()
    if ev:
        return ev
    cid = (chunk_id or "").strip()
    return cid


def edge_evidence_key(e: EdgeRecord) -> str:
    return evidence_key(e.evidence, e.chunk_id)

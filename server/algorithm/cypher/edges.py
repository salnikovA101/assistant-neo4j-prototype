"""Cypher for relationship ANN, induced bridges, and explorer search."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from neo4j import AsyncDriver

from server.algorithm.models import parse_confidence


def sanitize_vector_index_name(name: str) -> str:
    """Quote an index identifier returned by SHOW VECTOR INDEXES."""
    raw = str(name or "")
    if not raw.strip() or any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        raise ValueError(f"invalid vector index name: {name!r}")
    return f"`{raw.replace('`', '``')}`"


def relationship_ann_query(index_name: str, *, run_id: str = "") -> str:
    """S2 ANN Cypher with mandatory run_id filtering inside SEARCH."""
    idx = sanitize_vector_index_name(index_name)
    if not (run_id or "").strip():
        raise ValueError("run_id is required for relationship ANN")
    return f"""
    CYPHER 25
    MATCH ()-[r]->()
      SEARCH r IN (
        VECTOR INDEX {idx}
        FOR $embedding
        WHERE r.run_id = $run_id
        LIMIT $k
      ) SCORE AS score
    RETURN elementId(r) AS rid,
           type(r) AS rel_type,
           elementId(startNode(r)) AS start_id,
           elementId(endNode(r)) AS end_id,
           coalesce(startNode(r).name, '') AS start_name,
           coalesce(endNode(r).name, '') AS end_name,
           labels(startNode(r)) AS start_labels,
           labels(endNode(r)) AS end_labels,
           coalesce(r.chunk_id, '') AS chunk_id,
           coalesce(r.evidence, '') AS evidence,
           coalesce(r.run_id, '') AS run_id,
           score AS score
    """


def induced_bridges_query(*, run_id: str = "") -> str:
    if not (run_id or "").strip():
        raise ValueError("run_id is required for induced bridges")
    return f"""
UNWIND $node_ids AS nid
MATCH (n)-[r]-(m)
WHERE elementId(n) = nid
  AND elementId(m) IN $node_ids
  AND elementId(n) < elementId(m)
  AND NOT elementId(r) IN $exclude_ids
  AND r.run_id = $run_id
  AND r.evidence_embedding IS NOT NULL
WITH DISTINCT r
WITH r, vector.similarity.cosine(r.evidence_embedding, $sqVec) AS score
ORDER BY score DESC
LIMIT $limit
RETURN elementId(r) AS rid,
       type(r) AS rel_type,
       elementId(startNode(r)) AS start_id,
       elementId(endNode(r)) AS end_id,
       coalesce(startNode(r).name, '') AS start_name,
       coalesce(endNode(r).name, '') AS end_name,
       labels(startNode(r)) AS start_labels,
       labels(endNode(r)) AS end_labels,
       coalesce(r.chunk_id, '') AS chunk_id,
       coalesce(r.evidence, '') AS evidence,
       coalesce(r.source_file, '') AS source_file,
       r.confidence AS confidence,
       coalesce(r.run_id, '') AS run_id,
       score AS score
"""

FETCH_EDGE_PROPS = f"""
UNWIND $ids AS rid
MATCH ()-[r]->()
WHERE elementId(r) = rid
  AND r.run_id = $run_id
RETURN elementId(r) AS rid,
       type(r) AS rel_type,
       elementId(startNode(r)) AS start_id,
       elementId(endNode(r)) AS end_id,
       coalesce(startNode(r).name, '') AS start_name,
       coalesce(endNode(r).name, '') AS end_name,
       labels(startNode(r)) AS start_labels,
       labels(endNode(r)) AS end_labels,
       coalesce(r.chunk_id, '') AS chunk_id,
       coalesce(r.evidence, '') AS evidence,
       coalesce(r.source_file, '') AS source_file,
       r.confidence AS confidence,
       coalesce(r.run_id, '') AS run_id
"""


FETCH_EDGE_EVIDENCE = """
UNWIND $ids AS rid
MATCH ()-[r]->()
WHERE elementId(r) = rid
  AND r.run_id = $run_id
RETURN elementId(r) AS rid,
       coalesce(r.evidence, '') AS evidence,
       coalesce(r.chunk_id, '') AS chunk_id,
       coalesce(r.source_file, '') AS source_file,
       r.confidence AS confidence
"""


# Read-only hydration for UI graph; intentionally selects no embedding fields.
FETCH_VIZ_BY_EDGE_IDS = f"""
UNWIND $ids AS rid
MATCH (a)-[r]->(b)
WHERE elementId(r) = rid
  AND r.run_id = $run_id
RETURN elementId(r) AS id,
       type(r) AS type,
       elementId(a) AS from_id,
       elementId(b) AS to_id,
       coalesce(a.name, '') AS from_name,
       coalesce(b.name, '') AS to_name,
       labels(a) AS from_labels,
       labels(b) AS to_labels,
       a.leiden_community AS from_community,
       b.leiden_community AS to_community,
       coalesce(r.evidence, '') AS evidence,
       coalesce(r.chunk_id, '') AS chunk_id,
       coalesce(r.source_file, '') AS source_file,
       r.confidence AS confidence,
       coalesce(r.run_id, '') AS run_id
"""


async def fetch_edge_properties(
    driver: AsyncDriver, element_ids: Iterable[str], *, run_id: str
) -> list[dict[str, Any]]:
    ids = list({i for i in element_ids if i})
    if not ids:
        return []
    corpus_run_id = (run_id or "").strip()
    if not corpus_run_id:
        raise ValueError("run_id is required for edge hydration")
    batch_size = 400
    chunks = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]

    async def _one(chunk: list[str]) -> list[dict[str, Any]]:
        async with driver.session() as session:
            result = await session.run(
                FETCH_EDGE_PROPS, ids=chunk, run_id=corpus_run_id
            )
            return [dict(r) async for r in result]

    parts = await asyncio.gather(*[_one(c) for c in chunks])
    out: list[dict[str, Any]] = []
    for p in parts:
        out.extend(p)
    return out


async def fetch_edge_evidence(
    driver: AsyncDriver, element_ids: Iterable[str], *, run_id: str
) -> dict[str, dict[str, Any]]:
    ids = list({i for i in element_ids if i})
    if not ids:
        return {}
    corpus_run_id = (run_id or "").strip()
    if not corpus_run_id:
        raise ValueError("run_id is required for evidence hydration")
    batch_size = 400
    chunks = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]

    async def _one(chunk: list[str]) -> dict[str, dict[str, Any]]:
        partial: dict[str, dict[str, Any]] = {}
        async with driver.session() as session:
            result = await session.run(
                FETCH_EDGE_EVIDENCE, ids=chunk, run_id=corpus_run_id
            )
            async for r in result:
                partial[r["rid"]] = {
                    "evidence": r["evidence"] or "",
                    "chunk_id": r["chunk_id"] or "",
                    "source_file": r["source_file"] or "",
                    "confidence": parse_confidence(r["confidence"]),
                }
        return partial

    parts = await asyncio.gather(*[_one(c) for c in chunks])
    out: dict[str, dict[str, Any]] = {}
    for p in parts:
        out.update(p)
    return out


async def fetch_viz_edges(
    driver: AsyncDriver, element_ids: Iterable[str], *, run_id: str
) -> list[dict[str, Any]]:
    corpus_run_id = (run_id or "").strip()
    if not corpus_run_id:
        raise ValueError("run_id is required for graph hydration")
    ids = list({i for i in element_ids if i})
    if not ids:
        return []
    batch_size = 400
    chunks = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]

    async def _one(chunk: list[str]) -> list[dict[str, Any]]:
        async with driver.session() as session:
            result = await session.run(
                FETCH_VIZ_BY_EDGE_IDS, ids=chunk, run_id=corpus_run_id
            )
            return [dict(r) async for r in result]

    parts = await asyncio.gather(*[_one(c) for c in chunks])
    out: list[dict[str, Any]] = []
    for p in parts:
        out.extend(p)
    return out


async def fetch_induced_bridges_by_sim(
    driver: AsyncDriver,
    node_ids: Iterable[str],
    sq_vec: list[float],
    *,
    exclude_ids: Iterable[str] | None = None,
    limit: int = 200,
    run_id: str = "",
) -> list[dict[str, Any]]:
    """Induced bridges ranked by cosine to sq_vec; LIMIT applied in Cypher."""
    ids = list({i for i in node_ids if i})
    if len(ids) < 2 or limit <= 0 or not sq_vec:
        return []
    excl = list({i for i in (exclude_ids or []) if i})
    rid = (run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required for induced bridges")
    params: dict[str, Any] = {
        "node_ids": ids,
        "exclude_ids": excl,
        "sqVec": list(sq_vec),
        "limit": int(limit),
        "run_id": rid,
    }
    async with driver.session() as session:
        result = await session.run(
            induced_bridges_query(run_id=rid),
            **params,
        )
        return [dict(r) async for r in result]


async def query_relationship_ann(
    driver: AsyncDriver,
    index_name: str,
    embedding: list[float],
    top_k: int,
    *,
    run_id: str = "",
) -> list[dict[str, Any]]:
    rid = (run_id or "").strip()
    if not rid:
        raise ValueError("run_id is required for relationship ANN")
    cypher = relationship_ann_query(index_name, run_id=rid)
    params: dict[str, Any] = {
        "k": int(top_k),
        "embedding": embedding,
        "run_id": rid,
    }
    async with driver.session() as session:
        result = await session.run(cypher, **params)
        return [dict(r) async for r in result]


EVIDENCE_BY_CHUNKS = """
UNWIND $chunk_ids AS cid
MATCH ()-[r]->()
WHERE r.chunk_id = cid
  AND r.run_id = $run_id
RETURN type(r) AS rel_type,
       coalesce(startNode(r).name, '') AS start_name,
       coalesce(endNode(r).name, '') AS end_name,
       coalesce(r.chunk_id, '') AS chunk_id,
       coalesce(r.evidence, '') AS evidence
"""


async def fetch_evidences_for_edge_keys(
    driver: AsyncDriver, edge_keys: Iterable[str], *, run_id: str
) -> dict[str, str]:
    """
    Map portable edge_key → raw evidence text (stripped).
    Loads by chunk_id then matches compute_edge_key.
    """
    from server.algorithm.edge_keys import compute_edge_key, parse_edge_key

    corpus_run_id = (run_id or "").strip()
    if not corpus_run_id:
        raise ValueError("run_id is required for evidence lookup")
    keys = [k for k in edge_keys if k]
    if not keys:
        return {}
    chunks: set[str] = set()
    wanted: set[str] = set()
    bad: list[str] = []
    for k in keys:
        try:
            parts = parse_edge_key(k)
        except ValueError:
            bad.append(k)
            continue
        wanted.add(k)
        if parts["chunk_id"]:
            chunks.add(parts["chunk_id"])
    if bad:
        raise ValueError(f"malformed edge_key(s): {bad[:5]}")
    if not chunks:
        return {}

    chunk_list = list(chunks)
    batch_size = 50
    rows: list[dict[str, Any]] = []
    for i in range(0, len(chunk_list), batch_size):
        batch = chunk_list[i : i + batch_size]
        async with driver.session() as session:
            result = await session.run(
                EVIDENCE_BY_CHUNKS,
                chunk_ids=batch,
                run_id=corpus_run_id,
            )
            rows.extend([dict(r) async for r in result])

    out: dict[str, str] = {}
    for r in rows:
        ek = compute_edge_key(
            r.get("start_name"),
            r.get("rel_type"),
            r.get("end_name"),
            r.get("chunk_id"),
            r.get("evidence"),
        )
        if ek in wanted and ek not in out:
            out[ek] = (r.get("evidence") or "").strip()
    return out

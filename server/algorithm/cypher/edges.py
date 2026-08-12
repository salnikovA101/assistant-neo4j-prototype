"""Cypher helpers for V6 (bridges ranked by cosine in DB)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from typing import Any

from neo4j import AsyncDriver

# Whitelist only; nodes usually carry an extra non-schema label too.
_PRIMARY_LABEL_CYPHER = "['Metabolite', 'Microbe', 'StarterCulture', 'EnvironmentCondition']"
_START_LABEL = f"[l IN labels(startNode(r)) WHERE l IN {_PRIMARY_LABEL_CYPHER}][0]"
_END_LABEL = f"[l IN labels(endNode(r)) WHERE l IN {_PRIMARY_LABEL_CYPHER}][0]"
_START_LABEL_REL = f"[l IN labels(startNode(relationship)) WHERE l IN {_PRIMARY_LABEL_CYPHER}][0]"
_END_LABEL_REL = f"[l IN labels(endNode(relationship)) WHERE l IN {_PRIMARY_LABEL_CYPHER}][0]"

FETCH_EDGE_PROPS = f"""
UNWIND $ids AS rid
MATCH ()-[r]->()
WHERE elementId(r) = rid
RETURN elementId(r) AS rid,
       type(r) AS rel_type,
       elementId(startNode(r)) AS start_id,
       elementId(endNode(r)) AS end_id,
       coalesce(startNode(r).name, '') AS start_name,
       coalesce(endNode(r).name, '') AS end_name,
       coalesce({_START_LABEL}, '') AS start_label,
       coalesce({_END_LABEL}, '') AS end_label,
       coalesce(r.chunk_id, '') AS chunk_id,
       coalesce(r.evidence, '') AS evidence,
       coalesce(r.source_file, '') AS source_file,
       coalesce(r.confidence, 1.0) AS confidence,
       r.evidence_embedding AS embedding
"""


FETCH_EDGE_EVIDENCE = """
UNWIND $ids AS rid
MATCH ()-[r]->()
WHERE elementId(r) = rid
RETURN elementId(r) AS rid,
       coalesce(r.evidence, '') AS evidence,
       coalesce(r.chunk_id, '') AS chunk_id,
       coalesce(r.source_file, '') AS source_file,
       coalesce(r.confidence, 1.0) AS confidence
"""


# Read-only hydration for UI graph; intentionally selects no embedding fields.
FETCH_VIZ_BY_EDGE_IDS = f"""
UNWIND $ids AS rid
MATCH (a)-[r]->(b)
WHERE elementId(r) = rid
RETURN elementId(r) AS id,
       type(r) AS type,
       elementId(a) AS from_id,
       elementId(b) AS to_id,
       coalesce(a.name, '') AS from_name,
       coalesce(b.name, '') AS to_name,
       coalesce([l IN labels(a) WHERE l IN {_PRIMARY_LABEL_CYPHER}][0], '') AS from_label,
       coalesce([l IN labels(b) WHERE l IN {_PRIMARY_LABEL_CYPHER}][0], '') AS to_label,
       a.leiden_community AS from_community,
       b.leiden_community AS to_community,
       coalesce(r.evidence, '') AS evidence,
       coalesce(r.chunk_id, '') AS chunk_id,
       coalesce(r.source_file, '') AS source_file,
       coalesce(r.confidence, 1.0) AS confidence,
       coalesce(r.run_id, '') AS run_id
"""


# Induced bridges on endpoints, ranked by cosine(evidence_emb, $sqVec), LIMIT in DB.
INDUCED_BRIDGES_BY_SIM = f"""
UNWIND $node_ids AS nid
MATCH (n)-[r]-(m)
WHERE elementId(n) = nid
  AND elementId(m) IN $node_ids
  AND elementId(n) < elementId(m)
  AND NOT elementId(r) IN $exclude_ids
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
       coalesce({_START_LABEL}, '') AS start_label,
       coalesce({_END_LABEL}, '') AS end_label,
       coalesce(r.chunk_id, '') AS chunk_id,
       coalesce(r.evidence, '') AS evidence,
       coalesce(r.source_file, '') AS source_file,
       coalesce(r.confidence, 1.0) AS confidence,
       r.evidence_embedding AS embedding,
       score AS score
"""


async def fetch_edge_properties(driver: AsyncDriver, element_ids: Iterable[str]) -> list[dict[str, Any]]:
    ids = list({i for i in element_ids if i})
    if not ids:
        return []
    batch_size = 400
    chunks = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]

    async def _one(chunk: list[str]) -> list[dict[str, Any]]:
        async with driver.session() as session:
            result = await session.run(FETCH_EDGE_PROPS, ids=chunk)
            return [dict(r) async for r in result]

    parts = await asyncio.gather(*[_one(c) for c in chunks])
    out: list[dict[str, Any]] = []
    for p in parts:
        out.extend(p)
    return out


async def fetch_edge_evidence(driver: AsyncDriver, element_ids: Iterable[str]) -> dict[str, dict[str, Any]]:
    ids = list({i for i in element_ids if i})
    if not ids:
        return {}
    batch_size = 400
    chunks = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]

    async def _one(chunk: list[str]) -> dict[str, dict[str, Any]]:
        partial: dict[str, dict[str, Any]] = {}
        async with driver.session() as session:
            result = await session.run(FETCH_EDGE_EVIDENCE, ids=chunk)
            async for r in result:
                partial[r["rid"]] = {
                    "evidence": r["evidence"] or "",
                    "chunk_id": r["chunk_id"] or "",
                    "source_file": r["source_file"] or "",
                    "confidence": float(r["confidence"] or 1.0),
                }
        return partial

    parts = await asyncio.gather(*[_one(c) for c in chunks])
    out: dict[str, dict[str, Any]] = {}
    for p in parts:
        out.update(p)
    return out


async def fetch_viz_edges(driver: AsyncDriver, element_ids: Iterable[str]) -> list[dict[str, Any]]:
    ids = list({i for i in element_ids if i})
    if not ids:
        return []
    batch_size = 400
    chunks = [ids[i : i + batch_size] for i in range(0, len(ids), batch_size)]

    async def _one(chunk: list[str]) -> list[dict[str, Any]]:
        async with driver.session() as session:
            result = await session.run(FETCH_VIZ_BY_EDGE_IDS, ids=chunk)
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
) -> list[dict[str, Any]]:
    """Induced bridges ranked by cosine to sq_vec; LIMIT applied in Cypher."""
    ids = list({i for i in node_ids if i})
    if len(ids) < 2 or limit <= 0 or not sq_vec:
        return []
    excl = list({i for i in (exclude_ids or []) if i})
    async with driver.session() as session:
        result = await session.run(
            INDUCED_BRIDGES_BY_SIM,
            node_ids=ids,
            exclude_ids=excl,
            sqVec=list(sq_vec),
            limit=int(limit),
        )
        return [dict(r) async for r in result]


async def query_relationship_ann(
    driver: AsyncDriver,
    index_name: str,
    embedding: list[float],
    top_k: int,
) -> list[dict[str, Any]]:
    cypher = f"""
    CALL db.index.vector.queryRelationships($index, $k, $embedding)
    YIELD relationship, score
    RETURN elementId(relationship) AS rid,
           type(relationship) AS rel_type,
           elementId(startNode(relationship)) AS start_id,
           elementId(endNode(relationship)) AS end_id,
           coalesce(startNode(relationship).name, '') AS start_name,
           coalesce(endNode(relationship).name, '') AS end_name,
           coalesce({_START_LABEL_REL}, '') AS start_label,
           coalesce({_END_LABEL_REL}, '') AS end_label,
           coalesce(relationship.chunk_id, '') AS chunk_id,
           coalesce(relationship.evidence, '') AS evidence,
           score AS score
    """
    async with driver.session() as session:
        result = await session.run(cypher, index=index_name, k=int(top_k), embedding=embedding)
        return [dict(r) async for r in result]


EVIDENCE_BY_CHUNKS = """
UNWIND $chunk_ids AS cid
MATCH ()-[r]->()
WHERE r.chunk_id = cid
RETURN type(r) AS rel_type,
       coalesce(startNode(r).name, '') AS start_name,
       coalesce(endNode(r).name, '') AS end_name,
       coalesce(r.chunk_id, '') AS chunk_id,
       coalesce(r.evidence, '') AS evidence
"""


async def fetch_evidences_for_edge_keys(driver: AsyncDriver, edge_keys: Iterable[str]) -> dict[str, str]:
    """
    Map portable edge_key → raw evidence text (stripped).
    Loads by chunk_id then matches compute_edge_key.
    """
    from server.algorithm.edge_keys import compute_edge_key, parse_edge_key

    keys = [k for k in edge_keys if k]
    if not keys:
        return {}
    chunks: set[str] = set()
    wanted: set[str] = set()
    for k in keys:
        try:
            parts = parse_edge_key(k)
        except ValueError:
            continue
        wanted.add(k)
        if parts["chunk_id"]:
            chunks.add(parts["chunk_id"])
    if not chunks:
        return {}

    chunk_list = list(chunks)
    batch_size = 50
    rows: list[dict[str, Any]] = []
    for i in range(0, len(chunk_list), batch_size):
        batch = chunk_list[i : i + batch_size]
        async with driver.session() as session:
            result = await session.run(EVIDENCE_BY_CHUNKS, chunk_ids=batch)
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

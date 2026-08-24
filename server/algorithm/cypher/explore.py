"""Read-only triplet search for the UI explorer. No embeddings."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncDriver

from server.algorithm.models import PRIMARY_NODE_LABELS

EXPLORE_LIMITS: tuple[int, ...] = (10, 100, 1000)
EXPLORE_FIELDS: tuple[str, ...] = ("all", "name", "label", "rel", "evidence", "source")

_PRIMARY_LABEL_CYPHER = "[" + ", ".join(repr(x) for x in PRIMARY_NODE_LABELS) + "]"
_FROM_LABEL = f"[l IN labels(a) WHERE l IN {_PRIMARY_LABEL_CYPHER}][0]"
_TO_LABEL = f"[l IN labels(b) WHERE l IN {_PRIMARY_LABEL_CYPHER}][0]"

# Match relationships first (subject —rel→ object), then take their endpoints.
_FETCH_TRIPLETS = f"""
MATCH (a)-[r]->(b)
WHERE any(l IN labels(a) WHERE l IN {_PRIMARY_LABEL_CYPHER})
  AND any(l IN labels(b) WHERE l IN {_PRIMARY_LABEL_CYPHER})
  AND ($run_id = '' OR r.run_id = $run_id)
  AND ($cursor = '' OR elementId(r) > $cursor)
  AND $q <> ''
  AND (
    $q = ''
    OR (
      $field IN ['all', 'name'] AND (
        toLower(coalesce(a.name, '')) CONTAINS $q
        OR toLower(coalesce(b.name, '')) CONTAINS $q
      )
    )
    OR (
      $field IN ['all', 'rel'] AND toLower(type(r)) CONTAINS $q
    )
    OR (
      $field IN ['all', 'label'] AND (
        any(l IN labels(a) WHERE toLower(l) CONTAINS $q)
        OR any(l IN labels(b) WHERE toLower(l) CONTAINS $q)
      )
    )
    OR (
      $field IN ['all', 'evidence'] AND toLower(coalesce(r.evidence, '')) CONTAINS $q
    )
    OR (
      $field IN ['all', 'source'] AND toLower(coalesce(r.source_file, '')) CONTAINS $q
    )
  )
WITH DISTINCT a, r, b,
     CASE
       WHEN toLower(coalesce(a.name, '')) = $q OR toLower(coalesce(b.name, '')) = $q THEN 100
       WHEN toLower(coalesce(a.name, '')) STARTS WITH $q OR toLower(coalesce(b.name, '')) STARTS WITH $q THEN 80
       WHEN toLower(type(r)) = $q THEN 90
       WHEN toLower(type(r)) STARTS WITH $q THEN 70
       WHEN toLower(coalesce(a.name, '')) CONTAINS $q OR toLower(coalesce(b.name, '')) CONTAINS $q THEN 60
       WHEN toLower(coalesce(r.source_file, '')) CONTAINS $q THEN 30
       ELSE 20
     END AS relevance
RETURN
       elementId(r) AS id,
       type(r) AS type,
       elementId(a) AS from_id,
       elementId(b) AS to_id,
       coalesce(a.name, '') AS from_name,
       coalesce(b.name, '') AS to_name,
       coalesce({_FROM_LABEL}, '') AS from_label,
       coalesce({_TO_LABEL}, '') AS to_label,
       coalesce(r.evidence, '') AS evidence,
       coalesce(r.chunk_id, '') AS chunk_id,
       coalesce(r.source_file, '') AS source_file,
       coalesce(r.confidence, 1.0) AS confidence,
       coalesce(r.run_id, '') AS run_id
ORDER BY relevance DESC, id
LIMIT $limit
"""

_EXPAND_TRIPLETS = f"""
MATCH (a)-[r]-(b)
WHERE elementId(a) = $node_id
  AND any(l IN labels(a) WHERE l IN {_PRIMARY_LABEL_CYPHER})
  AND any(l IN labels(b) WHERE l IN {_PRIMARY_LABEL_CYPHER})
  AND ($run_id = '' OR r.run_id = $run_id)
WITH startNode(r) AS s, endNode(r) AS t, r
RETURN DISTINCT
       elementId(r) AS id,
       type(r) AS type,
       elementId(s) AS from_id,
       elementId(t) AS to_id,
       coalesce(s.name, '') AS from_name,
       coalesce(t.name, '') AS to_name,
       coalesce([l IN labels(s) WHERE l IN {_PRIMARY_LABEL_CYPHER}][0], '') AS from_label,
       coalesce([l IN labels(t) WHERE l IN {_PRIMARY_LABEL_CYPHER}][0], '') AS to_label,
       coalesce(r.evidence, '') AS evidence,
       coalesce(r.chunk_id, '') AS chunk_id,
       coalesce(r.source_file, '') AS source_file,
       coalesce(r.confidence, 1.0) AS confidence,
       coalesce(r.run_id, '') AS run_id
LIMIT $limit
"""


def clamp_explore_limit(limit: int) -> int:
    if limit in EXPLORE_LIMITS:
        return limit
    return 100


def clamp_explore_field(field: str) -> str:
    value = (field or "all").strip().lower()
    return value if value in EXPLORE_FIELDS else "all"


def nodes_from_triplet_rows(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in edges:
        fid = str(row.get("from_id") or "")
        tid = str(row.get("to_id") or "")
        if fid and fid not in by_id:
            by_id[fid] = {
                "id": fid,
                "name": str(row.get("from_name") or ""),
                "label": str(row.get("from_label") or ""),
            }
        if tid and tid not in by_id:
            by_id[tid] = {
                "id": tid,
                "name": str(row.get("to_name") or ""),
                "label": str(row.get("to_label") or ""),
            }
    return list(by_id.values())


async def fetch_explore_rows(
    driver: AsyncDriver,
    *,
    q: str,
    limit: int,
    run_id: str,
    field: str = "all",
    cursor: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    query = (q or "").strip().lower()
    cap = clamp_explore_limit(int(limit))
    rid = (run_id or "").strip()
    scope = clamp_explore_field(field)
    params = {"q": query, "limit": cap + 1, "run_id": rid, "field": scope, "cursor": (cursor or "").strip()}

    async with driver.session() as session:
        edges = [dict(r) async for r in await session.run(_FETCH_TRIPLETS, **params)]
    has_more = len(edges) > cap
    edges = edges[:cap]
    next_cursor = str(edges[-1].get("id") or "") if has_more and edges else None
    return nodes_from_triplet_rows(edges), edges, next_cursor


async def fetch_expand_rows(
    driver: AsyncDriver,
    *,
    node_id: str,
    limit: int,
    run_id: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    params = {
        "node_id": (node_id or "").strip(),
        "limit": clamp_explore_limit(int(limit)),
        "run_id": (run_id or "").strip(),
    }
    async with driver.session() as session:
        edges = [dict(r) async for r in await session.run(_EXPAND_TRIPLETS, **params)]
    return nodes_from_triplet_rows(edges), edges

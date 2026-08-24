"""Read-only triplet search for the UI explorer. No embeddings."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncDriver

from server.algorithm.models import PRIMARY_NODE_LABELS

EXPLORE_LIMITS: tuple[int, ...] = (10, 100, 1000)
EXPLORE_FIELDS: tuple[str, ...] = ("all", "name", "rel", "evidence")

_PRIMARY_LABEL_CYPHER = "[" + ", ".join(repr(x) for x in PRIMARY_NODE_LABELS) + "]"
_FROM_LABEL = f"[l IN labels(a) WHERE l IN {_PRIMARY_LABEL_CYPHER}][0]"
_TO_LABEL = f"[l IN labels(b) WHERE l IN {_PRIMARY_LABEL_CYPHER}][0]"

# Match relationships first (subject —rel→ object), then take their endpoints.
_FETCH_TRIPLETS = f"""
MATCH (a)-[r]->(b)
WHERE any(l IN labels(a) WHERE l IN {_PRIMARY_LABEL_CYPHER})
  AND any(l IN labels(b) WHERE l IN {_PRIMARY_LABEL_CYPHER})
  AND ($run_id = '' OR r.run_id = $run_id)
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
      $field IN ['all', 'evidence'] AND (
        toLower(coalesce(r.evidence, '')) CONTAINS $q
        OR toLower(coalesce(r.source_file, '')) CONTAINS $q
      )
    )
  )
RETURN DISTINCT
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    query = (q or "").strip().lower()
    cap = clamp_explore_limit(int(limit))
    rid = (run_id or "").strip()
    scope = clamp_explore_field(field)
    params = {"q": query, "limit": cap, "run_id": rid, "field": scope}

    async with driver.session() as session:
        edges = [dict(r) async for r in await session.run(_FETCH_TRIPLETS, **params)]
    return nodes_from_triplet_rows(edges), edges

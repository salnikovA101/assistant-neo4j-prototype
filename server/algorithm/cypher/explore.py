"""Read-only triplet search for the UI explorer. No embeddings."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncDriver

from server.algorithm.models import normalize_labels

# Presets remain useful for callers that want a small/medium/large request,
# but the explorer also accepts a custom limit within the guarded range.
EXPLORE_LIMITS: tuple[int, ...] = (10, 100, 1000)
MIN_EXPLORE_LIMIT = 1
MAX_EXPLORE_LIMIT = 5000
EXPLORE_FIELDS: tuple[str, ...] = ("all", "name", "label", "rel", "evidence", "source")

def _filter_predicates(*, skip: str = "") -> str:
    parts: list[str] = []
    if skip != "node_labels":
        parts.append(
            "AND (size($node_labels) = 0 "
            "OR any(l IN labels(a) WHERE l IN $node_labels) "
            "OR any(l IN labels(b) WHERE l IN $node_labels))"
        )
    if skip != "relationship_types":
        parts.append("AND (size($relationship_types) = 0 OR type(r) IN $relationship_types)")
    if skip != "sources":
        parts.append("AND (size($sources) = 0 OR coalesce(r.source_file, '') IN $sources)")
    parts.extend(
        [
            "AND trim(coalesce(r.evidence, '')) <> ''",
            "AND ($min_confidence IS NULL OR r.confidence >= $min_confidence)",
        ]
    )
    return "\n  ".join(parts)


_QUERY_PREDICATE = """AND (
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
  )"""

_MATCH_BASE = """MATCH (a)-[r]->(b)
WHERE r.run_id = $run_id"""

# Match relationships first (subject —rel→ object), then take their endpoints.
_FETCH_TRIPLETS = f"""
{_MATCH_BASE}
  AND ($cursor = '' OR elementId(r) > $cursor)
  {_filter_predicates()}
  {_QUERY_PREDICATE}
  AND ($q <> '' OR $has_filters)
WITH DISTINCT a, r, b,
     CASE
       WHEN toLower(coalesce(a.name, '')) = $q OR toLower(coalesce(b.name, '')) = $q THEN 100
       WHEN toLower(coalesce(a.name, '')) STARTS WITH $q OR toLower(coalesce(b.name, '')) STARTS WITH $q THEN 80
       WHEN toLower(type(r)) = $q THEN 90
       WHEN toLower(type(r)) STARTS WITH $q THEN 70
       WHEN toLower(coalesce(a.name, '')) CONTAINS $q OR toLower(coalesce(b.name, '')) CONTAINS $q THEN 60
       WHEN toLower(coalesce(r.source_file, '')) CONTAINS $q THEN 30
       ELSE 20
     END AS relevance,
     CASE WHEN trim(coalesce(r.evidence, '')) <> '' THEN 1 ELSE 0 END AS has_evidence,
     coalesce(r.confidence, -1.0) AS confidence_sort
RETURN
       elementId(r) AS id,
       type(r) AS type,
       elementId(a) AS from_id,
       elementId(b) AS to_id,
       coalesce(a.name, '') AS from_name,
       coalesce(b.name, '') AS to_name,
       labels(a) AS from_labels,
       labels(b) AS to_labels,
       coalesce(r.evidence, '') AS evidence,
       coalesce(r.chunk_id, '') AS chunk_id,
       coalesce(r.source_file, '') AS source_file,
       r.confidence AS confidence,
       coalesce(r.run_id, '') AS run_id
ORDER BY relevance DESC, has_evidence DESC, confidence_sort DESC, id
LIMIT $limit
"""

_EXPAND_TRIPLETS = f"""
MATCH (anchor)-[r]-(neighbor)
WITH anchor, r, startNode(r) AS a, endNode(r) AS b
WHERE elementId(anchor) = $node_id
  AND r.run_id = $run_id
  AND NOT elementId(r) IN $exclude_edge_ids
  AND (
    $direction = 'all'
    OR ($direction = 'outgoing' AND elementId(a) = $node_id)
    OR ($direction = 'incoming' AND elementId(b) = $node_id)
  )
  {_filter_predicates()}
WITH DISTINCT a, b, r,
     CASE WHEN trim(coalesce(r.evidence, '')) <> '' THEN 1 ELSE 0 END AS evidence_rank,
     coalesce(r.confidence, -1.0) AS confidence_rank
RETURN
       elementId(r) AS id,
       type(r) AS type,
       elementId(a) AS from_id,
       elementId(b) AS to_id,
       coalesce(a.name, '') AS from_name,
       coalesce(b.name, '') AS to_name,
       labels(a) AS from_labels,
       labels(b) AS to_labels,
       coalesce(r.evidence, '') AS evidence,
       coalesce(r.chunk_id, '') AS chunk_id,
       coalesce(r.source_file, '') AS source_file,
       r.confidence AS confidence,
       coalesce(r.run_id, '') AS run_id
ORDER BY evidence_rank DESC, confidence_rank DESC, id
LIMIT $limit
"""

_EXPAND_TOTAL = f"""
MATCH (anchor)-[r]-(neighbor)
WITH anchor, r, startNode(r) AS a, endNode(r) AS b
WHERE elementId(anchor) = $node_id
  AND r.run_id = $run_id
  AND (
    $direction = 'all'
    OR ($direction = 'outgoing' AND elementId(a) = $node_id)
    OR ($direction = 'incoming' AND elementId(b) = $node_id)
  )
  {_filter_predicates()}
RETURN count(DISTINCT r) AS count
"""

_FACET_SUMMARY = f"""
{_MATCH_BASE}
  {_filter_predicates()}
  {_QUERY_PREDICATE}
WITH collect(DISTINCT r) AS rels, collect(DISTINCT a) + collect(DISTINCT b) AS endpoints
UNWIND CASE WHEN size(endpoints) = 0 THEN [null] ELSE endpoints END AS n
RETURN size(rels) AS relationship_count, count(DISTINCT n) AS node_count
"""

_FACET_NODE_LABELS = f"""
{_MATCH_BASE}
  {_filter_predicates(skip='node_labels')}
  {_QUERY_PREDICATE}
WITH collect(DISTINCT a) + collect(DISTINCT b) AS endpoints
UNWIND endpoints AS n
UNWIND labels(n) AS value
RETURN value, count(DISTINCT n) AS count
ORDER BY count DESC, value
"""

_FACET_RELATIONSHIPS = f"""
{_MATCH_BASE}
  {_filter_predicates(skip='relationship_types')}
  {_QUERY_PREDICATE}
RETURN type(r) AS value, count(DISTINCT r) AS count
ORDER BY count DESC, value
"""

_FACET_SOURCES = f"""
{_MATCH_BASE}
  {_filter_predicates(skip='sources')}
  {_QUERY_PREDICATE}
  AND trim(coalesce(r.source_file, '')) <> ''
  AND ($source_query = '' OR toLower(r.source_file) CONTAINS $source_query)
RETURN r.source_file AS value, count(DISTINCT r) AS count
ORDER BY count DESC, value
SKIP $source_offset
LIMIT $source_limit
"""


def clamp_explore_limit(limit: int) -> int:
    try:
        value = int(limit)
    except (TypeError, ValueError):
        return 100
    return max(MIN_EXPLORE_LIMIT, min(value, MAX_EXPLORE_LIMIT))


def clamp_explore_field(field: str) -> str:
    value = (field or "all").strip().lower()
    return value if value in EXPLORE_FIELDS else "all"


def _clean_list(values: Any, *, limit: int) -> list[str]:
    if not isinstance(values, (list, tuple)):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = str(raw or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def normalize_graph_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    raw = filters or {}
    confidence = raw.get("min_confidence")
    try:
        min_confidence = None if confidence is None or confidence == "" else max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        min_confidence = None
    return {
        "node_labels": _clean_list(raw.get("node_labels"), limit=32),
        "relationship_types": _clean_list(raw.get("relationship_types"), limit=128),
        "sources": _clean_list(raw.get("sources"), limit=128),
        "min_confidence": min_confidence,
    }


def graph_filters_active(filters: dict[str, Any] | None) -> bool:
    normalized = normalize_graph_filters(filters)
    return bool(
        normalized["node_labels"]
        or normalized["relationship_types"]
        or normalized["sources"]
        or normalized["min_confidence"] is not None
    )


def _query_params(*, q: str, field: str, run_id: str, filters: dict[str, Any] | None) -> dict[str, Any]:
    normalized = normalize_graph_filters(filters)
    corpus_run_id = (run_id or "").strip()
    if not corpus_run_id:
        raise ValueError("run_id is required for graph exploration")
    return {
        "q": (q or "").strip().lower(),
        "field": clamp_explore_field(field),
        "run_id": corpus_run_id,
        **normalized,
    }


def nodes_from_triplet_rows(edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in edges:
        fid = str(row.get("from_id") or "")
        tid = str(row.get("to_id") or "")
        if fid and fid not in by_id:
            by_id[fid] = {
                "id": fid,
                "name": str(row.get("from_name") or ""),
                "labels": normalize_labels(row.get("from_labels")),
            }
        if tid and tid not in by_id:
            by_id[tid] = {
                "id": tid,
                "name": str(row.get("to_name") or ""),
                "labels": normalize_labels(row.get("to_labels")),
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
    filters: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str | None]:
    cap = clamp_explore_limit(int(limit))
    params = {
        **_query_params(q=q, field=field, run_id=run_id, filters=filters),
        "limit": cap + 1,
        "cursor": (cursor or "").strip(),
        "has_filters": graph_filters_active(filters),
    }

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
    exclude_edge_ids: list[str] | None = None,
    direction: str = "all",
    filters: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, bool]:
    cap = clamp_explore_limit(int(limit))
    normalized_direction = direction if direction in {"all", "incoming", "outgoing"} else "all"
    params = {
        **_query_params(q="", field="all", run_id=run_id, filters=filters),
        "node_id": (node_id or "").strip(),
        "limit": cap + 1,
        "exclude_edge_ids": _clean_list(exclude_edge_ids, limit=5000),
        "direction": normalized_direction,
    }
    async with driver.session() as session:
        edges = [dict(r) async for r in await session.run(_EXPAND_TRIPLETS, **params)]
        total_row = await (await session.run(_EXPAND_TOTAL, **params)).single()
    has_more = len(edges) > cap
    edges = edges[:cap]
    total = int(total_row.get("count") or 0) if total_row else 0
    return nodes_from_triplet_rows(edges), edges, total, has_more


async def fetch_graph_facets(
    driver: AsyncDriver,
    *,
    q: str,
    run_id: str,
    field: str = "all",
    filters: dict[str, Any] | None = None,
    source_query: str = "",
    source_cursor: str = "",
    source_limit: int = 50,
) -> dict[str, Any]:
    try:
        source_offset = max(0, int(source_cursor or "0"))
    except (TypeError, ValueError):
        source_offset = 0
    source_cap = max(1, min(int(source_limit), 200))
    params = {
        **_query_params(q=q, field=field, run_id=run_id, filters=filters),
        "source_query": (source_query or "").strip().lower(),
        "source_offset": source_offset,
        "source_limit": source_cap + 1,
    }
    async with driver.session() as session:
        summary_row = await (await session.run(_FACET_SUMMARY, **params)).single()
        node_rows = [dict(row) async for row in await session.run(_FACET_NODE_LABELS, **params)]
        rel_rows = [dict(row) async for row in await session.run(_FACET_RELATIONSHIPS, **params)]
        source_rows = [dict(row) async for row in await session.run(_FACET_SOURCES, **params)]

    has_more_sources = len(source_rows) > source_cap
    source_rows = source_rows[:source_cap]

    def items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {"value": str(row.get("value") or ""), "count": int(row.get("count") or 0)}
            for row in rows
            if str(row.get("value") or "").strip()
        ]

    return {
        "matchingRelationships": int(summary_row.get("relationship_count") or 0) if summary_row else 0,
        "matchingNodes": int(summary_row.get("node_count") or 0) if summary_row else 0,
        "nodeLabels": items(node_rows),
        "relationshipTypes": items(rel_rows),
        "sources": {
            "items": items(source_rows),
            "nextCursor": str(source_offset + source_cap) if has_more_sources else None,
            "hasMore": has_more_sources,
        },
    }

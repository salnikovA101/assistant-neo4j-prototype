"""Full-corpus graph explorer payload. Same node/edge shape as graph_viz, no LLM."""

from __future__ import annotations

from typing import Any

from neo4j import AsyncDriver

from server.algorithm.cypher.explore import fetch_expand_rows, fetch_explore_rows
from server.tools.graph_viz import DEFAULT_NODE_COLOR, NODE_COLORS


_SKIP_PROP_KEYS = frozenset({"embedding", "evidence_embedding"})


def _clean_props(props: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in props.items()
        if key not in _SKIP_PROP_KEYS and not str(key).endswith("_embedding")
    }


def rows_to_explore_payload(
    node_rows: list[dict[str, Any]],
    edge_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for row in node_rows:
        node_id = str(row.get("id") or "")
        if not node_id:
            continue
        group = str(row.get("label") or "").strip() or "Unknown"
        caption = str(row.get("name") or "").strip()
        nodes.append(
            {
                "id": node_id,
                "label": group,
                "caption": caption,
                "group": group,
                "color": NODE_COLORS.get(group, DEFAULT_NODE_COLOR),
                "properties": _clean_props(
                    {
                        "name": caption,
                        "label": group,
                    }
                ),
            }
        )

    edges: list[dict[str, Any]] = []
    for row in edge_rows:
        edge_id = str(row.get("id") or "")
        from_id = str(row.get("from_id") or "")
        to_id = str(row.get("to_id") or "")
        if not edge_id or not from_id or not to_id:
            continue
        rel = str(row.get("type") or "RELATED")
        from_group = str(row.get("from_label") or "").strip()
        to_group = str(row.get("to_label") or "").strip()
        from_name = str(row.get("from_name") or "").strip()
        to_name = str(row.get("to_name") or "").strip()
        confidence = row.get("confidence")
        try:
            confidence_f = float(confidence) if confidence is not None and confidence != "" else None
        except (TypeError, ValueError):
            confidence_f = None
        edges.append(
            {
                "id": edge_id,
                "from": from_id,
                "to": to_id,
                "label": rel,
                "role": "spine",
                "chain_ids": [],
                "from_name": from_name,
                "to_name": to_name,
                "from_group": from_group,
                "to_group": to_group,
                "properties": _clean_props(
                    {
                        "evidence": str(row.get("evidence") or ""),
                        "chunk_id": str(row.get("chunk_id") or ""),
                        "source_file": str(row.get("source_file") or ""),
                        "confidence": confidence_f,
                        "sim": 0.0,
                        "source": "",
                        "run_id": str(row.get("run_id") or ""),
                    }
                ),
            }
        )

    return {"views": [], "all": {"nodes": nodes, "edges": edges}}


async def build_graph_explore_payload(
    driver: AsyncDriver,
    *,
    q: str,
    limit: int,
    run_id: str,
    field: str = "all",
    cursor: str = "",
) -> dict[str, Any]:
    node_rows, edge_rows, next_cursor = await fetch_explore_rows(
        driver, q=q, limit=limit, run_id=run_id, field=field, cursor=cursor
    )
    payload = rows_to_explore_payload(node_rows, edge_rows)
    payload["page"] = {"nextCursor": next_cursor, "hasMore": bool(next_cursor)}
    return payload


async def build_graph_expand_payload(
    driver: AsyncDriver,
    *,
    node_id: str,
    limit: int,
    run_id: str,
) -> dict[str, Any]:
    node_rows, edge_rows = await fetch_expand_rows(
        driver, node_id=node_id, limit=limit, run_id=run_id
    )
    return rows_to_explore_payload(node_rows, edge_rows)

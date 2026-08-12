"""Accepted-chain graph payloads for the UI. Read-only Neo4j hydration, no LLM."""

from __future__ import annotations

from typing import Any, Iterable

from neo4j import AsyncDriver

from server.algorithm.cypher.edges import fetch_viz_edges

NODE_COLORS: dict[str, str] = {
    "Metabolite": "#c990c0",
    "Microbe": "#569480",
    "StarterCulture": "#4c8dff",
    "EnvironmentCondition": "#f0a85e",
}
DEFAULT_NODE_COLOR = "#a5abb6"


def _first(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iter_chain_edges(chain: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any], str]]:
    for edge in chain.get("edges") or []:
        if isinstance(edge, dict):
            yield "spine", edge, ""
    for hub_id, fan_edges in (chain.get("fans") or {}).items():
        for edge in fan_edges or []:
            if isinstance(edge, dict):
                yield "fan", edge, str(hub_id)


def _add_node(
    nodes: dict[str, dict[str, Any]],
    node_id: str,
    *,
    name: str = "",
    label: str = "",
    community: Any = None,
) -> None:
    if not node_id:
        return
    caption = (name or "").strip() or f"Node {node_id[-6:]}"
    group = (label or "").strip() or "Unknown"
    color = NODE_COLORS.get(group, DEFAULT_NODE_COLOR)

    properties: dict[str, Any] = {"name": caption}
    if group != "Unknown":
        properties["label"] = group
    if community is not None:
        properties["leiden_community"] = community

    existing = nodes.get(node_id)
    if existing is None:
        nodes[node_id] = {
            "id": node_id,
            "label": group,
            "caption": caption,
            "group": group,
            "color": color,
            "properties": properties,
        }
        return

    if existing.get("group") in ("", "Unknown") and group != "Unknown":
        existing["group"] = group
        existing["label"] = group
        existing["color"] = color
    if not existing.get("caption"):
        existing["caption"] = caption
    existing.setdefault("properties", {})
    for key, value in properties.items():
        existing["properties"].setdefault(key, value)


def _edge_payload(
    edge: dict[str, Any],
    hydration: dict[str, Any],
    *,
    role: str,
    chain_id: str,
    hub_id: str = "",
) -> dict[str, Any] | None:
    edge_id = _first(edge.get("element_id"), edge.get("id"), hydration.get("id"))
    from_id = _first(hydration.get("from_id"), edge.get("start_id"))
    to_id = _first(hydration.get("to_id"), edge.get("end_id"))
    if not edge_id or not from_id or not to_id:
        return None

    properties: dict[str, Any] = {
        "edge_key": _first(edge.get("edge_key"), hydration.get("id")),
        "evidence": _first(hydration.get("evidence"), edge.get("evidence"), ""),
        "chunk_id": _first(hydration.get("chunk_id"), edge.get("chunk_id"), ""),
        "source_file": _first(hydration.get("source_file"), edge.get("source_file"), ""),
        "confidence": _as_float(_first(hydration.get("confidence"), edge.get("confidence"), 1.0), 1.0),
        "run_id": _first(hydration.get("run_id"), ""),
        "sim": _as_float(edge.get("sim"), 0.0),
        "source": _first(edge.get("source"), ""),
    }
    if hub_id:
        properties["hub_id"] = hub_id

    return {
        "id": str(edge_id),
        "from": str(from_id),
        "to": str(to_id),
        "label": _first(hydration.get("type"), edge.get("type"), "RELATED"),
        "role": role,
        "chain_ids": [chain_id],
        "properties": properties,
    }


def _merge_edge(target: dict[str, dict[str, Any]], edge: dict[str, Any]) -> None:
    existing = target.get(edge["id"])
    if existing is None:
        target[edge["id"]] = edge
        return

    existing["chain_ids"] = sorted(set(existing.get("chain_ids") or []) | set(edge.get("chain_ids") or []))
    if existing.get("role") != "spine" and edge.get("role") == "spine":
        existing["role"] = "spine"
    for key, value in (edge.get("properties") or {}).items():
        existing.setdefault("properties", {}).setdefault(key, value)


def build_chain_views(
    chains: list[dict[str, Any]],
    hydration: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    hydration = hydration or {}
    views: list[dict[str, Any]] = []

    for idx, chain in enumerate(chains, 1):
        chain_id = str(chain.get("chain_id") or f"a{idx}")
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}

        for role, edge, hub_id in _iter_chain_edges(chain):
            edge_id = _first(edge.get("element_id"), edge.get("id"))
            row = hydration.get(str(edge_id), {}) if edge_id else {}
            payload = _edge_payload(edge, row, role=role, chain_id=chain_id, hub_id=hub_id)
            if payload is None:
                continue

            _add_node(
                nodes,
                payload["from"],
                name=_first(row.get("from_name"), edge.get("start"), edge.get("start_name"), ""),
                label=_first(row.get("from_label"), edge.get("start_label"), ""),
                community=row.get("from_community"),
            )
            _add_node(
                nodes,
                payload["to"],
                name=_first(row.get("to_name"), edge.get("end"), edge.get("end_name"), ""),
                label=_first(row.get("to_label"), edge.get("end_label"), ""),
                community=row.get("to_community"),
            )
            _merge_edge(edges, payload)

        views.append(
            {
                "id": chain_id,
                "label": f"Цепь {idx}",
                "score": _as_float(chain.get("score"), 0.0),
                "nodes": list(nodes.values()),
                "edges": list(edges.values()),
            }
        )

    return views


def merge_views(views: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}

    for view in views:
        for node in view.get("nodes") or []:
            existing = nodes.get(node["id"])
            if existing is None:
                nodes[node["id"]] = node
            else:
                if existing.get("group") in ("", "Unknown") and node.get("group") not in ("", "Unknown"):
                    existing["group"] = node["group"]
                    existing["label"] = node["label"]
                    existing["color"] = node["color"]
                existing.setdefault("properties", {})
                for key, value in (node.get("properties") or {}).items():
                    existing["properties"].setdefault(key, value)

        for edge in view.get("edges") or []:
            _merge_edge(edges, edge)

    return {"nodes": list(nodes.values()), "edges": list(edges.values())}


async def build_graph_viz_payload(
    driver: AsyncDriver,
    chains: list[dict[str, Any]],
) -> dict[str, Any]:
    edge_ids: list[str] = []
    for chain in chains:
        for _, edge, _ in _iter_chain_edges(chain):
            edge_id = _first(edge.get("element_id"), edge.get("id"))
            if edge_id:
                edge_ids.append(str(edge_id))

    rows = await fetch_viz_edges(driver, edge_ids)
    hydration = {str(row.get("id")): row for row in rows if row.get("id")}

    views = build_chain_views(chains, hydration)
    return {"views": views, "all": merge_views(views)}

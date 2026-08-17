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


def _is_blank(value: Any) -> bool:
    return value is None or value == ""


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iter_chain_edges(chain: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any], str]]:
    for edge in chain.get("edges") or []:
        if isinstance(edge, dict):
            yield "spine", edge, ""
    for hub_id, fan_edges in (chain.get("fans") or {}).items():
        for edge in fan_edges or []:
            if isinstance(edge, dict):
                yield "fan", edge, str(hub_id)


def _fill_if_blank(target: dict[str, Any], key: str, value: Any) -> None:
    if _is_blank(target.get(key)) and not _is_blank(value):
        target[key] = value


def _merge_props(dst: dict[str, Any], src: dict[str, Any]) -> None:
    for key, value in src.items():
        if _is_blank(dst.get(key)) and not _is_blank(value):
            dst[key] = value


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
    caption = (name or "").strip()
    group = (label or "").strip() or "Unknown"
    color = NODE_COLORS.get(group, DEFAULT_NODE_COLOR)

    properties: dict[str, Any] = {}
    if caption:
        properties["name"] = caption
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
    if caption and not (existing.get("caption") or "").strip():
        existing["caption"] = caption
    existing.setdefault("properties", {})
    _merge_props(existing["properties"], properties)


def _stable_edge_id(
    edge: dict[str, Any],
    hydration: dict[str, Any],
    *,
    from_id: str,
    to_id: str,
    rel: str,
) -> str | None:
    eid = _first(edge.get("element_id"), edge.get("id"), hydration.get("id"))
    if eid:
        return str(eid)
    key = _first(edge.get("edge_key"))
    if key:
        return f"ek:{key}"
    if from_id and to_id and rel:
        chunk = _first(edge.get("chunk_id"), hydration.get("chunk_id"), "") or ""
        return f"tmp:{from_id}:{rel}:{to_id}:{chunk}"
    return None


def _edge_payload(
    edge: dict[str, Any],
    hydration: dict[str, Any],
    *,
    role: str,
    chain_id: str,
    hub_id: str = "",
    hub_name: str = "",
) -> dict[str, Any] | None:
    from_id = _first(hydration.get("from_id"), edge.get("start_id"))
    to_id = _first(hydration.get("to_id"), edge.get("end_id"))
    rel = _first(hydration.get("type"), edge.get("type"), "RELATED")
    if not from_id or not to_id:
        return None

    edge_id = _stable_edge_id(
        edge, hydration, from_id=str(from_id), to_id=str(to_id), rel=str(rel)
    )
    if not edge_id:
        return None

    from_name = str(
        _first(hydration.get("from_name"), edge.get("start"), edge.get("start_name"), "") or ""
    ).strip()
    to_name = str(
        _first(hydration.get("to_name"), edge.get("end"), edge.get("end_name"), "") or ""
    ).strip()
    from_group = str(
        _first(hydration.get("from_label"), edge.get("start_label"), "") or ""
    ).strip()
    to_group = str(
        _first(hydration.get("to_label"), edge.get("end_label"), "") or ""
    ).strip()

    properties: dict[str, Any] = {
        "evidence": _first(hydration.get("evidence"), edge.get("evidence"), "") or "",
        "chunk_id": _first(hydration.get("chunk_id"), edge.get("chunk_id"), "") or "",
        "source_file": _first(hydration.get("source_file"), edge.get("source_file"), "") or "",
        "confidence": _as_optional_float(
            _first(edge.get("confidence"), hydration.get("confidence"))
        ),
        "sim": _as_float(edge.get("sim"), 0.0),
        "source": _first(edge.get("source"), "") or "",
    }

    payload: dict[str, Any] = {
        "id": str(edge_id),
        "from": str(from_id),
        "to": str(to_id),
        "label": str(rel),
        "role": role,
        "chain_ids": [chain_id],
        "from_name": from_name,
        "to_name": to_name,
        "from_group": from_group,
        "to_group": to_group,
        "properties": properties,
    }
    if hub_id:
        payload["hub_name"] = (hub_name or "").strip()
    return payload


def _merge_edge(target: dict[str, dict[str, Any]], edge: dict[str, Any]) -> None:
    existing = target.get(edge["id"])
    if existing is None:
        target[edge["id"]] = edge
        return

    existing["chain_ids"] = sorted(
        set(existing.get("chain_ids") or []) | set(edge.get("chain_ids") or [])
    )
    if existing.get("role") != "spine" and edge.get("role") == "spine":
        existing["role"] = "spine"
    for key in ("from_name", "to_name", "from_group", "to_group", "hub_name"):
        _fill_if_blank(existing, key, edge.get(key))
    existing.setdefault("properties", {})
    _merge_props(existing["properties"], edge.get("properties") or {})


def _merge_node(nodes: dict[str, dict[str, Any]], node: dict[str, Any]) -> None:
    existing = nodes.get(node["id"])
    if existing is None:
        nodes[node["id"]] = {
            "id": node["id"],
            "label": node.get("label") or "Unknown",
            "caption": node.get("caption") or "",
            "group": node.get("group") or "Unknown",
            "color": node.get("color") or DEFAULT_NODE_COLOR,
            "properties": dict(node.get("properties") or {}),
        }
        return

    new_group = node.get("group") or ""
    if existing.get("group") in ("", "Unknown") and new_group not in ("", "Unknown"):
        existing["group"] = new_group
        existing["label"] = node.get("label") or new_group
        existing["color"] = node.get("color") or NODE_COLORS.get(new_group, DEFAULT_NODE_COLOR)
    new_cap = (node.get("caption") or "").strip()
    if new_cap and not (existing.get("caption") or "").strip():
        existing["caption"] = new_cap
        existing.setdefault("properties", {})["name"] = new_cap
    existing.setdefault("properties", {})
    _merge_props(existing["properties"], node.get("properties") or {})


def build_chain_views(
    chains: list[dict[str, Any]],
    hydration: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    hydration = hydration or {}
    views: list[dict[str, Any]] = []

    for idx, chain in enumerate(chains, 1):
        chain_id = str(chain.get("chain_id") or f"a{idx}")
        view_id = f"a{idx}"
        hub_names = chain.get("fan_hub_names") or {}
        nodes: dict[str, dict[str, Any]] = {}
        edges: dict[str, dict[str, Any]] = {}

        for role, edge, hub_id in _iter_chain_edges(chain):
            edge_id = _first(edge.get("element_id"), edge.get("id"))
            row = hydration.get(str(edge_id), {}) if edge_id else {}
            named = ""
            if hub_id:
                named = str(hub_names.get(hub_id) or "").strip()
            payload = _edge_payload(
                edge,
                row,
                role=role,
                chain_id=view_id,
                hub_id=hub_id,
                hub_name=named,
            )
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
            if hub_id and not payload.get("hub_name"):
                hub_node = nodes.get(hub_id)
                if hub_node and hub_node.get("caption"):
                    payload["hub_name"] = hub_node["caption"]
            _fill_if_blank(payload, "from_name", nodes.get(payload["from"], {}).get("caption"))
            _fill_if_blank(payload, "to_name", nodes.get(payload["to"], {}).get("caption"))
            _fill_if_blank(payload, "from_group", nodes.get(payload["from"], {}).get("group"))
            _fill_if_blank(payload, "to_group", nodes.get(payload["to"], {}).get("group"))
            _merge_edge(edges, payload)

        views.append(
            {
                "id": view_id,
                "label": f"Цепь {idx}",
                "score": _as_float(chain.get("score"), 0.0),
                "source_chain_id": chain_id,
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
            _merge_node(nodes, node)
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

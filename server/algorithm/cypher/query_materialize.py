"""Turn one query_graph result into a single viz chain — ids from this RETURN only."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from server.algorithm.cypher.query_compile import VizIdColumn

QUERY_GRAPH_KIND = "query_graph"


@dataclass
class VizRow:
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)


def is_hidden_column(key: object) -> bool:
    name = str(key)
    return name.startswith("__") or name.lower() in {"embedding"} or name.lower().endswith("_embedding")


def split_public_row(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in row.items() if not is_hidden_column(key)}


def extract_viz_row(row: dict[str, Any], viz_ids: list[VizIdColumn] | None) -> VizRow:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    by_alias = {col.alias: col for col in (viz_ids or [])}
    for alias, col in by_alias.items():
        raw = row.get(alias)
        if raw is None or raw == "":
            continue
        eid = str(raw)
        if col.kind == "node":
            _merge_node(nodes, {"id": eid, "element_id": eid, "name": "", "labels": []})
        elif col.kind == "rel":
            _merge_edge(edges, {"element_id": eid, "type": "", "source": QUERY_GRAPH_KIND})
    for key, value in row.items():
        if is_hidden_column(key):
            continue
        _absorb_graphy(value, nodes, edges)
    return VizRow(nodes=list(nodes.values()), edges=list(edges.values()))


def materialize_query_chain(viz_rows: list[VizRow] | None) -> dict[str, Any] | None:
    nodes: dict[str, dict[str, Any]] = {}
    edges: dict[str, dict[str, Any]] = {}
    for row in viz_rows or []:
        for node in row.nodes:
            _merge_node(nodes, node)
        for edge in row.edges:
            _merge_edge(edges, edge)
    if not nodes and not edges:
        return None
    edge_list = list(edges.values())
    return {
        "kind": QUERY_GRAPH_KIND,
        "source": QUERY_GRAPH_KIND,
        "source_graph": "",
        "edges": edge_list,
        "nodes": list(nodes.values()),
        "fans": {},
        "fan_hub_names": {},
        "edge_keys": [
            str(item.get("edge_key") or item.get("element_id") or "")
            for item in edge_list
            if str(item.get("edge_key") or item.get("element_id") or "")
        ],
    }


def _merge_node(target: dict[str, dict[str, Any]], node: dict[str, Any]) -> None:
    eid = str(node.get("element_id") or node.get("id") or "")
    if not eid:
        return
    item = dict(node)
    item["id"] = eid
    item["element_id"] = eid
    existing = target.get(eid)
    if existing is None:
        target[eid] = item
        return
    for key, value in item.items():
        if value in (None, "", [], {}) and existing.get(key) not in (None, "", [], {}):
            continue
        if key in {"name", "caption"} and existing.get(key) and not value:
            continue
        existing[key] = value


def _merge_edge(target: dict[str, dict[str, Any]], edge: dict[str, Any]) -> None:
    eid = str(edge.get("element_id") or edge.get("id") or "")
    if not eid:
        return
    item = dict(edge)
    item["element_id"] = eid
    item.setdefault("source", QUERY_GRAPH_KIND)
    existing = target.get(eid)
    if existing is None:
        target[eid] = item
        return
    for key, value in item.items():
        if value in (None, "", [], {}) and existing.get(key) not in (None, "", [], {}):
            continue
        existing[key] = value


def _absorb_graphy(
    value: Any, nodes: dict[str, dict[str, Any]], edges: dict[str, dict[str, Any]]
) -> None:
    kind = _graphy_kind(value)
    if kind == "node":
        _merge_node(nodes, _node_payload(value))
        return
    if kind == "rel":
        payload = _rel_payload(value)
        _merge_edge(edges, payload)
        start, end = _rel_ends(value)
        if start is not None:
            _merge_node(nodes, _node_payload(start))
        if end is not None:
            _merge_node(nodes, _node_payload(end))
        return
    if kind == "path":
        for node in getattr(value, "nodes", ()) or ():
            _absorb_graphy(node, nodes, edges)
        for rel in getattr(value, "relationships", ()) or ():
            _absorb_graphy(rel, nodes, edges)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            _absorb_graphy(item, nodes, edges)


def _graphy_kind(value: Any) -> str | None:
    try:
        from neo4j.graph import Node, Path, Relationship
    except Exception:
        Node = Path = Relationship = ()  # type: ignore[misc, assignment]
    if Node and isinstance(value, Node):
        return "node"
    if Relationship and isinstance(value, Relationship):
        return "rel"
    if Path and isinstance(value, Path):
        return "path"
    if value is None or isinstance(value, (str, bytes, int, float, bool)):
        return None
    if hasattr(value, "nodes") and hasattr(value, "relationships"):
        return "path"
    if hasattr(value, "element_id") and hasattr(value, "type") and not hasattr(value, "labels"):
        return "rel"
    if hasattr(value, "element_id") and hasattr(value, "labels"):
        return "node"
    return None


def _node_payload(node: Any) -> dict[str, Any]:
    eid = str(getattr(node, "element_id", "") or "")
    name = ""
    if hasattr(node, "get"):
        raw = node.get("name")
        if raw is not None:
            name = str(raw)
    labels = list(getattr(node, "labels", ()) or ())
    return {"id": eid, "element_id": eid, "name": name, "labels": labels}


def _rel_payload(rel: Any) -> dict[str, Any]:
    eid = str(getattr(rel, "element_id", "") or "")
    start, end = _rel_ends(rel)
    evidence = ""
    source_file = ""
    chunk_id = ""
    confidence = None
    if hasattr(rel, "get"):
        evidence = str(rel.get("evidence") or "")
        source_file = str(rel.get("source_file") or "")
        chunk_id = str(rel.get("chunk_id") or "")
        confidence = rel.get("confidence")
    return {
        "element_id": eid,
        "type": str(getattr(rel, "type", "") or ""),
        "start_id": str(getattr(start, "element_id", "") or "") if start is not None else "",
        "end_id": str(getattr(end, "element_id", "") or "") if end is not None else "",
        "start": _node_name(start),
        "end": _node_name(end),
        "evidence": evidence,
        "source_file": source_file,
        "chunk_id": chunk_id,
        "confidence": confidence,
        "source": QUERY_GRAPH_KIND,
    }


def _rel_ends(rel: Any) -> tuple[Any, Any]:
    nodes = getattr(rel, "nodes", None)
    if nodes is not None:
        pair = tuple(nodes)
        if len(pair) >= 2:
            return pair[0], pair[1]
        if len(pair) == 1:
            return pair[0], None
    start = getattr(rel, "start_node", None)
    end = getattr(rel, "end_node", None)
    return start, end


def _node_name(node: Any) -> str:
    if node is None:
        return ""
    if hasattr(node, "get"):
        raw = node.get("name")
        if raw is not None:
            return str(raw)
    return ""

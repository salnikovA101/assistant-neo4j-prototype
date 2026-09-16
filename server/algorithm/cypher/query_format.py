"""Deterministic markdown for query_graph — no extra LLM, no JSON dumps."""

from __future__ import annotations

from typing import Any, Iterable

from server.algorithm.cypher.query_compile import (
    HISTORY_CHARS,
    MAX_FIELD_CHARS,
    MAX_RESULT_CHARS,
)
from server.tools.source_registry import SourceRegistry

NO_MATCHES_PREFIX = "NO_MATCHES"
QUERY_ERROR_PREFIX = "QUERY_ERROR"


def clip(text: str, limit: int) -> tuple[str, bool]:
    raw = text if text is not None else ""
    if len(raw) <= limit:
        return raw, False
    return raw[: max(0, limit - 1)] + "…", True


def format_empty() -> str:
    return (
        f"{NO_MATCHES_PREFIX}\n"
        "No matches for this query. If you used MATCH … CONTAINS, retry with fulltext:\n"
        "CALL db.index.fulltext.queryNodes($__ft_nodes, $q) YIELD node, score "
        "MATCH (node)-[r]-(m) RETURN …\n"
        "and/or CALL db.index.fulltext.queryRelationships($__ft_rels, $q) YIELD relationship, score "
        "MATCH (a)-[relationship]-(b) RETURN …\n"
        "Put formula + English name in parameters q (e.g. KMnO4 permanganate~). "
        "Do not broaden to a generic token first. Do not claim the object is absent from the database."
    )


def format_db_error(message: str, *, timeout: bool = False) -> str:
    clean = " ".join(str(message or "database error").split())
    clean, _ = clip(clean, 400)
    if timeout:
        return (
            f"{QUERY_ERROR_PREFIX} timeout\n"
            "The query was stopped by the time limit. Narrow MATCH, use a shorter path, or a smaller LIMIT. "
            "This is not an empty database."
        )
    lower = clean.lower()
    if "fulltext" in lower or "no such index" in lower or "index" in lower and "query_graph" in lower:
        return (
            f"{QUERY_ERROR_PREFIX} db\n"
            "The database rejected the query: fulltext index is unavailable. "
            "Use toLower(n.name) CONTAINS $q or toLower(r.evidence) CONTAINS $q. "
            "This is not an empty database."
        )
    return (
        f"{QUERY_ERROR_PREFIX} db\n"
        f"The database rejected the query: {clean}. Fix the Cypher. This is not an empty database."
    )


def format_records(
    rows: list[dict[str, Any]],
    *,
    registry: SourceRegistry,
    truncated_rows: bool,
    output_limit: int,
    is_pure_aggregate: bool,
) -> str:
    if not rows:
        return format_empty()
    prepared = [_prepare_row(row, registry) for row in rows]
    if is_pure_aggregate and len(prepared) == 1 and len(prepared[0]) == 1:
        key, value = next(iter(prepared[0].items()))
        body = (
            f"### Graph calculation\n"
            f"Metric: `{key}`\n"
            f"Value: {value}\n\n"
            "Calculation over the current corpus, not a single article."
        )
        return _finish(body, shown=1, truncated_rows=False, output_limit=output_limit, aggregate=True)
    if _looks_like_edges(prepared):
        lines = ["### Relationships"]
        for i, row in enumerate(prepared, 1):
            lines.append(_format_edge(i, row))
        body = "\n".join(lines)
        return _finish(
            body,
            shown=len(prepared),
            truncated_rows=truncated_rows,
            output_limit=output_limit,
            aggregate=False,
        )
    keys = _stable_keys(prepared)
    if len(keys) <= 1 and len(prepared) == 1:
        key = keys[0] if keys else "value"
        body = f"### Result\n{key}: {prepared[0].get(key, '')}"
        return _finish(body, shown=1, truncated_rows=truncated_rows, output_limit=output_limit, aggregate=is_pure_aggregate)
    header = "| " + " | ".join(keys) + " |"
    sep = "| " + " | ".join("---" for _ in keys) + " |"
    lines = ["### Result", header, sep]
    for row in prepared:
        lines.append("| " + " | ".join(_cell(row.get(k, "")) for k in keys) + " |")
    return _finish(
        "\n".join(lines),
        shown=len(prepared),
        truncated_rows=truncated_rows,
        output_limit=output_limit,
        aggregate=is_pure_aggregate,
    )


def history_stub(text: str, *, ok: bool = True) -> str:
    raw = (text or "").strip()
    if not ok:
        msg = raw
        if msg.lower().startswith("error:"):
            msg = msg[6:].strip()
        msg, _ = clip(msg, 200)
        return f"Tool error: {msg}"
    if raw.startswith(QUERY_ERROR_PREFIX) or raw.startswith("TOOL_ERROR"):
        return f"Tool error: {clip(raw, 220)[0]}"
    if raw.startswith(NO_MATCHES_PREFIX):
        return "query_graph: no matches for this Cypher. Not a database outage."
    clipped, cut = clip(raw, HISTORY_CHARS)
    note = " [truncated for history]" if cut else ""
    return f"query_graph result.{note}\n{clipped}"


def _prepare_row(row: dict[str, Any], registry: SourceRegistry) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in row.items():
        if str(key).startswith("_"):
            continue
        text = stringify(value)
        if _is_source_key(key):
            sid = registry.register(text)
            text = f"(source:{sid})" if sid else text
        text, cut = clip(text, MAX_FIELD_CHARS)
        if cut and _is_source_key(key):
            pass
        out[str(key)] = text
    return out


def _is_source_key(key: object) -> bool:
    k = str(key).lower()
    return k in {"source_file", "source", "sourcefile"} or k.endswith(".source_file")


def _looks_like_edges(rows: list[dict[str, str]]) -> bool:
    if not rows:
        return False
    keys = {k.lower() for k in rows[0]}
    return "evidence" in keys and (
        "source_file" in keys or "source" in keys or any("source" in k for k in keys)
    )


def _format_edge(index: int, row: dict[str, str]) -> str:
    names = _name_pair(row)
    rel = _rel_name(row)
    evidence = row.get("evidence") or row.get("r.evidence") or ""
    source = _source_cell(row)
    conf = row.get("confidence") or row.get("r.confidence") or ""
    head = f"{index}. {names[0]} —{rel}→ {names[1]}" if names else f"{index}."
    extra = f"  \"{evidence}\" {source}" if evidence else f"  {source}"
    if conf:
        extra += f" conf={conf}"
    return f"{head}\n{extra}".rstrip()


def _name_pair(row: dict[str, str]) -> tuple[str, str] | None:
    left_keys = ("a", "n", "from", "start", "source_name", "from_name", "a.name", "n.name")
    right_keys = ("b", "m", "to", "end", "target", "to_name", "b.name", "m.name")
    left = _first(row, left_keys)
    right = _first(row, right_keys)
    if left and right:
        return left, right
    keys = [k for k in row if k.lower() not in {"evidence", "source_file", "source", "confidence", "rel", "type", "score", "chunk_id"}]
    if len(keys) >= 2:
        return row[keys[0]], row[keys[1]]
    return None


def _rel_name(row: dict[str, str]) -> str:
    return _first(row, ("rel", "type", "relation", "type(r)", "r")) or "related"


def _source_cell(row: dict[str, str]) -> str:
    return _first(row, ("source_file", "source", "r.source_file")) or ""


def _first(row: dict[str, str], keys: Iterable[str]) -> str:
    lower = {k.lower(): v for k, v in row.items()}
    for key in keys:
        if key.lower() in lower and lower[key.lower()]:
            return lower[key.lower()]
    return ""


def _stable_keys(rows: list[dict[str, str]]) -> list[str]:
    keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys


def _cell(value: str) -> str:
    return (value or "").replace("|", "\\|").replace("\n", " ")


def _finish(
    body: str,
    *,
    shown: int,
    truncated_rows: bool,
    output_limit: int,
    aggregate: bool,
) -> str:
    lines = [body.rstrip(), ""]
    if aggregate:
        lines.append("Shown: calculation over the graph (not a row sample).")
    else:
        cap_note = f"output cap {output_limit}; not a total count"
        if truncated_rows:
            lines.append(f"Shown: {shown} rows ({cap_note}). Output truncated; more rows exist.")
        else:
            lines.append(f"Shown: {shown} rows ({cap_note}).")
    text = "\n".join(lines).rstrip() + "\n"
    clipped, cut = clip(text, MAX_RESULT_CHARS)
    if cut:
        clipped += "\n[Text truncated by the server.]\n"
    return clipped


def stringify(value: Any) -> str:
    if value is None:
        return ""
    try:
        from neo4j.graph import Node, Path, Relationship
    except Exception:  # pragma: no cover - driver always present in app
        Node = Path = Relationship = ()  # type: ignore[misc, assignment]
    if Node and isinstance(value, Node):
        name = value.get("name")
        return str(name) if name is not None else ",".join(value.labels)
    if Relationship and isinstance(value, Relationship):
        ev = value.get("evidence")
        return f"{value.type}: {ev}" if ev else str(value.type)
    if Path and isinstance(value, Path):
        names = []
        for node in value.nodes:
            names.append(str(node.get("name") or ",".join(node.labels)))
        return " — ".join(names)
    if isinstance(value, (list, tuple)):
        inner = ", ".join(stringify(v) for v in value[:8])
        extra = ", …" if len(value) > 8 else ""
        return f"[{inner}{extra}]"
    if isinstance(value, dict):
        items = list(value.items())[:6]
        inner = ", ".join(f"{k}={stringify(v)}" for k, v in items)
        extra = ", …" if len(value) > 6 else ""
        return "{" + inner + extra + "}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value == int(value):
            return str(int(value))
        return f"{value:.4g}"
    return str(value)

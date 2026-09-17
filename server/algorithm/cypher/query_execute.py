"""Read-only Neo4j execution for compiled query_graph Cypher."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neo4j import READ_ACCESS, AsyncDriver, Query
from neo4j.exceptions import ClientError, TransientError

from server.algorithm.cypher.query_compile import (
    QUERY_TIMEOUT_SEC,
    CompiledQuery,
    QueryCompileError,
    merge_params,
)
from server.algorithm.cypher.query_format import format_db_error, stringify
from server.algorithm.cypher.query_materialize import (
    VizRow,
    extract_viz_row,
    is_hidden_column,
    split_public_row,
)


@dataclass
class ExecutedQuery:
    rows: list[dict[str, Any]]
    truncated: bool
    viz_rows: list[VizRow]


async def execute_compiled(
    driver: AsyncDriver,
    compiled: CompiledQuery,
    *,
    user_params: dict[str, Any],
    run_id: str,
) -> ExecutedQuery:
    params = merge_params(compiled, user_params, run_id)
    query = Query(compiled.cypher, timeout=QUERY_TIMEOUT_SEC)
    try:
        async with driver.session(default_access_mode=READ_ACCESS) as session:
            result = await session.run(query, params)
            records = [record.data() async for record in result]
            summary = await result.consume()
    except (ClientError, TransientError) as exc:
        timeout = "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
        raise QueryExecuteError(format_db_error(str(exc), timeout=timeout)) from exc
    except Exception as exc:
        raise QueryExecuteError(format_db_error(str(exc))) from exc
    counters = getattr(summary, "counters", None)
    if counters is not None and getattr(counters, "contains_updates", False):
        raise QueryExecuteError(
            format_db_error("write was rejected by the read session", timeout=False)
        )
    truncated = False
    rows: list[dict[str, Any]] = []
    viz_rows: list[VizRow] = []
    for rec in records:
        viz_rows.append(extract_viz_row(rec, compiled.viz_ids))
        rows.append(_plain(split_public_row(rec)))
    if compiled.fetch_limit is not None and len(rows) > compiled.output_limit:
        truncated = True
        rows = rows[: compiled.output_limit]
        viz_rows = viz_rows[: compiled.output_limit]
    return ExecutedQuery(rows=rows, truncated=truncated, viz_rows=viz_rows)


class QueryExecuteError(Exception):
    def __init__(self, tool_text: str) -> None:
        super().__init__(tool_text)
        self.tool_text = tool_text


def _plain(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if is_hidden_column(key):
            continue
        out[str(key)] = stringify(value) if _is_graphy(value) else value
    return out


def _is_graphy(value: Any) -> bool:
    try:
        from neo4j.graph import Node, Path, Relationship
    except Exception:
        return False
    return isinstance(value, (Node, Path, Relationship)) or (
        isinstance(value, (list, tuple)) and value and _is_graphy(value[0])
    )

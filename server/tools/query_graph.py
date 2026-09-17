"""LLM tool: model writes Cypher, server isolates the corpus and returns markdown."""

from __future__ import annotations

import logging
from typing import Any

from server.algorithm.cypher.query_compile import (
    DEFAULT_MAX_ROWS,
    QUERY_TIMEOUT_SEC,
    QueryCompileError,
    compile_query,
)
from server.algorithm.cypher.query_execute import QueryExecuteError, execute_compiled
from server.algorithm.cypher.query_format import format_empty, format_records
from server.algorithm.cypher.query_materialize import VizRow, materialize_query_chain
from server.core.db import get_driver
from server.core.graph_runs import record_accepted_chains
from server.core.sessions import current_sources
from server.core.turn_state import (
    QUERY_GRAPH_TOOL,
    current_turn,
    finalize_tool_result,
    has_tool_slot,
    is_quota_error_result,
    tool_limit_message,
)
from server.tools.source_registry import SourceRegistry

logger = logging.getLogger(__name__)

QUERY_GRAPH_DESCRIPTION = """\
query_graph: run one read-only Cypher query against the current Neo4j corpus.
You write Cypher; the server injects corpus isolation (run_id) and returns
markdown. Never pass run_id, URI, credentials, or timeouts.

Use for mentions, counts, rankings, filters, neighbors, intersections, bounded
paths, and checks of a named object. For open-ended “find materials about X”
use ask_subgraph (semantic evidence search). You may alternate in one answer.

LIMIT in the Cypher is honored up to 100. If you omit LIMIT (and omit max_rows),
the server applies 20. max_rows is the same cap when Cypher has no LIMIT.

Mentions / “what is in the graph about TERM”: do NOT start with MATCH … CONTAINS.
Use the fulltext indexes (Lucene). Copy this pattern; the server binds index names.
parameters: {"q": "KMnO4 permanganate~"}  (formula + English name; ~ is fuzzy)

CALL db.index.fulltext.queryNodes($__ft_nodes, $q) YIELD node, score
MATCH (node)-[r]-(m)
RETURN node.name AS n, type(r) AS rel, m.name AS m, r.evidence AS evidence,
       r.source_file AS source_file, score
ORDER BY score DESC LIMIT 20

If that is empty or thin, in the same turn also search evidence:

CALL db.index.fulltext.queryRelationships($__ft_rels, $q) YIELD relationship, score
MATCH (a)-[relationship]-(b)
RETURN a.name AS n, type(relationship) AS rel, b.name AS m,
       relationship.evidence AS evidence, relationship.source_file AS source_file, score
ORDER BY score DESC LIMIT 20

Do not invent index names. toLower(n.name) CONTAINS $q is only a fallback after
fulltext NO_MATCHES, or when combining with type/confidence filters. Do not
broaden to a generic token like 'potassium' before trying formula + fulltext.
Vector SEARCH / evidence_embedding is ask_subgraph. Cyrillic in queries misses.

Write ONE Neo4j read query: MATCH, OPTIONAL MATCH, WHERE, WITH, RETURN, DISTINCT,
ORDER BY, SKIP, LIMIT, UNION, count/sum/avg/min/max, CASE, lists, UNWIND,
EXISTS(pattern), named paths *1..4, and the fulltext CALLs above.
English-only strings (translate from Russian first).

Graph: nodes have `name`, labels (omit if unsure), and `run_ids` (server-applied).
Facts live on relationships: type(r), evidence, source_file, chunk_id, confidence
in [0,1]. Do not write run_id. Never RETURN embeddings, properties(), or elementId.
Forbidden: CREATE/MERGE/SET/DELETE/REMOVE, APOC, SHOW, DDL, unbounded *,
shortestPath, any CALL except these fulltext helpers.

action=schema lists live labels and types.

Feedback: NO_MATCHES = no rows for this Cypher, not “the object is absent from
the database”. If you used CONTAINS, retry the fulltext templates (names, then
evidence); do not broaden to potassium. QUERY_ERROR <code> — read the code
(syntax / unsupported / scope / db / timeout) and fix the Cypher; do not
restate the user question. QUERY_ERROR does not spend a call slot. Server limit:
8 successful query_graph calls per answer (schema counts). Each result ends with
n/8; if this tool is exhausted, use ask_subgraph if it still has slots. If both
limits are spent, the result says tools are exhausted — answer now. Sources in
the tool markdown are already folded to (source:N); the user sees the same
document names in their source list. Do not say the graph only stores file
numbers.
"""

_SCHEMA_CYPHER_LABELS = """
MATCH (n)-[r]-()
WHERE r.run_id = $__run_id
  AND trim(coalesce(toString(r.evidence), '')) <> ''
UNWIND labels(n) AS label
RETURN DISTINCT label ORDER BY label
"""
_SCHEMA_CYPHER_TYPES = """
MATCH ()-[r]->()
WHERE r.run_id = $__run_id
  AND trim(coalesce(toString(r.evidence), '')) <> ''
RETURN DISTINCT type(r) AS relationshipType
ORDER BY relationshipType
"""


class QueryGraphTool:
    def __init__(self, source_registry: SourceRegistry) -> None:
        self.source_registry = source_registry

    async def __call__(
        self,
        cypher: str | None = None,
        parameters: dict[str, Any] | None = None,
        max_rows: int | None = None,
        action: str | None = None,
        **ignored: Any,
    ) -> str:
        if ignored:
            logger.info("query_graph: ignoring extra arguments %s", list(ignored))
        act = (action or "query").strip().lower()
        if not has_tool_slot(QUERY_GRAPH_TOOL):
            return tool_limit_message(QUERY_GRAPH_TOOL)
        if act == "schema":
            text = await self._schema()
        elif act not in {"query", ""}:
            text = (
                "QUERY_ERROR syntax\n"
                "Broken here: action must be query or schema. Fix the call; do not restate the user question."
            )
        else:
            text = await self._query(cypher, parameters, max_rows)
        return finalize_tool_result(
            QUERY_GRAPH_TOOL,
            text,
            success=not is_quota_error_result(text),
        )

    async def _query(
        self,
        cypher: str | None,
        parameters: dict[str, Any] | None,
        max_rows: int | None,
    ) -> str:
        turn = current_turn()
        run_id = (turn.run_id if turn else "") or ""
        if not run_id:
            return (
                "QUERY_ERROR scope\n"
                "The query cannot be isolated to this corpus: missing run_id in the server turn context."
            )
        try:
            compiled = compile_query(cypher or "", parameters=parameters, max_rows=max_rows)
        except QueryCompileError as exc:
            return exc.as_tool_text()
        user_params = dict(parameters or {})
        try:
            executed = await execute_compiled(
                get_driver(),
                compiled,
                user_params=user_params,
                run_id=run_id,
            )
        except QueryExecuteError as exc:
            return exc.tool_text
        except QueryCompileError as exc:
            return exc.as_tool_text()
        rows, truncated = executed.rows, executed.truncated
        registry = current_sources() or self.source_registry
        if turn is not None and turn.store is not None and turn.conversation_id:
            snapshot = await turn.store.conversation_source_snapshot(turn.conversation_id)
            registry.restore(snapshot)
        text = format_records(
            rows,
            registry=registry,
            truncated_rows=truncated,
            output_limit=compiled.output_limit,
            is_pure_aggregate=compiled.is_pure_aggregate,
        )
        source_files = [
            registry.resolve(sid) or ""
            for sid in sorted({int(s) for s in _source_ids(text)})
        ]
        source_files = [f for f in source_files if f]
        if turn is not None and turn.store is not None and turn.conversation_id and source_files:
            snapshot = await turn.store.merge_conversation_sources(
                turn.conversation_id, source_files
            )
            registry.restore(snapshot)
        if not rows:
            return format_empty()
        _record_query_graph_chain(executed.viz_rows)
        return text

    async def _schema(self) -> str:
        turn = current_turn()
        run_id = (turn.run_id if turn else "") or ""
        if not run_id:
            return (
                "QUERY_ERROR scope\n"
                "The query cannot be isolated to this corpus: missing run_id in the server turn context."
            )
        from neo4j import READ_ACCESS, Query

        driver = get_driver()
        try:
            async with driver.session(default_access_mode=READ_ACCESS) as session:
                labels = [
                    str(row["label"])
                    async for row in await session.run(
                        Query(_SCHEMA_CYPHER_LABELS, timeout=QUERY_TIMEOUT_SEC),
                        {"__run_id": run_id},
                    )
                ]
                rels = [
                    str(row["relationshipType"])
                    async for row in await session.run(
                        Query(_SCHEMA_CYPHER_TYPES, timeout=QUERY_TIMEOUT_SEC),
                        {"__run_id": run_id},
                    )
                ]
        except Exception as exc:
            from server.algorithm.cypher.query_format import format_db_error
            return format_db_error(str(exc))
        label_line = ", ".join(labels[:80]) or "(none)"
        rel_line = ", ".join(rels[:80]) or "(none)"
        return (
            "### Corpus schema\n"
            "This corpus only (evidence edges with the current run_id; not the whole database).\n"
            f"Node labels: {label_line}\n"
            f"Relationship types: {rel_line}\n"
            "Node fields: name, labels(n), run_ids (server-applied)\n"
            "Relationship fields: type, evidence, source_file, chunk_id, confidence\n"
            "Do not query embeddings. run_id is applied by the server.\n"
            "Fulltext first (do not invent index names):\n"
            "CALL db.index.fulltext.queryNodes($__ft_nodes, $q) YIELD node, score "
            "MATCH (node)-[r]-(m) RETURN …\n"
            "CALL db.index.fulltext.queryRelationships($__ft_rels, $q) YIELD relationship, score "
            "MATCH (a)-[relationship]-(b) RETURN …\n"
            "CONTAINS only after fulltext NO_MATCHES.\n"
        )


def _source_ids(text: str) -> list[int]:
    from server.tools.source_registry import session_source_ids_in_text
    return session_source_ids_in_text(text)


def _record_query_graph_chain(viz_rows: list[VizRow]) -> None:
    chain = materialize_query_chain(viz_rows)
    if chain is None:
        return
    record_accepted_chains([chain])


def query_graph_openai_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "query_graph",
            "description": QUERY_GRAPH_DESCRIPTION,
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["query", "schema"],
                        "description": "query (default) runs Cypher; schema lists labels and relationship types.",
                    },
                    "cypher": {
                        "type": "string",
                        "description": "One read-only Neo4j Cypher query. Required for action=query. Do not include run_id.",
                    },
                    "parameters": {
                        "type": "object",
                        "description": "Cypher $parameters. Do not pass run_id or __* names.",
                    },
                    "max_rows": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "description": (
                            "Output row cap when Cypher has no LIMIT "
                            f"(default {DEFAULT_MAX_ROWS}, max 100). "
                            "Ignored if Cypher already has LIMIT (that LIMIT is honored up to 100). "
                            "Does not pre-cut aggregations."
                        ),
                    },
                },
            },
        },
    }

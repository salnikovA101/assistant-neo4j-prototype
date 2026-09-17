"""Optional live Neo4j checks against the local corpus (read-only)."""

from __future__ import annotations

import os

import pytest

from server.algorithm.cypher.query_compile import compile_query
from server.algorithm.cypher.query_execute import execute_compiled
from server.algorithm.cypher.query_format import format_records
from server.core.db import close_driver, get_driver, init_driver
from server.core.turn_state import bind_turn
from server.tools.query_graph import QueryGraphTool
from server.tools.source_registry import SourceRegistry
from server.utils.config import load_config


def _run_id() -> str:
    cfg = load_config()
    return os.environ.get("QUERY_GRAPH_RUN_ID") or cfg.workspaces.get("packaging") or ""


async def _driver_or_skip():
    cfg = load_config()
    uris = [cfg.neo4j.uri]
    if "host.docker.internal" in cfg.neo4j.uri:
        uris.append(cfg.neo4j.uri.replace("host.docker.internal", "localhost"))
        uris.append(cfg.neo4j.uri.replace("host.docker.internal", "127.0.0.1"))
    last = None
    for uri in uris:
        try:
            await close_driver()
        except Exception:
            pass
        try:
            init_driver(uri, cfg.neo4j.user, cfg.neo4j.password)
            driver = get_driver()
            async with driver.session() as session:
                await session.run("RETURN 1 AS ok")
            return driver
        except Exception as exc:
            last = exc
    pytest.skip(f"Neo4j not reachable: {last}")


@pytest.mark.asyncio
async def test_live_mentions_and_source_ranking():
    driver = await _driver_or_skip()
    run_id = _run_id()
    if not run_id:
        pytest.skip("no packaging run_id")
    compiled = compile_query(
        "MATCH (a)-[r]-(b) "
        "WHERE toLower(a.name) CONTAINS $q OR toLower(b.name) CONTAINS $q "
        "OR toLower(coalesce(r.evidence,'')) CONTAINS $q "
        "RETURN a.name AS a, type(r) AS rel, b.name AS b, r.evidence AS evidence, "
        "r.source_file AS source_file LIMIT 5",
        parameters={"q": "kefir"},
        max_rows=5,
    )
    executed = await execute_compiled(
        driver, compiled, user_params={"q": "kefir"}, run_id=run_id
    )
    rows, truncated = executed.rows, executed.truncated
    text = format_records(
        rows,
        registry=SourceRegistry(),
        truncated_rows=truncated,
        output_limit=5,
        is_pure_aggregate=False,
    )
    assert "QUERY_ERROR" not in text
    assert all("__id_" not in row for row in rows)
    if rows:
        assert any(item.nodes or item.edges for item in executed.viz_rows)
    ranking = compile_query(
        "MATCH ()-[r]->() RETURN r.source_file AS source_file, count(r) AS edge_count "
        "ORDER BY edge_count DESC LIMIT 5"
    )
    ranked = await execute_compiled(driver, ranking, user_params={}, run_id=run_id)
    rows = ranked.rows
    assert rows
    assert "edge_count" in rows[0]
    assert ranking.viz_ids == []
    assert all(not item.nodes and not item.edges for item in ranked.viz_rows)


@pytest.mark.asyncio
async def test_live_fulltext_name_lookup():
    driver = await _driver_or_skip()
    run_id = _run_id()
    compiled = compile_query(
        "CALL db.index.fulltext.queryNodes($__ft_nodes, $q) YIELD node, score "
        "MATCH (node)-[r]-(m) "
        "RETURN node.name AS n, type(r) AS rel, m.name AS m, r.evidence AS evidence, "
        "r.source_file AS source_file "
        "ORDER BY score DESC LIMIT 5",
        parameters={"q": "kefir~"},
        max_rows=5,
    )
    executed = await execute_compiled(
        driver, compiled, user_params={"q": "kefir~"}, run_id=run_id
    )
    rows, truncated = executed.rows, executed.truncated
    text = format_records(
        rows,
        registry=SourceRegistry(),
        truncated_rows=truncated,
        output_limit=5,
        is_pure_aggregate=False,
    )
    assert "QUERY_ERROR" not in text
    assert compiled.viz_ids
    if rows:
        assert any(item.edges or item.nodes for item in executed.viz_rows)


@pytest.mark.asyncio
async def test_live_create_never_reaches_neo4j():
    await _driver_or_skip()
    tool = QueryGraphTool(SourceRegistry())
    with bind_turn("medium", max_searches=2, context={"run_id": _run_id()}):
        text = await tool(cypher="MATCH (n) SET n.hacked = true RETURN n")
    assert text.startswith("QUERY_ERROR unsupported")


@pytest.mark.asyncio
async def test_live_schema_action():
    await _driver_or_skip()
    tool = QueryGraphTool(SourceRegistry())
    with bind_turn("medium", max_searches=2, context={"run_id": _run_id()}):
        text = await tool(action="schema")
    assert text.startswith("### Corpus schema")
    assert "This corpus only" in text
    assert "Node labels:" in text
    assert "Relationship types:" in text
    assert "run_ids" in text
    assert "db.labels" not in text

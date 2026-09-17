"""Per-tool successful-call quotas for ask_subgraph and query_graph."""

from __future__ import annotations

import pytest

from server.algorithm.cypher.query_execute import ExecutedQuery
from server.algorithm.cypher.query_materialize import VizRow
from server.core.turn_state import (
    ASK_SUBGRAPH_TOOL,
    BOTH_EXHAUSTED_PHRASE,
    QUERY_GRAPH_TOOL,
    bind_turn,
    tool_quota_state,
)
from server.tools.query_graph import QueryGraphTool
from server.tools.source_registry import SourceRegistry
from server.tools.subgraph_search import NO_RESULTS, TOOL_ERROR, SubgraphSearchAgent


def _empty_executed(*_args, **_kwargs) -> ExecutedQuery:
    return ExecutedQuery(rows=[], truncated=False, viz_rows=[VizRow()])


@pytest.mark.asyncio
async def test_query_graph_eight_successes_block_ninth_without_driver(monkeypatch):
    calls: list[int] = []

    async def fake_exec(*args, **kwargs):
        del args, kwargs
        calls.append(1)
        return _empty_executed()

    monkeypatch.setattr("server.tools.query_graph.execute_compiled", fake_exec)
    monkeypatch.setattr("server.tools.query_graph.get_driver", lambda: object())
    tool = QueryGraphTool(SourceRegistry())
    cypher = "MATCH (a)-[r]-(b) RETURN a.name LIMIT 1"
    with bind_turn("medium", max_query=8, context={"run_id": "run-1"}):
        for n in range(8):
            text = await tool(cypher=cypher)
            assert text.startswith("NO_MATCHES")
            assert f"query_graph {n + 1}/8" in text
            assert "ask_subgraph 0/2 remaining" in text
            assert tool_quota_state(QUERY_GRAPH_TOOL) == (n + 1, 8)
        ninth = await tool(cypher=cypher)
        assert ninth.startswith(TOOL_ERROR)
        assert "query_graph limit" in ninth
        assert "8/8" in ninth
        assert "Use ask_subgraph" in ninth
        assert BOTH_EXHAUSTED_PHRASE not in ninth
        assert tool_quota_state(QUERY_GRAPH_TOOL) == (8, 8)
    assert len(calls) == 8


@pytest.mark.asyncio
async def test_query_graph_compile_error_does_not_spend_slot():
    tool = QueryGraphTool(SourceRegistry())
    with bind_turn("medium", max_query=8, context={"run_id": "run-1"}):
        err = await tool(cypher="CREATE (n) RETURN n")
        assert err.startswith("QUERY_ERROR unsupported")
        assert tool_quota_state(QUERY_GRAPH_TOOL) == (0, 8)
        assert "query_graph 0/8 used" in err
        assert "You may call it again" in err


@pytest.mark.asyncio
async def test_query_graph_schema_success_spends_slot(monkeypatch):
    async def fake_schema(self):
        del self
        return "### Corpus schema\nNode labels: Thing\n"

    monkeypatch.setattr(QueryGraphTool, "_schema", fake_schema)
    tool = QueryGraphTool(SourceRegistry())
    with bind_turn("medium", max_query=8, context={"run_id": "run-1"}):
        text = await tool(action="schema")
        assert "Corpus schema" in text
        assert "query_graph 1/8 used" in text
        assert tool_quota_state(QUERY_GRAPH_TOOL) == (1, 8)


@pytest.mark.asyncio
async def test_both_tool_limits_append_answer_now_on_last_success(monkeypatch):
    async def fake_run(driver, **kwargs):
        del driver, kwargs
        return {"accepted": []}

    async def fake_exec(*args, **kwargs):
        del args, kwargs
        return _empty_executed()

    monkeypatch.setattr("server.algorithm.pipeline.run", fake_run)
    monkeypatch.setattr("server.tools.subgraph_search.get_driver", lambda: object())
    monkeypatch.setattr("server.tools.query_graph.execute_compiled", fake_exec)
    monkeypatch.setattr("server.tools.query_graph.get_driver", lambda: object())
    ask = SubgraphSearchAgent()
    query = QueryGraphTool(SourceRegistry())
    with bind_turn("medium", max_searches=2, max_query=8, context={"run_id": "run-1"}):
        first_ask = await ask.query(["Which starter cultures are used in kefir?"])
        second_ask = await ask.query(["How does temperature affect syneresis in kefir?"])
        assert first_ask.startswith(NO_RESULTS)
        assert BOTH_EXHAUSTED_PHRASE not in first_ask
        assert BOTH_EXHAUSTED_PHRASE not in second_ask
        for n in range(7):
            text = await query(cypher="MATCH (a)-[r]-(b) RETURN a.name LIMIT 1")
            assert f"query_graph {n + 1}/8" in text
            assert BOTH_EXHAUSTED_PHRASE not in text
        last = await query(cypher="MATCH (a)-[r]-(b) RETURN a.name LIMIT 1")
        assert "query_graph 8/8 exhausted" in last
        assert "ask_subgraph 2/2 exhausted" in last
        assert BOTH_EXHAUSTED_PHRASE in last
        refused = await query(cypher="MATCH (a)-[r]-(b) RETURN a.name LIMIT 1")
        assert refused.startswith(TOOL_ERROR)
        assert BOTH_EXHAUSTED_PHRASE in refused
        assert tool_quota_state(ASK_SUBGRAPH_TOOL) == (2, 2)
        assert tool_quota_state(QUERY_GRAPH_TOOL) == (8, 8)

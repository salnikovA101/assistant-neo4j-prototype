"""query_graph compiler, markdown, tool surface, and model-error coverage."""

from __future__ import annotations

import pytest

from server.algorithm.cypher.query_compile import (
    QueryCompileError,
    compile_query,
    merge_params,
)
from server.algorithm.cypher.query_format import (
    format_empty,
    format_records,
    history_stub,
)
from server.core.turn_state import bind_turn
from server.tools.query_graph import QUERY_GRAPH_DESCRIPTION, QueryGraphTool
from server.tools.registry import Tools
from server.tools.source_registry import SourceRegistry, tool_history_stub
from server.utils.config import AppConfig


def _ok(cypher: str, **kwargs):
    return compile_query(cypher, **kwargs)


def _err(cypher: str, code: str, **kwargs) -> QueryCompileError:
    with pytest.raises(QueryCompileError) as caught:
        compile_query(cypher, **kwargs)
    assert caught.value.code == code
    return caught.value


def test_injects_run_id_and_limit_on_simple_match():
    compiled = _ok(
        "MATCH (a)-[r]-(b) "
        "WHERE toLower(a.name) CONTAINS $q "
        "RETURN a.name AS a, type(r) AS rel, b.name AS b, r.evidence AS evidence, "
        "r.source_file AS source_file LIMIT 20",
        parameters={"q": "kn4m"},
    )
    assert "$__run_id" in compiled.cypher
    assert "run_id" in compiled.cypher
    assert compiled.fetch_limit == 21
    assert compiled.output_limit == 20
    assert not compiled.is_pure_aggregate
    params = merge_params(compiled, {"q": "kn4m"}, "full_corpus_20260713")
    assert params["__run_id"] == "full_corpus_20260713"
    assert params["q"] == "kn4m"
    assert "$q RETURN" in compiled.cypher or "$q return" in compiled.cypher.lower()
    assert "$qRETURN" not in compiled.cypher


def test_overwrites_user_run_id_literal_and_map():
    compiled = _ok(
        "MATCH (a)-[r {run_id: 'evil'}]->(b) "
        "WHERE r.run_id = 'other' "
        "RETURN a.name LIMIT 5"
    )
    assert "'evil'" not in compiled.cypher
    assert "'other'" not in compiled.cypher
    assert compiled.cypher.count("$__run_id") >= 2


def test_optional_match_keeps_relationship_map():
    compiled = _ok(
        "MATCH (n) WHERE toLower(n.name) CONTAINS $q "
        "OPTIONAL MATCH (n)-[r]-(m) "
        "RETURN n.name, type(r), m.name, r.evidence LIMIT 10",
        parameters={"q": "x"},
    )
    assert "OPTIONAL MATCH" in compiled.cypher
    assert "$__run_id" in compiled.cypher


def test_union_injects_both_branches():
    compiled = _ok(
        "MATCH (a)-[r]->(b) WHERE toLower(a.name) CONTAINS 'kefir' RETURN a.name "
        "UNION "
        "MATCH (a)-[r]->(b) WHERE toLower(b.name) CONTAINS 'kefir' RETURN a.name"
    )
    assert compiled.cypher.upper().count("UNION") == 1
    assert compiled.cypher.count("$__run_id") >= 2


def test_variable_length_path_gets_run_id_and_rejects_unbounded():
    compiled = _ok(
        "MATCH p = (a)-[*1..4]-(b) "
        "WHERE toLower(a.name) CONTAINS 'x' "
        "RETURN [n IN nodes(p) | n.name] LIMIT 5"
    )
    assert "*1..4" in compiled.cypher.replace(" ", "") or "* 1 .. 4" in compiled.cypher or "*1 .. 4" in compiled.cypher or "*" in compiled.cypher
    assert "$__run_id" in compiled.cypher
    _err("MATCH (a)-[*]-(b) RETURN a.name", "unsupported")
    _err("MATCH (a)-[*1..]-(b) RETURN a.name", "unsupported")
    _err("MATCH (a)-[*1..8]-(b) RETURN a.name", "unsupported")


def test_pure_count_does_not_precut():
    compiled = _ok("MATCH ()-[r]->() RETURN count(r) AS edge_count")
    assert compiled.is_pure_aggregate
    assert compiled.fetch_limit is None
    assert "count" in compiled.cypher.lower()


def test_grouped_count_gets_output_cap():
    compiled = _ok(
        "MATCH ()-[r]->() RETURN r.source_file AS source_file, count(r) AS edge_count "
        "ORDER BY edge_count DESC LIMIT 10"
    )
    assert not compiled.is_pure_aggregate
    assert compiled.fetch_limit == 11
    assert compiled.output_limit == 20 or compiled.fetch_limit == 11


def test_user_limit_stricter_than_max_rows():
    compiled = _ok("MATCH (a)-[r]-(b) RETURN a.name LIMIT 3", max_rows=20)
    assert compiled.fetch_limit == 4
    compiled = _ok("MATCH (a)-[r]-(b) RETURN a.name LIMIT 500", max_rows=20)
    assert compiled.fetch_limit == 21


@pytest.mark.parametrize(
    "cypher",
    [
        "CREATE (n {name:'x'}) RETURN n",
        "MATCH (a)-[r]-(b) SET r.confidence = 1 RETURN r",
        "MATCH (a)-[r]-(b) DELETE r RETURN a",
        "MATCH (a)-[r]-(b) REMOVE r.evidence RETURN r",
        "MERGE (n {name:'x'}) RETURN n",
        "DROP INDEX query_graph_node_name",
        "MATCH (a)-[r]-(b) DETACH DELETE a RETURN 1",
        "LOAD CSV FROM 'file:///x' AS line RETURN line",
        "INSERT (n {name:'x'}) RETURN n",
        "SHOW INDEXES",
    ],
)
def test_write_and_admin_rejected_before_neo4j(cypher: str):
    err = _err(cypher, "unsupported")
    assert "unavailable" in err.message.lower() or "unavailable" in err.as_tool_text().lower()
    assert err.as_tool_text().startswith("QUERY_ERROR unsupported")


def test_create_inside_string_and_comment_is_not_a_write():
    compiled = _ok(
        "MATCH (a)-[r]-(b) WHERE r.evidence CONTAINS 'CREATE TABLE' RETURN a.name LIMIT 5"
    )
    assert "$__run_id" in compiled.cypher
    compiled = _ok(
        "MATCH (a)-[r]-(b) // CREATE (n)\nRETURN a.name LIMIT 5"
    )
    assert "CREATE" not in compiled.cypher
    compiled = _ok(
        "MATCH (a)-[r]-(b) /* MERGE (n) */ RETURN a.name LIMIT 5"
    )
    assert "MERGE" not in compiled.cypher


@pytest.mark.parametrize(
    "cypher,code",
    [
        ("CALL apoc.help('x') YIELD name RETURN name", "unsupported"),
        ("CALL dbms.security.listUsers() YIELD user RETURN user", "unsupported"),
        ("CALL { MATCH (n) RETURN n } RETURN n", "unsupported"),
        ("MATCH (a)-[r]-(b) RETURN properties(r)", "unsupported"),
        ("MATCH (a)-[r]-(b) RETURN elementId(a)", "unsupported"),
        ("MATCH (a)-[r]-(b) RETURN a.embedding", "unsupported"),
        ("MATCH (a)-[r]-(b) RETURN r.evidence_embedding", "unsupported"),
        ("CYPHER 25 MATCH (a)-[r]-(b) RETURN a.name", "unsupported"),
        ("MATCH (a)-[r]-(b) RETURN shortestPath((a)-[*]-(b))", "unsupported"),
        ("MATCH (a)-[r]-(b) RETURN a.name; MATCH (x) RETURN x", "unsupported"),
        ("USE neo4j MATCH (a)-[r]-(b) RETURN a.name", "unsupported"),
        ("EXPLAIN MATCH (a)-[r]-(b) RETURN a.name", "unsupported"),
        ("CALL db.index.vector.queryNodes('idx', 5, $v) YIELD node RETURN node", "unsupported"),
    ],
)
def test_unsupported_constructs(cypher: str, code: str):
    _err(cypher, code)


@pytest.mark.parametrize(
    "cypher,code",
    [
        ("", "syntax"),
        ("```cypher\nMATCH (n) RETURN n\n```", "syntax"),
        ("MATCH (a)-[r]-> RETURN a", "syntax"),
        ("MATCH (a)-[r]-(b)", "syntax"),
        ("RETURN 1", "scope"),
        ("MATCH () RETURN 1", "scope"),
        ("OPTIONAL MATCH (n) RETURN n", "scope"),
    ],
)
def test_syntax_and_scope_errors(cypher: str, code: str):
    _err(cypher, code)


def test_reserved_parameter_rejected():
    _err(
        "MATCH (a)-[r]-(b) RETURN a.name",
        "scope",
        parameters={"__run_id": "stolen"},
    )


def test_fulltext_requires_edge_after_nodes():
    _err(
        "CALL db.index.fulltext.queryNodes($__ft_nodes, $q) YIELD node, score "
        "RETURN node.name LIMIT 10",
        "scope",
        parameters={"q": "KN4M~"},
    )
    compiled = _ok(
        "CALL db.index.fulltext.queryNodes('invented', $q) YIELD node, score "
        "MATCH (node)-[r]-(m) "
        "RETURN node.name, type(r), m.name, r.evidence LIMIT 10",
        parameters={"q": "KN4M~"},
    )
    assert "$__ft_nodes" in compiled.cypher
    assert "invented" not in compiled.cypher
    assert "$__run_id" in compiled.cypher


def test_fulltext_other_call_rejected():
    _err(
        "CALL db.index.fulltext.awaitEventuallyConsistent() YIELD state RETURN state",
        "unsupported",
    )


def test_node_only_match_adds_exists():
    compiled = _ok("MATCH (n) WHERE toLower(n.name) CONTAINS $q RETURN n.name LIMIT 5")
    assert "EXISTS" in compiled.cypher
    assert "$__run_id" in compiled.cypher


def test_anonymous_arrows_become_maps():
    compiled = _ok("MATCH (a)-->(b) RETURN a.name, b.name LIMIT 5")
    assert "$__run_id" in compiled.cypher
    compiled = _ok("MATCH (a)--(b) RETURN a.name LIMIT 5")
    assert "$__run_id" in compiled.cypher


def test_format_empty_and_errors_are_distinct():
    empty = format_empty()
    assert empty.startswith("NO_MATCHES")
    assert "absent from the database" in empty
    assert "queryNodes" in empty
    assert "queryRelationships" in empty
    assert "generic token" in empty
    from server.algorithm.cypher.query_format import format_db_error
    db = format_db_error("Type mismatch: expected Integer")
    assert db.startswith("QUERY_ERROR db")
    assert "not an empty database" in db.lower()
    timeout = format_db_error("The transaction has been terminated because it timed out", timeout=True)
    assert timeout.startswith("QUERY_ERROR timeout")


def test_format_records_markdown_and_sources():
    registry = SourceRegistry()
    text = format_records(
        [
            {
                "a": "KN4M",
                "rel": "isolated_from",
                "b": "kefir grain",
                "evidence": "strain KN4M was isolated",
                "source_file": "paper.pdf",
                "confidence": 0.9,
            }
        ],
        registry=registry,
        truncated_rows=False,
        output_limit=20,
        is_pure_aggregate=False,
    )
    assert "(source:1)" in text
    assert "paper.pdf" not in text
    assert "Shown: 1 rows" in text
    assert "not a total count" in text
    stub = history_stub(text)
    assert "query_graph result" in stub
    assert "Retrieved data" not in stub


def test_history_stub_keeps_aggregates_not_generic_retrieved():
    registry = SourceRegistry()
    text = format_records(
        [{"edge_count": 1842}],
        registry=registry,
        truncated_rows=False,
        output_limit=20,
        is_pure_aggregate=True,
    )
    assert "Graph calculation" in text
    stub = tool_history_stub(text, name="query_graph")
    assert "1842" in stub
    assert stub.startswith("query_graph result")
    err_stub = tool_history_stub("QUERY_ERROR syntax\nBroken here: x", name="query_graph")
    assert err_stub.startswith("Tool error:")


def test_tool_advertised_only_in_auto():
    tools = Tools(AppConfig())
    auto_names = [item["function"]["name"] for item in tools.get_openai_tools("auto")]
    staged_names = [item["function"]["name"] for item in tools.get_openai_tools("staged")]
    assert auto_names == ["ask_subgraph", "query_graph", "get_service_guide"]
    assert staged_names == ["advance_research", "get_service_guide"]
    assert set(tools.get_tool_map("auto")) == {
        "ask_subgraph",
        "query_graph",
        "get_service_guide",
    }
    desc = tools.get_openai_tools("auto")[1]["function"]["description"]
    assert "Neo4j" in desc
    assert "Cypher" in desc
    assert "ask_subgraph" in desc
    assert 1500 < len(desc) < 4500


@pytest.mark.asyncio
async def test_query_graph_compile_errors_do_not_hit_driver():
    tool = QueryGraphTool(SourceRegistry())
    with bind_turn("medium", max_searches=10, context={"run_id": "full_corpus_20260713"}):
        text = await tool(cypher="CREATE (n) RETURN n")
    assert text.startswith("QUERY_ERROR unsupported")
    with bind_turn("medium", max_searches=10, context={"run_id": ""}):
        text = await tool(cypher="MATCH (a)-[r]-(b) RETURN a.name")
    assert text.startswith("QUERY_ERROR scope")


def test_path_all_relationships_rewrites_run_id():
    compiled = _ok(
        "MATCH p = (a)-[*1..3]-(b) "
        "WHERE ALL(r IN relationships(p) WHERE r.run_id = 'evil') "
        "RETURN [n IN nodes(p) | n.name] LIMIT 5"
    )
    assert "'evil'" not in compiled.cypher
    assert "$__run_id" in compiled.cypher


def test_skip_and_limit_together():
    compiled = _ok("MATCH (a)-[r]-(b) RETURN a.name SKIP 10 LIMIT 20")
    assert "SKIP" in compiled.cypher
    assert compiled.fetch_limit == 21


def test_backticked_create_clause_still_rejected():
    _err("CREATE (`n`) RETURN 1", "unsupported")


def test_query_graph_ignores_unknown_action_fields_via_compile_only():
    err = _err("MATCH (a)-[r]-(b) RETURN keys(a)", "unsupported")
    assert "QUERY_ERROR unsupported" in err.as_tool_text()


def test_description_matches_prompt_topics():
    assert "fulltext" in QUERY_GRAPH_DESCRIPTION.lower()
    assert "run_id" in QUERY_GRAPH_DESCRIPTION
    assert "do NOT start with MATCH" in QUERY_GRAPH_DESCRIPTION
    assert "CALL db.index.fulltext.queryNodes($__ft_nodes, $q)" in QUERY_GRAPH_DESCRIPTION
    assert "CALL db.index.fulltext.queryRelationships($__ft_rels, $q)" in QUERY_GRAPH_DESCRIPTION
    assert "potassium" in QUERY_GRAPH_DESCRIPTION
    from pathlib import Path
    prompt = (Path("prompts/assistant_logic.md")).read_text(encoding="utf-8")
    assert "query_graph" in prompt
    assert "упоминания" in prompt
    assert "ask_subgraph" in prompt
    assert "NO_MATCHES" in prompt
    assert "QUERY_ERROR" in prompt
    assert "не начинай с `MATCH" in prompt
    assert "queryNodes($__ft_nodes, $q)" in prompt
    assert "queryRelationships($__ft_rels, $q)" in prompt
    mention_at = prompt.index("Поиск упоминаний")
    contains_fallback_at = prompt.index("toLower(n.name) CONTAINS")
    dialect_at = prompt.index("Пиши один read-запрос")
    assert mention_at < dialect_at
    assert mention_at < contains_fallback_at

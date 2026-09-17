"""query_graph answer-graph: hidden elementId columns, one chain, no name lookup."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from server.algorithm.cypher.query_compile import VizIdColumn, compile_query
from server.algorithm.cypher.query_execute import execute_compiled
from server.algorithm.cypher.query_format import format_records
from server.algorithm.cypher.query_materialize import (
    QUERY_GRAPH_KIND,
    VizRow,
    extract_viz_row,
    materialize_query_chain,
    split_public_row,
)
from server.core.app_store import AppStore
from server.core.graph_runs import (
    current_graph_collector,
    new_graph_collector,
    record_accepted_chains,
    reset_graph_collector,
)
from server.core.turn_state import bind_turn
from server.tools.query_graph import QueryGraphTool
from server.tools.source_registry import SourceRegistry
from server.tools.graph_viz import build_chain_views


class FakeNode:
    def __init__(self, eid: str, name: str = "", labels: tuple[str, ...] = ()):
        self.element_id = eid
        self.labels = frozenset(labels)
        self._props = {"name": name}

    def get(self, key, default=None):
        return self._props.get(key, default)


class FakeRel:
    def __init__(self, eid: str, typ: str, start: FakeNode, end: FakeNode, evidence: str = ""):
        self.element_id = eid
        self.type = typ
        self.nodes = (start, end)
        self._props = {"evidence": evidence, "source_file": "paper.pdf"}

    def get(self, key, default=None):
        return self._props.get(key, default)


class FakePath:
    def __init__(self, nodes: list[FakeNode], rels: list[FakeRel]):
        self.nodes = nodes
        self.relationships = rels


class FakeRecord:
    def __init__(self, data: dict):
        self._data = data

    def data(self) -> dict:
        return dict(self._data)


class FakeResult:
    def __init__(self, rows: list[dict]):
        self._rows = rows

    def __aiter__(self):
        async def _gen():
            for row in self._rows:
                yield FakeRecord(row)

        return _gen()

    async def consume(self):
        return SimpleNamespace(counters=SimpleNamespace(contains_updates=False))


class FakeSession:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def run(self, query, params):
        del query, params
        return FakeResult(self.rows)


class FakeDriver:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def session(self, default_access_mode=None):
        del default_access_mode
        return FakeSession(self.rows)


def _viz(compiled):
    return [(c.var, c.kind, c.alias) for c in compiled.viz_ids]


def test_fulltext_template_injects_element_ids_for_projected_vars():
    compiled = compile_query(
        "CALL db.index.fulltext.queryNodes($__ft_nodes, $q) YIELD node, score "
        "MATCH (node)-[r]-(m) "
        "RETURN node.name AS n, type(r) AS rel, m.name AS m, r.evidence AS evidence, "
        "r.source_file AS source_file, score "
        "ORDER BY score DESC LIMIT 20",
        parameters={"q": "kefir~"},
    )
    vars_kinds = {(var, kind) for var, kind, _alias in _viz(compiled)}
    assert vars_kinds == {("node", "node"), ("r", "rel"), ("m", "node")}
    assert "elementId(node)" in compiled.cypher.replace(" ", "")
    assert "elementId(r)" in compiled.cypher.replace(" ", "")
    assert "elementId(m)" in compiled.cypher.replace(" ", "")
    assert "AS __id_node" in compiled.cypher
    assert "ORDER BY" in compiled.cypher
    assert compiled.fetch_limit == 21


def test_fulltext_rel_template_injects_relationship_id():
    compiled = compile_query(
        "CALL db.index.fulltext.queryRelationships($__ft_rels, $q) YIELD relationship, score "
        "MATCH (a)-[relationship]-(b) "
        "RETURN a.name AS n, type(relationship) AS rel, b.name AS m, "
        "relationship.evidence AS evidence, relationship.source_file AS source_file, score "
        "ORDER BY score DESC LIMIT 20",
        parameters={"q": "kefir~"},
    )
    vars_kinds = {(var, kind) for var, kind, _alias in _viz(compiled)}
    assert ("relationship", "rel") in vars_kinds
    assert ("a", "node") in vars_kinds
    assert ("b", "node") in vars_kinds
    assert "elementId(relationship)" in compiled.cypher.replace(" ", "")


def test_neighbors_injects_only_projected_graph_vars():
    compiled = compile_query(
        "MATCH (a)-[r]-(b) "
        "RETURN a.name AS a, type(r) AS rel, b.name AS b, r.evidence AS evidence, "
        "r.source_file AS source_file LIMIT 20"
    )
    assert {(var, kind) for var, kind, _ in _viz(compiled)} == {
        ("a", "node"),
        ("r", "rel"),
        ("b", "node"),
    }


def test_return_one_endpoint_does_not_inject_other_match_vars():
    compiled = compile_query("MATCH (a)-[r]-(b) RETURN a.name LIMIT 5")
    assert _viz(compiled) == [("a", "node", "__id_a")]
    assert "elementId(r)" not in compiled.cypher.replace(" ", "")
    assert "elementId(b)" not in compiled.cypher.replace(" ", "")


def test_user_elementid_still_rejected():
    with pytest.raises(Exception) as caught:
        compile_query("MATCH (a)-[r]-(b) RETURN elementId(a)")
    assert caught.value.code == "unsupported"


def test_distinct_and_aggregates_do_not_inject():
    distinct = compile_query("MATCH (n)-[r]-() RETURN DISTINCT n.name LIMIT 10")
    assert distinct.viz_ids == []
    assert "elementId" not in distinct.cypher
    grouped = compile_query(
        "MATCH ()-[r]->() RETURN r.source_file AS source_file, count(r) AS edge_count "
        "ORDER BY edge_count DESC LIMIT 5"
    )
    assert grouped.viz_ids == []
    assert "elementId" not in grouped.cypher
    pure = compile_query("MATCH ()-[r]->() RETURN count(r) AS edge_count")
    assert pure.is_pure_aggregate
    assert pure.viz_ids == []
    collect = compile_query("MATCH (n)-[r]-() RETURN collect(n) AS nodes")
    assert collect.is_pure_aggregate
    assert collect.viz_ids == []


def test_with_scalar_alias_drops_graph_scope():
    compiled = compile_query(
        "MATCH (n)-[r]-() WITH n.name AS n RETURN n LIMIT 10"
    )
    assert compiled.viz_ids == []
    assert "elementId" not in compiled.cypher


def test_union_same_columns_injects_both_branches():
    compiled = compile_query(
        "MATCH (a)-[r]->(b) WHERE toLower(a.name) CONTAINS 'kefir' RETURN a.name "
        "UNION "
        "MATCH (a)-[r]->(b) WHERE toLower(b.name) CONTAINS 'kefir' RETURN a.name"
    )
    assert compiled.viz_ids
    assert compiled.viz_ids[0].var == "a"
    compact = compiled.cypher.replace(" ", "")
    assert compact.count("elementId(a)") == 2
    assert compact.count("AS__id_a") == 2


def test_union_mismatched_projections_skips_inject():
    compiled = compile_query(
        "MATCH (a)-[r]->(b) RETURN a.name "
        "UNION "
        "MATCH (a)-[r]->(b) RETURN b.name"
    )
    assert compiled.viz_ids == []
    assert "elementId" not in compiled.cypher


def test_return_graph_objects_still_injects_ids():
    compiled = compile_query("MATCH (n)-[r]-(m) RETURN n, r, m LIMIT 5")
    assert {(var, kind) for var, kind, _ in _viz(compiled)} == {
        ("n", "node"),
        ("r", "rel"),
        ("m", "node"),
    }


def test_split_public_row_drops_hidden_columns():
    public = split_public_row(
        {"n": "A", "__id_n": "4:1", "embedding": [1.0], "ev_embedding": [0.1], "rel": "X"}
    )
    assert public == {"n": "A", "rel": "X"}


def test_extract_hidden_ids_does_not_invent_neighbors():
    cols = [
        VizIdColumn(alias="__id_a", kind="node", var="a"),
        VizIdColumn(alias="__id_r", kind="rel", var="r"),
    ]
    row = extract_viz_row(
        {"a": "A", "rel": "X", "__id_a": "n-a", "__id_r": "e-1", "__id_b": "n-b-not-in-viz-ids"},
        cols,
    )
    assert {n["element_id"] for n in row.nodes} == {"n-a"}
    assert {e["element_id"] for e in row.edges} == {"e-1"}


def test_extract_sidecar_nodes_rels_paths_and_collect():
    a = FakeNode("n1", "A", ("Microbe",))
    b = FakeNode("n2", "B", ("Metabolite",))
    rel = FakeRel("e1", "PRODUCES", a, b, "from paper")
    path = FakePath([a, b], [rel])
    from_rel = extract_viz_row({"x": rel}, [])
    assert {n["element_id"] for n in from_rel.nodes} == {"n1", "n2"}
    assert from_rel.edges[0]["element_id"] == "e1"
    assert from_rel.edges[0]["type"] == "PRODUCES"
    from_path = extract_viz_row({"p": path}, [])
    assert {e["element_id"] for e in from_path.edges} == {"e1"}
    collected = extract_viz_row({"nodes": [a, FakeNode("n3", "C")]}, [])
    assert {n["element_id"] for n in collected.nodes} == {"n1", "n3"}


def test_materialize_one_chain_dedupes_and_skips_empty():
    assert materialize_query_chain([]) is None
    assert materialize_query_chain([VizRow(), VizRow()]) is None
    chain = materialize_query_chain(
        [
            VizRow(
                nodes=[{"element_id": "n1", "id": "n1", "name": "A"}],
                edges=[{"element_id": "e1", "type": "X"}],
            ),
            VizRow(
                nodes=[{"element_id": "n1", "id": "n1", "name": "A"}],
                edges=[{"element_id": "e1", "type": "X"}],
            ),
            VizRow(nodes=[{"element_id": "n2", "id": "n2", "name": "B"}], edges=[]),
        ]
    )
    assert chain is not None
    assert chain["kind"] == QUERY_GRAPH_KIND
    assert len(chain["edges"]) == 1
    assert {n["element_id"] for n in chain["nodes"]} == {"n1", "n2"}


def test_chain_signature_keeps_unit_hash_and_separates_query_graph():
    unit = {"edge_keys": ["ek-1"], "edges": [{"element_id": "5:1"}]}
    same_unit = {"spine_evidence_seq": ["ek-1"]}
    # edge_keys wins over edges for the first chain
    assert AppStore._chain_signature(unit) == AppStore._chain_signature(
        {"edge_keys": ["ek-1"]}
    )
    qg = {
        "kind": QUERY_GRAPH_KIND,
        "edge_keys": ["ek-1"],
        "edges": [{"element_id": "5:1"}],
    }
    assert AppStore._chain_signature(qg) != AppStore._chain_signature(unit)
    node_only_a = {
        "kind": QUERY_GRAPH_KIND,
        "nodes": [{"element_id": "n1"}],
        "edges": [],
    }
    node_only_b = {
        "kind": QUERY_GRAPH_KIND,
        "nodes": [{"element_id": "n2"}],
        "edges": [],
    }
    assert AppStore._chain_signature(node_only_a) != AppStore._chain_signature(node_only_b)
    assert AppStore._chain_signature(same_unit) == AppStore._chain_signature(
        {"spine_evidence_seq": ["ek-1"]}
    )


def test_build_chain_views_renders_isolated_query_graph_nodes():
    views = build_chain_views(
        [
            {
                "kind": QUERY_GRAPH_KIND,
                "edges": [],
                "fans": {},
                "nodes": [
                    {"id": "4:1", "element_id": "4:1", "name": "Lactobacillus", "labels": ["Microbe"]}
                ],
            }
        ]
    )
    assert len(views) == 1
    assert views[0]["edges"] == []
    assert views[0]["nodes"][0]["caption"] == "Lactobacillus"
    assert views[0]["nodes"][0]["id"] == "4:1"


def test_with_star_keeps_node_for_inject():
    compiled = compile_query("MATCH (n)-[r]-() WITH n RETURN n.name LIMIT 5")
    assert any(col.var == "n" and col.kind == "node" for col in compiled.viz_ids)


def test_path_list_comprehension_does_not_inject_path_id():
    compiled = compile_query(
        "MATCH p = (a)-[*1..3]-(b) "
        "WHERE toLower(a.name) CONTAINS 'x' "
        "RETURN [n IN nodes(p) | n.name] LIMIT 5"
    )
    assert compiled.viz_ids == []
    assert "elementId(p)" not in compiled.cypher.replace(" ", "")


def test_service_guide_says_query_graph_topology_enters_answer_data():
    from pathlib import Path

    guide = (Path("prompts") / "service_guide.md").read_text(encoding="utf-8")
    section = guide.split("\n### Данные\n", 1)[1].split("\n### ", 1)[0]
    assert "точных запросов к графу" in section
    assert "вершины или рёбра" in section
    assert "числа и схема" in section


@pytest.mark.asyncio
async def test_execute_compiled_strips_ids_from_markdown_rows():
    compiled = compile_query(
        "MATCH (a)-[r]-(b) RETURN a.name AS n, type(r) AS rel, b.name AS m LIMIT 5"
    )
    raw = {
        "n": "A",
        "rel": "PRODUCES",
        "m": "B",
    }
    for col in compiled.viz_ids:
        raw[col.alias] = f"id-{col.var}"
    executed = await execute_compiled(
        FakeDriver([raw]), compiled, user_params={}, run_id="run-1"
    )
    assert list(executed.rows[0]) == ["n", "rel", "m"]
    assert executed.rows[0] == {"n": "A", "rel": "PRODUCES", "m": "B"}
    text = format_records(
        executed.rows,
        registry=SourceRegistry(),
        truncated_rows=False,
        output_limit=5,
        is_pure_aggregate=False,
    )
    assert "__id_" not in text
    assert "elementId" not in text
    node_ids = {n["element_id"] for n in executed.viz_rows[0].nodes}
    edge_ids = {e["element_id"] for e in executed.viz_rows[0].edges}
    assert "id-a" in node_ids
    assert "id-b" in node_ids
    assert "id-r" in edge_ids


@pytest.mark.asyncio
async def test_execute_compiled_truncates_viz_with_rows():
    compiled = compile_query("MATCH (a)-[r]-(b) RETURN a.name LIMIT 2")
    alias = compiled.viz_ids[0].alias
    rows = [{"a.name": f"n{i}", alias: f"id-{i}"} for i in range(4)]
    executed = await execute_compiled(
        FakeDriver(rows), compiled, user_params={}, run_id="run-1"
    )
    assert executed.truncated is True
    assert len(executed.rows) == 2
    assert len(executed.viz_rows) == 2


@pytest.mark.asyncio
async def test_query_graph_records_one_chain_in_collector(monkeypatch):
    compiled_holder: dict = {}

    async def fake_exec(driver, compiled, *, user_params, run_id):
        del driver, user_params, run_id
        compiled_holder["c"] = compiled
        from server.algorithm.cypher.query_execute import ExecutedQuery

        return ExecutedQuery(
            rows=[
                {
                    "n": "A",
                    "rel": "X",
                    "m": "B",
                    "evidence": "quote",
                    "source_file": "paper.pdf",
                }
            ],
            truncated=False,
            viz_rows=[
                VizRow(
                    nodes=[
                        {"element_id": "n1", "id": "n1", "name": "A"},
                        {"element_id": "n2", "id": "n2", "name": "B"},
                    ],
                    edges=[{"element_id": "e1", "type": "X"}],
                )
            ],
        )

    monkeypatch.setattr("server.tools.query_graph.execute_compiled", fake_exec)
    monkeypatch.setattr("server.tools.query_graph.get_driver", lambda: object())
    tool = QueryGraphTool(SourceRegistry())
    token = new_graph_collector()
    try:
        with bind_turn("medium", max_searches=4, context={"run_id": "run-1"}):
            text = await tool(
                cypher=(
                    "MATCH (a)-[r]-(b) RETURN a.name AS n, type(r) AS rel, b.name AS m, "
                    "r.evidence AS evidence, r.source_file AS source_file"
                )
            )
        assert "(source:1)" in text or "quote" in text
        chains = current_graph_collector() or []
        assert len(chains) == 1
        assert chains[0]["kind"] == QUERY_GRAPH_KIND
        assert len(chains[0]["edges"]) == 1
        assert compiled_holder["c"].viz_ids
    finally:
        reset_graph_collector(token)


@pytest.mark.asyncio
async def test_query_graph_count_and_empty_do_not_record(monkeypatch):
    from server.algorithm.cypher.query_execute import ExecutedQuery

    async def fake_count(driver, compiled, *, user_params, run_id):
        del driver, compiled, user_params, run_id
        return ExecutedQuery(rows=[{"edge_count": 12}], truncated=False, viz_rows=[VizRow()])

    monkeypatch.setattr("server.tools.query_graph.execute_compiled", fake_count)
    monkeypatch.setattr("server.tools.query_graph.get_driver", lambda: object())
    tool = QueryGraphTool(SourceRegistry())
    token = new_graph_collector()
    try:
        with bind_turn("medium", max_searches=4, context={"run_id": "run-1"}):
            text = await tool(cypher="MATCH ()-[r]->() RETURN count(r) AS edge_count")
        assert "12" in text
        assert current_graph_collector() == []
    finally:
        reset_graph_collector(token)

    async def fake_empty(driver, compiled, *, user_params, run_id):
        del driver, compiled, user_params, run_id
        return ExecutedQuery(
            rows=[],
            truncated=False,
            viz_rows=[VizRow(nodes=[{"element_id": "should-not-record"}])],
        )

    monkeypatch.setattr("server.tools.query_graph.execute_compiled", fake_empty)
    token = new_graph_collector()
    try:
        with bind_turn("medium", max_searches=4, context={"run_id": "run-1"}):
            text = await tool(cypher="MATCH (a)-[r]-(b) RETURN a.name")
        assert text.startswith("NO_MATCHES")
        assert current_graph_collector() == []
    finally:
        reset_graph_collector(token)


@pytest.mark.asyncio
async def test_query_graph_and_ask_subgraph_preserve_call_order(monkeypatch):
    from server.algorithm.cypher.query_execute import ExecutedQuery

    async def fake_exec(driver, compiled, *, user_params, run_id):
        del driver, compiled, user_params, run_id
        return ExecutedQuery(
            rows=[{"n": "A"}],
            truncated=False,
            viz_rows=[VizRow(nodes=[{"element_id": "n1", "id": "n1", "name": "A"}])],
        )

    monkeypatch.setattr("server.tools.query_graph.execute_compiled", fake_exec)
    monkeypatch.setattr("server.tools.query_graph.get_driver", lambda: object())
    tool = QueryGraphTool(SourceRegistry())
    token = new_graph_collector()
    try:
        record_accepted_chains([{"chain_id": "a1", "edges": [{"element_id": "unit-e"}]}])
        with bind_turn("medium", max_searches=4, context={"run_id": "run-1"}):
            await tool(cypher="MATCH (n)-[r]-() RETURN n.name AS n")
            await tool(cypher="MATCH (n)-[r]-() RETURN n.name AS n")
        ids = [c["chain_id"] for c in (current_graph_collector() or [])]
        kinds = [c.get("kind") for c in (current_graph_collector() or [])]
        assert ids == ["a1", "a2", "a3"]
        assert kinds == [None, QUERY_GRAPH_KIND, QUERY_GRAPH_KIND]
    finally:
        reset_graph_collector(token)

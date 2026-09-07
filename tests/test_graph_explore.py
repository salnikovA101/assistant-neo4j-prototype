from pydantic import ValidationError
import pytest

from server.algorithm.cypher.explore import (
    EXPLORE_FIELDS,
    EXPLORE_LIMITS,
    MAX_EXPLORE_LIMIT,
    MIN_EXPLORE_LIMIT,
    _EXPAND_TRIPLETS,
    _FACET_NODE_LABELS,
    _FACET_RELATIONSHIPS,
    _FACET_SOURCES,
    _FETCH_TRIPLETS,
    clamp_explore_field,
    clamp_explore_limit,
    graph_filters_active,
    normalize_graph_filters,
    nodes_from_triplet_rows,
)
from server.core.http_api import GraphExpandBody, GraphExploreBody, GraphFacetsBody, GraphFilters
from server.tools.graph_explore import rows_to_explore_payload


def test_explore_limits_and_fields() -> None:
    assert EXPLORE_LIMITS == (10, 100, 1000)
    assert MIN_EXPLORE_LIMIT == 1
    assert MAX_EXPLORE_LIMIT == 5000
    assert EXPLORE_FIELDS == ("all", "name", "label", "rel", "evidence", "source")
    assert clamp_explore_limit(25) == 25
    assert clamp_explore_limit(1000) == 1000
    assert clamp_explore_limit(0) == 1
    assert clamp_explore_limit(6000) == 5000
    assert clamp_explore_field("EVIDENCE") == "evidence"
    assert clamp_explore_field("nope") == "all"
    GraphExploreBody(q="lactobacillus", limit=10)
    GraphExploreBody(limit=100, field="rel")
    GraphExploreBody(limit=25)
    with pytest.raises(ValidationError):
        GraphExploreBody(field="vertex")  # type: ignore[arg-type]


def test_explore_cypher_matches_triplets_not_bare_nodes() -> None:
    blob = _FETCH_TRIPLETS
    assert "embedding" not in blob
    assert "$q" in blob
    assert "$limit" in blob
    assert "$field" in blob
    assert "$cursor" in blob
    assert "$q <> ''" in blob
    assert "relevance" in blob
    assert "ORDER BY relevance DESC" in blob
    assert "run_id" in blob
    assert "MATCH (a)-[r]->(b)" in blob
    assert "type(r)" in blob
    assert "r.evidence" in blob
    assert "MATCH (n)" not in blob


def test_graph_filters_and_facets_contract() -> None:
    filters = GraphFilters(
        node_labels=["Microbe"],
        relationship_types=["PRODUCES"],
        sources=["paper.pdf"],
        min_confidence=0.7,
    )
    assert graph_filters_active(filters.model_dump())
    normalized = normalize_graph_filters(
        {"node_labels": ["Microbe", "Microbe", ""], "min_confidence": 2}
    )
    assert normalized["node_labels"] == ["Microbe"]
    assert normalized["min_confidence"] == 1.0
    assert not graph_filters_active(None)
    GraphFacetsBody(q="kefir", filters=filters, source_limit=50)
    with pytest.raises(ValidationError):
        GraphFilters(min_confidence=1.1)

    for query in (_FETCH_TRIPLETS, _EXPAND_TRIPLETS):
        assert "$node_labels" in query
        assert "OR any(l IN labels(a) WHERE l IN $node_labels)" in query
        assert "OR any(l IN labels(b) WHERE l IN $node_labels)" in query
        assert "$relationship_types" in query
        assert "$sources" in query
        assert "trim(coalesce(r.evidence, '')) <> ''" in query
        assert "$min_confidence" in query
        assert "r.run_id = $run_id" in query
    assert "$exclude_edge_ids" in _EXPAND_TRIPLETS
    assert "$direction" in _EXPAND_TRIPLETS
    assert "ORDER BY evidence_rank DESC" in _EXPAND_TRIPLETS
    assert "count(DISTINCT n)" in _FACET_NODE_LABELS
    assert "UNWIND labels(n) AS value" in _FACET_NODE_LABELS
    assert "count(DISTINCT r)" in _FACET_RELATIONSHIPS
    assert "ORDER BY count DESC, value" in _FACET_SOURCES


def test_graph_expand_body_accepts_incremental_request() -> None:
    body = GraphExpandBody(
        node_id="4:1",
        limit=25,
        exclude_edge_ids=["5:1", "5:2"],
        direction="outgoing",
        filters={"node_labels": ["Metabolite"]},
    )
    assert body.limit == 25
    assert body.exclude_edge_ids == ["5:1", "5:2"]
    assert body.direction == "outgoing"
    assert body.filters.node_labels == ["Metabolite"]


def test_nodes_come_from_triplet_endpoints() -> None:
    nodes = nodes_from_triplet_rows(
        [
            {
                "from_id": "4:1",
                "to_id": "4:2",
                "from_name": "L. plantarum",
                "to_name": "lactic acid",
                "from_labels": ["Microbe", "BiologicalObject", "Microbe"],
                "to_labels": [],
            }
        ]
    )
    assert {n["id"] for n in nodes} == {"4:1", "4:2"}
    assert {n["name"] for n in nodes} == {"L. plantarum", "lactic acid"}
    assert next(n for n in nodes if n["id"] == "4:1")["labels"] == ["BiologicalObject", "Microbe"]
    assert next(n for n in nodes if n["id"] == "4:2")["labels"] == []


def test_rows_to_explore_payload_shape_and_no_embedding() -> None:
    payload = rows_to_explore_payload(
        [
            {"id": "4:1", "name": "L. plantarum", "labels": ["NewClass", "Entity"]},
            {"id": "4:2", "name": "lactic acid", "labels": [], "embedding": [1, 2]},
        ],
        [
            {
                "id": "5:1",
                "type": "PRODUCES",
                "from_id": "4:1",
                "to_id": "4:2",
                "from_name": "L. plantarum",
                "to_name": "lactic acid",
                "from_labels": ["NewClass", "Entity"],
                "to_labels": [],
                "evidence": "produces lactic acid",
                "chunk_id": "c1",
                "source_file": "paper.pdf",
                "confidence": 0.9,
                "run_id": "full_corpus_20260713",
                "embedding": [0.1],
                "evidence_embedding": [0.2],
            }
        ],
    )

    assert payload["views"] == []
    nodes = payload["all"]["nodes"]
    edges = payload["all"]["edges"]
    assert {n["id"] for n in nodes} == {"4:1", "4:2"}
    assert nodes[0]["caption"] == "L. plantarum"
    assert nodes[0]["group"] == "Вершина"
    assert nodes[0]["labels"] == ["Entity", "NewClass"]
    assert nodes[1]["labels"] == []
    assert "embedding" not in str(nodes)
    assert len(edges) == 1
    props = edges[0]["properties"]
    assert props["evidence"] == "produces lactic acid"
    assert "embedding" not in props
    assert "evidence_embedding" not in props
    dumped = str(payload)
    assert "[1, 2]" not in dumped
    assert "[0.1]" not in dumped


def test_graph_explore_http_accepts_custom_limit_and_guards_range() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.post("/graph_explore")
    def explore(body: GraphExploreBody):
        return {"q": body.q, "limit": body.limit, "field": body.field}

    client = TestClient(app)
    custom = client.post("/graph_explore", json={"q": "x", "limit": 25})
    assert custom.status_code == 200
    assert custom.json()["limit"] == 25
    bad = client.post("/graph_explore", json={"q": "x", "limit": 0})
    assert bad.status_code == 422
    too_large = client.post("/graph_explore", json={"q": "x", "limit": 5001})
    assert too_large.status_code == 422
    empty = client.post("/graph_explore", json={"q": "", "limit": 10})
    assert empty.status_code == 200
    assert empty.json() == {"q": "", "limit": 10, "field": "all"}
    scoped = client.post(
        "/graph_explore", json={"q": "produces", "limit": 100, "field": "rel"}
    )
    assert scoped.status_code == 200
    assert scoped.json()["field"] == "rel"

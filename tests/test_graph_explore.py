from pydantic import ValidationError
import pytest

from server.algorithm.cypher.explore import (
    EXPLORE_FIELDS,
    EXPLORE_LIMITS,
    _FETCH_TRIPLETS,
    clamp_explore_field,
    clamp_explore_limit,
    nodes_from_triplet_rows,
)
from server.core.http_api import GraphExploreBody
from server.tools.graph_explore import rows_to_explore_payload


def test_explore_limits_and_fields() -> None:
    assert EXPLORE_LIMITS == (10, 100, 1000)
    assert EXPLORE_FIELDS == ("all", "name", "label", "rel", "evidence", "source")
    assert clamp_explore_limit(25) == 100
    assert clamp_explore_limit(1000) == 1000
    assert clamp_explore_field("EVIDENCE") == "evidence"
    assert clamp_explore_field("nope") == "all"
    GraphExploreBody(q="lactobacillus", limit=10)
    GraphExploreBody(limit=100, field="rel")
    with pytest.raises(ValidationError):
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


def test_nodes_come_from_triplet_endpoints() -> None:
    nodes = nodes_from_triplet_rows(
        [
            {
                "from_id": "4:1",
                "to_id": "4:2",
                "from_name": "L. plantarum",
                "to_name": "lactic acid",
                "from_label": "Microbe",
                "to_label": "Metabolite",
            }
        ]
    )
    assert {n["id"] for n in nodes} == {"4:1", "4:2"}
    assert {n["name"] for n in nodes} == {"L. plantarum", "lactic acid"}


def test_rows_to_explore_payload_shape_and_no_embedding() -> None:
    payload = rows_to_explore_payload(
        [
            {"id": "4:1", "name": "L. plantarum", "label": "Microbe"},
            {"id": "4:2", "name": "lactic acid", "label": "Metabolite", "embedding": [1, 2]},
        ],
        [
            {
                "id": "5:1",
                "type": "PRODUCES",
                "from_id": "4:1",
                "to_id": "4:2",
                "from_name": "L. plantarum",
                "to_name": "lactic acid",
                "from_label": "Microbe",
                "to_label": "Metabolite",
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
    assert nodes[0]["group"] == "Microbe"
    assert "embedding" not in str(nodes)
    assert len(edges) == 1
    props = edges[0]["properties"]
    assert props["evidence"] == "produces lactic acid"
    assert "embedding" not in props
    assert "evidence_embedding" not in props
    dumped = str(payload)
    assert "[1, 2]" not in dumped
    assert "[0.1]" not in dumped


def test_graph_explore_http_rejects_non_chip_limit() -> None:
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    app = FastAPI()

    @app.post("/graph_explore")
    def explore(body: GraphExploreBody):
        return {"q": body.q, "limit": body.limit, "field": body.field}

    client = TestClient(app)
    bad = client.post("/graph_explore", json={"q": "x", "limit": 25})
    assert bad.status_code == 422
    empty = client.post("/graph_explore", json={"q": "", "limit": 10})
    assert empty.status_code == 200
    assert empty.json() == {"q": "", "limit": 10, "field": "all"}
    scoped = client.post(
        "/graph_explore", json={"q": "produces", "limit": 100, "field": "rel"}
    )
    assert scoped.status_code == 200
    assert scoped.json()["field"] == "rel"

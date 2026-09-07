"""run_id wiring: config, S2/S3 Cypher, S3 fingerprint."""

from __future__ import annotations

import pytest

from server.algorithm.cypher.edges import (
    EVIDENCE_BY_CHUNKS,
    FETCH_VIZ_BY_EDGE_IDS,
    induced_bridges_query,
    relationship_ann_query,
    sanitize_vector_index_name,
)
from server.algorithm.graph_cache import build_fingerprint
from server.algorithm.embed_client import fetch_vector_indexes
from server.algorithm.models import SubQuestion
from server.algorithm.params import Params
from server.algorithm.pipeline import run
from server.utils.config import AppConfig, load_config, retrieval_param_overrides


def test_app_config_has_run_id_field():
    assert "run_id" in AppConfig.model_fields
    cfg = load_config()
    assert isinstance(cfg.run_id, str)


def test_retrieval_overrides_from_yaml():
    cfg = load_config()
    o = retrieval_param_overrides(cfg)
    assert o["run_id"] == (cfg.run_id or "").strip()
    assert o["rerank_enabled"] is False
    assert cfg.rerank_enabled is False


def test_sanitize_vector_index_name():
    assert sanitize_vector_index_name("produces_evidence_index") == (
        "`produces_evidence_index`"
    )
    assert sanitize_vector_index_name("index ` one") == "`index `` one`"
    assert sanitize_vector_index_name(" index ") == "` index `"
    with pytest.raises(ValueError):
        sanitize_vector_index_name("idx\nDROP INDEX")
    with pytest.raises(ValueError):
        sanitize_vector_index_name("")


def test_relationship_ann_query_refuses_unfiltered():
    with pytest.raises(ValueError, match="run_id"):
        relationship_ann_query("produces_evidence_index", run_id="")


def test_relationship_ann_query_filtered():
    q = relationship_ann_query("produces_evidence_index", run_id="full_corpus_20260713")
    assert "CYPHER 25" in q
    assert "SEARCH r IN" in q
    assert "VECTOR INDEX `produces_evidence_index`" in q
    assert "WHERE r.run_id = $run_id" in q
    assert "queryRelationships" not in q
    search = q[q.index("SEARCH r IN") : q.index(") SCORE AS score")]
    assert search.index("WHERE r.run_id = $run_id") < search.index("LIMIT $k")


def test_induced_bridges_query_run_id():
    with pytest.raises(ValueError, match="run_id"):
        induced_bridges_query(run_id="")
    filtered = induced_bridges_query(run_id="full_corpus_20260713")
    assert "AND r.run_id = $run_id" in filtered
    assert "vector.similarity.cosine" in filtered
    assert filtered.index("r.run_id = $run_id") < filtered.index("LIMIT $limit")


def test_hydration_queries_are_run_scoped():
    assert "AND r.run_id = $run_id" in FETCH_VIZ_BY_EDGE_IDS
    assert "AND r.run_id = $run_id" in EVIDENCE_BY_CHUNKS


def test_s3_fingerprint_includes_run_id():
    sqs = [SubQuestion(id="sq1", text="claim about peptides")]
    base = Params(L=100, run_id="")
    other = Params(L=100, run_id="full_corpus_20260713")
    assert build_fingerprint(base, sqs) != build_fingerprint(other, sqs)
    assert build_fingerprint(base, sqs)["params"]["run_id"] == ""
    assert (
        build_fingerprint(other, sqs)["params"]["run_id"] == "full_corpus_20260713"
    )


@pytest.mark.asyncio
async def test_pipeline_refuses_empty_run_id_before_database_work():
    result = await run(
        None,  # type: ignore[arg-type]
        subquestions=[{"id": "sq1", "text": "A concrete statement."}],
        params=Params(run_id=""),
    )
    assert result["error"] == "run_id_required"
    assert result["accepted"] == []


@pytest.mark.asyncio
async def test_pipeline_refuses_cached_edges_from_another_run():
    result = await run(
        None,  # type: ignore[arg-type]
        subquestions=[{"id": "sq1", "text": "A concrete statement."}],
        params=Params(run_id="run-current"),
        s3_bundle={
            "graphs": {
                "sq1": {
                    "source_graph": "sq1",
                    "edges": [
                        {
                            "edge_key": "edge-1",
                            "element_id": "rel-1",
                            "rel_type": "NEW_REL",
                            "start_id": "a",
                            "end_id": "b",
                            "start_name": "A",
                            "end_name": "B",
                            "evidence": "evidence",
                            "run_id": "run-other",
                        }
                    ],
                }
            }
        },
    )
    assert result["error"] == "cache_run_id_mismatch"


@pytest.mark.asyncio
async def test_vector_index_discovery_is_dynamic_deduplicated_and_dimension_safe():
    records = [
        {
            "name": "legacy_rel_a",
            "entityType": "RELATIONSHIP",
            "state": "ONLINE",
            "labelsOrTypes": ["TOTALLY_NEW_RELATION"],
            "properties": ["evidence_embedding", "run_id"],
            "options": {"indexConfig": {"vector.dimensions": 768}},
        },
        {
            "name": "rel_ev_v1_new_deadbeef00",
            "entityType": "RELATIONSHIP",
            "state": "ONLINE",
            "labelsOrTypes": ["TOTALLY_NEW_RELATION"],
            "properties": ["evidence_embedding", "run_id"],
            "options": {"indexConfig": {"vector.dimensions": 768}},
        },
        {
            "name": "rel_ev_v1_other_feedface00",
            "entityType": "RELATIONSHIP",
            "state": "ONLINE",
            "labelsOrTypes": ["REL WITH SPACES"],
            "properties": ["evidence_embedding", "run_id"],
            "options": {"indexConfig": {"vector.dimensions": 768}},
        },
        {
            "name": "wrong_dimension",
            "entityType": "RELATIONSHIP",
            "state": "ONLINE",
            "labelsOrTypes": ["WRONG_DIM"],
            "properties": ["evidence_embedding", "run_id"],
            "options": {"indexConfig": {"vector.dimensions": 1024}},
        },
        {
            "name": "offline",
            "entityType": "RELATIONSHIP",
            "state": "POPULATING",
            "labelsOrTypes": ["OFFLINE_REL"],
            "properties": ["evidence_embedding", "run_id"],
            "options": {"indexConfig": {"vector.dimensions": 768}},
        },
        {
            "name": "unfiltered_legacy",
            "entityType": "RELATIONSHIP",
            "state": "ONLINE",
            "labelsOrTypes": ["UNFILTERED_REL"],
            "properties": ["evidence_embedding"],
            "options": {"indexConfig": {"vector.dimensions": 768}},
        },
    ]

    class Result:
        def __aiter__(self):
            async def iterate():
                for record in records:
                    yield record
            return iterate()

    class Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def run(self, query):
            assert query == "SHOW VECTOR INDEXES"
            return Result()

    class Driver:
        def session(self):
            return Session()

    found = await fetch_vector_indexes(Driver(), expected_dimension=768)
    assert found["relationships"] == [
        "rel_ev_v1_other_feedface00",
        "rel_ev_v1_new_deadbeef00",
    ]

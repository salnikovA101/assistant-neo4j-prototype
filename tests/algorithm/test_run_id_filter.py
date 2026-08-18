"""run_id wiring: config, S2/S3 Cypher, S3 fingerprint."""

from __future__ import annotations

import pytest

from server.algorithm.cypher.edges import (
    induced_bridges_query,
    relationship_ann_query,
    sanitize_vector_index_name,
)
from server.algorithm.graph_cache import build_fingerprint
from server.algorithm.models import SubQuestion
from server.algorithm.params import Params
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
        "produces_evidence_index"
    )
    with pytest.raises(ValueError):
        sanitize_vector_index_name("idx; DROP INDEX")
    with pytest.raises(ValueError):
        sanitize_vector_index_name("")


def test_relationship_ann_query_unfiltered():
    q = relationship_ann_query("produces_evidence_index", run_id="")
    assert "queryRelationships" in q
    assert "SEARCH" not in q
    assert "$index" in q
    assert "WHERE r.run_id" not in q


def test_relationship_ann_query_filtered():
    q = relationship_ann_query("produces_evidence_index", run_id="full_corpus_20260713")
    assert "CYPHER 25" in q
    assert "SEARCH r IN" in q
    assert "VECTOR INDEX produces_evidence_index" in q
    assert "WHERE r.run_id = $run_id" in q
    assert "queryRelationships" not in q


def test_induced_bridges_query_run_id():
    plain = induced_bridges_query(run_id="")
    assert "r.run_id = $run_id" not in plain
    filtered = induced_bridges_query(run_id="full_corpus_20260713")
    assert "AND r.run_id = $run_id" in filtered
    assert "vector.similarity.cosine" in filtered


def test_s3_fingerprint_includes_run_id():
    sqs = [SubQuestion(id="sq1", text="claim about peptides")]
    base = Params(L=100, run_id="")
    other = Params(L=100, run_id="full_corpus_20260713")
    assert build_fingerprint(base, sqs) != build_fingerprint(other, sqs)
    assert build_fingerprint(base, sqs)["params"]["run_id"] == ""
    assert (
        build_fingerprint(other, sqs)["params"]["run_id"] == "full_corpus_20260713"
    )

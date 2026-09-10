"""Unit tests for scripts/vectorize_edges.py helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vectorize_edges.py"


def _mod():
    spec = importlib.util.spec_from_file_location("vectorize_edges", _SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def ve():
    return _mod()


def test_redact_uri_strips_userinfo(ve):
    assert ve._redact_uri("bolt://neo4j:secret@db.example:7687") == (
        "bolt://***@db.example:7687"
    )
    assert ve._redact_uri("bolt://localhost:7687") == "bolt://localhost:7687"


def test_create_cypher_quotes_arbitrary_identifiers(ve):
    q = ve._create_cypher("idx; DROP INDEX foo", "type` with spaces", 2048)
    assert "CREATE VECTOR INDEX `idx; DROP INDEX foo`" in q
    assert "[r:`type`` with spaces`]" in q
    assert "[r:` padded type `]" in ve._create_cypher("safe", " padded type ", 3)
    with pytest.raises(SystemExit, match="Invalid index name"):
        ve._create_cypher("idx\nDROP", "produces", 2048)
    with pytest.raises(SystemExit, match="Invalid relationship type"):
        ve._create_cypher("safe", "bad\nkind", 2048)


def test_create_cypher_includes_run_id_filter(ve):
    q = ve._create_cypher("produces_evidence_index", "produces", 2048)
    assert "WITH [r.run_id]" in q
    assert "`vector.dimensions`: 2048" in q


def test_index_name_for_rel_type(ve):
    assert ve._index_name_for_rel_type("produces") == "rel_ev_v1_produces_dd92a47df2"
    assert ve._index_name_for_rel_type("new type").startswith("rel_ev_v1_new_type_")
    assert ve._index_name_for_rel_type("новый тип").startswith("rel_ev_v1_type_")
    assert ve._index_name_for_rel_type("new type") != ve._index_name_for_rel_type("new-type")
    with pytest.raises(SystemExit, match="Invalid relationship type"):
        ve._index_name_for_rel_type("produces\n")


def test_pending_batch_filters_run_before_limit(ve):
    q = ve._FETCH_PENDING_BATCH_CYPHER
    assert q.index("r.run_id = $run_id") < q.index("LIMIT $limit")
    assert "ORDER BY eid" in q
    assert "AND r.run_id = $run_id" in ve._SET_CYPHER


def test_ensure_indexes_is_additive_and_one_per_type(ve, monkeypatch):
    class FakeDb:
        def __init__(self):
            self.consumed = []

        def consume(self, cypher, **params):
            self.consumed.append((cypher, params))

    db = FakeDb()
    monkeypatch.setattr(ve, "_required_rel_types", lambda _db, _run_id: ["EXISTING", "NEW TYPE"])
    monkeypatch.setattr(
        ve,
        "_rel_indexes",
        lambda _db: [{
            "name": "legacy_existing",
            "labelsOrTypes": ["EXISTING"],
            "properties": ["evidence_embedding", "run_id"],
            "state": "ONLINE",
            "options": {"indexConfig": {"vector.dimensions": 768}},
        }],
    )
    waited = []
    monkeypatch.setattr(ve, "_wait_online", lambda _db, names: waited.extend(names))

    created = ve.ensure_rel_vector_indexes(
        db, run_id="run-new", dim=768, dry_run=False
    )

    assert created == [ve._index_name_for_rel_type("NEW TYPE")]
    assert len(db.consumed) == 1
    assert "[r:`NEW TYPE`]" in db.consumed[0][0]
    assert "WITH [r.run_id]" in db.consumed[0][0]
    assert waited == created


def test_ensure_indexes_rejects_dimension_mismatch(ve, monkeypatch):
    monkeypatch.setattr(ve, "_required_rel_types", lambda _db, _run_id: ["REL"])
    monkeypatch.setattr(
        ve,
        "_rel_indexes",
        lambda _db: [{
            "name": "old",
            "labelsOrTypes": ["REL"],
            "properties": ["evidence_embedding", "run_id"],
            "state": "ONLINE",
            "options": {"indexConfig": {"vector.dimensions": 1024}},
        }],
    )
    with pytest.raises(SystemExit, match="do not match"):
        ve.ensure_rel_vector_indexes(object(), run_id="run-new", dim=768, dry_run=True)


def test_ensure_indexes_does_not_reuse_unfiltered_legacy_index(ve, monkeypatch):
    class FakeDb:
        def __init__(self):
            self.consumed = []

        def consume(self, cypher, **params):
            self.consumed.append((cypher, params))

    db = FakeDb()
    monkeypatch.setattr(ve, "_required_rel_types", lambda _db, _run_id: ["REL"])
    monkeypatch.setattr(
        ve,
        "_rel_indexes",
        lambda _db: [{
            "name": "legacy_unfiltered",
            "labelsOrTypes": ["REL"],
            "properties": ["evidence_embedding"],
            "state": "ONLINE",
            "options": {"indexConfig": {"vector.dimensions": 768}},
        }],
    )
    monkeypatch.setattr(ve, "_wait_online", lambda *_args, **_kwargs: None)

    created = ve.ensure_rel_vector_indexes(
        db, run_id="run-new", dim=768, dry_run=False
    )

    assert created == [ve._index_name_for_rel_type("REL")]
    assert "WITH [r.run_id]" in db.consumed[0][0]


def test_assert_batch_dim(ve):
    assert ve._assert_batch_dim([[0.1, 0.2], [0.3, 0.4]], None) == 2
    with pytest.raises(SystemExit, match="does not match"):
        ve._assert_batch_dim([[0.1, 0.2, 0.3]], 2)
    with pytest.raises(SystemExit, match="empty"):
        ve._assert_batch_dim([[]], None)


def test_confirm_skips_on_dry_run_and_yes(ve):
    ve._confirm("nope", yes=True, dry_run=False)
    ve._confirm("nope", yes=False, dry_run=True)


def test_confirm_refuses_non_tty(ve, monkeypatch):
    monkeypatch.setattr(ve.sys.stdin, "isatty", lambda: False)
    with pytest.raises(SystemExit, match="--yes"):
        ve._confirm("Overwrite", yes=False, dry_run=False)

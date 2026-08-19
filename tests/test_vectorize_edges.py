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


def test_create_cypher_rejects_injection(ve):
    with pytest.raises(SystemExit, match="Unsafe index name"):
        ve._create_cypher("idx; DROP INDEX foo", "produces", 2048)
    with pytest.raises(SystemExit, match="Unsafe relationship type"):
        ve._create_cypher("produces_evidence_index", "produces]-() MATCH", 2048)


def test_create_cypher_includes_run_id_filter(ve):
    q = ve._create_cypher("produces_evidence_index", "produces", 2048)
    assert "WITH [r.run_id]" in q
    assert "`vector.dimensions`: 2048" in q


def test_index_name_for_rel_type(ve):
    assert ve._index_name_for_rel_type("produces") == "produces_evidence_index"
    with pytest.raises(SystemExit, match="Unsafe relationship type"):
        ve._index_name_for_rel_type("produces`")


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

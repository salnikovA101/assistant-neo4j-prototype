"""Lucene index ensure: create, skip, drift DROP+CREATE."""

from __future__ import annotations

from pathlib import Path

import pytest

from server.algorithm.cypher.fulltext import (
    ensure_query_graph_fulltext,
    node_fulltext_cypher,
    rel_fulltext_cypher,
)
from server.algorithm.cypher.query_compile import FT_NODE_INDEX, FT_REL_INDEX


class FakeDb:
    def __init__(
        self,
        *,
        labels: list[str] | None = None,
        types: list[str] | None = None,
        indexes: list[dict] | None = None,
    ) -> None:
        self.labels = list(labels or [])
        self.types = list(types or [])
        self.indexes = list(indexes or [])
        self.consumed: list[str] = []

    def query(self, cypher: str, **params):
        if "db.labels" in cypher:
            return [{"label": label} for label in self.labels]
        if "relationshipTypes" in cypher:
            return [{"relationshipType": rel_type} for rel_type in self.types]
        if "SHOW INDEXES" in cypher:
            return list(self.indexes)
        raise AssertionError(cypher)

    def consume(self, cypher: str, **params) -> None:
        self.consumed.append(cypher)


def _node_index(labels: list[str]) -> dict:
    return {
        "name": FT_NODE_INDEX,
        "type": "FULLTEXT",
        "entityType": "NODE",
        "labelsOrTypes": labels,
        "properties": ["name"],
        "state": "ONLINE",
    }


def _rel_index(types: list[str]) -> dict:
    return {
        "name": FT_REL_INDEX,
        "type": "FULLTEXT",
        "entityType": "RELATIONSHIP",
        "labelsOrTypes": types,
        "properties": ["evidence"],
        "state": "ONLINE",
    }


def test_create_cypher_quotes_labels_and_types():
    node_q = node_fulltext_cypher(["Person", "weird`label"])
    assert f"CREATE FULLTEXT INDEX `{FT_NODE_INDEX}`" in node_q
    assert "(n:`Person`|`weird``label`)" in node_q
    assert "ON EACH [n.name]" in node_q
    rel_q = rel_fulltext_cypher(["PRODUCES", "has space"])
    assert f"CREATE FULLTEXT INDEX `{FT_REL_INDEX}`" in rel_q
    assert "[r:`PRODUCES`|`has space`]" in rel_q
    assert "ON EACH [r.evidence]" in rel_q


def test_ensure_creates_missing_indexes():
    db = FakeDb(labels=["Person", "Org"], types=["PRODUCES"])
    actions = ensure_query_graph_fulltext(db)
    assert actions == [f"create:{FT_NODE_INDEX}", f"create:{FT_REL_INDEX}"]
    assert len(db.consumed) == 2
    assert "FOR (n:`Org`|`Person`)" in db.consumed[0]
    assert "ON EACH [n.name]" in db.consumed[0]
    assert "[r:`PRODUCES`]" in db.consumed[1]
    assert "ON EACH [r.evidence]" in db.consumed[1]
    assert not any(q.strip().upper().startswith("DROP") for q in db.consumed)


def test_ensure_reuses_in_sync_indexes():
    db = FakeDb(
        labels=["Org", "Person"],
        types=["PRODUCES"],
        indexes=[_node_index(["Person", "Org"]), _rel_index(["PRODUCES"])],
    )
    actions = ensure_query_graph_fulltext(db)
    assert actions == [f"reuse:{FT_NODE_INDEX}", f"reuse:{FT_REL_INDEX}"]
    assert db.consumed == []


def test_ensure_rebuilds_on_label_drift():
    db = FakeDb(
        labels=["Person", "Org"],
        types=["PRODUCES"],
        indexes=[_node_index(["Person"]), _rel_index(["PRODUCES"])],
    )
    actions = ensure_query_graph_fulltext(db)
    assert f"drop:{FT_NODE_INDEX}" in actions
    assert f"create:{FT_NODE_INDEX}" in actions
    assert f"reuse:{FT_REL_INDEX}" in actions
    assert db.consumed[0].startswith("DROP INDEX")
    assert FT_NODE_INDEX in db.consumed[0]
    assert "FOR (n:`Org`|`Person`)" in db.consumed[1]


def test_ensure_rebuilds_on_rel_type_drift():
    db = FakeDb(
        labels=["Person"],
        types=["PRODUCES", "INHIBITS"],
        indexes=[_node_index(["Person"]), _rel_index(["PRODUCES"])],
    )
    actions = ensure_query_graph_fulltext(db)
    assert f"drop:{FT_REL_INDEX}" in actions
    assert f"create:{FT_REL_INDEX}" in actions
    assert f"reuse:{FT_NODE_INDEX}" in actions
    assert any("[r:`INHIBITS`|`PRODUCES`]" in q or "[r:`PRODUCES`|`INHIBITS`]" in q for q in db.consumed)


def test_ensure_rebuilds_on_property_mismatch():
    bad = _node_index(["Person"])
    bad["properties"] = ["name", "extra"]
    db = FakeDb(labels=["Person"], types=[], indexes=[bad])
    actions = ensure_query_graph_fulltext(db)
    assert f"drop:{FT_NODE_INDEX}" in actions
    assert f"create:{FT_NODE_INDEX}" in actions


def test_ensure_recreate_drops_even_when_in_sync():
    db = FakeDb(
        labels=["Person"],
        types=["PRODUCES"],
        indexes=[_node_index(["Person"]), _rel_index(["PRODUCES"])],
    )
    actions = ensure_query_graph_fulltext(db, recreate=True)
    assert actions == [
        f"drop:{FT_NODE_INDEX}",
        f"create:{FT_NODE_INDEX}",
        f"drop:{FT_REL_INDEX}",
        f"create:{FT_REL_INDEX}",
    ]
    assert sum(1 for q in db.consumed if q.strip().upper().startswith("DROP")) == 2
    assert sum(1 for q in db.consumed if "CREATE FULLTEXT" in q.upper()) == 2


def test_ensure_dry_run_does_not_write():
    db = FakeDb(labels=["Person"], types=["PRODUCES"])
    actions = ensure_query_graph_fulltext(db, dry_run=True)
    assert actions == [f"create:{FT_NODE_INDEX}", f"create:{FT_REL_INDEX}"]
    assert db.consumed == []


def test_ensure_skips_empty_catalog():
    db = FakeDb()
    actions = ensure_query_graph_fulltext(db)
    assert actions == []
    assert db.consumed == []


def test_vectorize_and_cli_call_ensure():
    root = Path(__file__).resolve().parents[1]
    vectorize = (root / "scripts" / "vectorize_edges.py").read_text(encoding="utf-8")
    cli = (root / "scripts" / "create_fulltext_indexes.py").read_text(encoding="utf-8")
    assert "ensure_query_graph_fulltext" in vectorize
    assert "recreate=bool(args.recreate_indexes)" in vectorize
    assert "ensure_query_graph_fulltext" in cli
    assert "from server.algorithm.embed" not in cli


def test_invalid_identifier_rejected():
    with pytest.raises(ValueError, match="Invalid identifier"):
        node_fulltext_cypher(["bad\nlabel"])

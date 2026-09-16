"""Ensure Lucene indexes used by query_graph (node names + relationship evidence).

CREATE FULLTEXT … IF NOT EXISTS does not grow the label/type union when new
labels appear. If SHOW INDEXES reports a different union than db.labels() /
db.relationshipTypes(), DROP + CREATE.
"""

from __future__ import annotations

from typing import Any, Protocol

from server.algorithm.cypher.query_compile import FT_NODE_INDEX, FT_REL_INDEX


class FulltextDb(Protocol):
    def query(self, cypher: str, **params: Any) -> list[dict]: ...
    def consume(self, cypher: str, **params: Any) -> None: ...


def _quote_ident(name: str) -> str:
    raw = str(name or "")
    if not raw.strip() or any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        raise ValueError(f"Invalid identifier: {name!r}")
    return "`" + raw.replace("`", "``") + "`"


def node_fulltext_cypher(labels: list[str]) -> str:
    union = "|".join(_quote_ident(label) for label in labels)
    return (
        f"CREATE FULLTEXT INDEX {_quote_ident(FT_NODE_INDEX)} "
        f"FOR (n:{union}) ON EACH [n.name]"
    )


def rel_fulltext_cypher(types: list[str]) -> str:
    union = "|".join(_quote_ident(rel_type) for rel_type in types)
    return (
        f"CREATE FULLTEXT INDEX {_quote_ident(FT_REL_INDEX)} "
        f"FOR ()-[r:{union}]-() ON EACH [r.evidence]"
    )


def _drop_cypher(name: str) -> str:
    return f"DROP INDEX {_quote_ident(name)} IF EXISTS"


def _fulltext_indexes(db: FulltextDb) -> dict[str, dict]:
    rows = db.query(
        """
        SHOW INDEXES
        YIELD name, type, entityType, labelsOrTypes, properties, state
        WHERE type = 'FULLTEXT'
        RETURN name, type, entityType, labelsOrTypes, properties, state
        """
    )
    out: dict[str, dict] = {}
    for row in rows:
        name = str(row.get("name") or "")
        if name:
            out[name] = row
    return out


def _labels(db: FulltextDb) -> list[str]:
    return sorted(
        {
            str(row["label"])
            for row in db.query("CALL db.labels() YIELD label RETURN label")
            if row.get("label")
        }
    )


def _rel_types(db: FulltextDb) -> list[str]:
    return sorted(
        {
            str(row["relationshipType"])
            for row in db.query(
                "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
            )
            if row.get("relationshipType")
        }
    )


def _union_matches(index: dict | None, current: list[str], *, entity: str, prop: str) -> bool:
    if not index:
        return False
    have = {str(x) for x in (index.get("labelsOrTypes") or [])}
    want = set(current)
    properties = [str(x) for x in (index.get("properties") or [])]
    entity_type = str(index.get("entityType") or "").upper()
    return have == want and properties == [prop] and entity_type == entity


def ensure_query_graph_fulltext(
    db: FulltextDb,
    *,
    recreate: bool = False,
    dry_run: bool = False,
) -> list[str]:
    """Create or rebuild query_graph Lucene indexes. Returns action tags."""
    labels = _labels(db)
    types = _rel_types(db)
    existing = _fulltext_indexes(db)
    actions: list[str] = []

    def handle(
        name: str,
        *,
        current: list[str],
        entity: str,
        prop: str,
        create: str,
        kind: str,
    ) -> None:
        if not current:
            print(f"No {kind}; skipped fulltext {name}")
            return
        info = existing.get(name)
        in_sync = _union_matches(info, current, entity=entity, prop=prop)
        drop = bool(info) and (recreate or not in_sync)
        create_needed = drop or not info
        if drop:
            print(f"DROP fulltext {name} ({'recreate' if recreate else 'label/type drift'})")
            if not dry_run:
                db.consume(_drop_cypher(name))
            actions.append(f"drop:{name}")
        if create_needed:
            print(f"CREATE fulltext {name} on {len(current)} {kind}")
            if not dry_run:
                db.consume(create)
            actions.append(f"create:{name}")
        else:
            print(f"Reusing fulltext {name} on {len(current)} {kind}")
            actions.append(f"reuse:{name}")

    handle(
        FT_NODE_INDEX,
        current=labels,
        entity="NODE",
        prop="name",
        create=node_fulltext_cypher(labels),
        kind="labels",
    )
    handle(
        FT_REL_INDEX,
        current=types,
        entity="RELATIONSHIP",
        prop="evidence",
        create=rel_fulltext_cypher(types),
        kind="relationship types",
    )
    return actions

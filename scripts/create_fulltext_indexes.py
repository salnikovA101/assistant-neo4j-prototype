#!/usr/bin/env python3
"""Create fulltext indexes used by query_graph (names + relationship evidence).

  .venv/bin/python scripts/create_fulltext_indexes.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.algorithm.cypher.query_compile import FT_NODE_INDEX, FT_REL_INDEX  # noqa: E402
from server.utils.config import load_config  # noqa: E402


def _driver(uri: str, user: str, password: str):
    candidates = [uri]
    if "host.docker.internal" in uri:
        candidates.append(uri.replace("host.docker.internal", "localhost"))
        candidates.append(uri.replace("host.docker.internal", "127.0.0.1"))
    last = None
    for cand in candidates:
        drv = GraphDatabase.driver(cand, auth=(user, password))
        try:
            drv.verify_connectivity()
            print(f"Connected: {cand}")
            return drv
        except ServiceUnavailable as e:
            last = e
            drv.close()
    raise last or RuntimeError("Neo4j unreachable")


def _quote_ident(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def main() -> int:
    cfg = load_config()
    driver = _driver(cfg.neo4j.uri, cfg.neo4j.user, cfg.neo4j.password)
    try:
        with driver.session() as session:
            labels = [
                rec["label"]
                for rec in session.run("CALL db.labels() YIELD label RETURN label")
                if rec["label"]
            ]
            types = [
                rec["relationshipType"]
                for rec in session.run(
                    "CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType"
                )
                if rec["relationshipType"]
            ]
            if labels:
                label_union = "|".join(_quote_ident(str(x)) for x in labels)
                session.run(
                    f"CREATE FULLTEXT INDEX {_quote_ident(FT_NODE_INDEX)} IF NOT EXISTS "
                    f"FOR (n:{label_union}) ON EACH [n.name]"
                )
                print(f"Ensured node fulltext {FT_NODE_INDEX} on {len(labels)} labels")
            else:
                print("No node labels; skipped node fulltext")
            if types:
                type_union = "|".join(_quote_ident(str(x)) for x in types)
                session.run(
                    f"CREATE FULLTEXT INDEX {_quote_ident(FT_REL_INDEX)} IF NOT EXISTS "
                    f"FOR ()-[r:{type_union}]-() ON EACH [r.evidence]"
                )
                print(f"Ensured relationship fulltext {FT_REL_INDEX} on {len(types)} types")
            else:
                print("No relationship types; skipped relationship fulltext")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

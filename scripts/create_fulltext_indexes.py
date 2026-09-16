#!/usr/bin/env python3
"""Create fulltext indexes used by query_graph (names + relationship evidence).

Thin wrapper around ensure_query_graph_fulltext (no Ollama). Also runs at the
end of scripts/vectorize_edges.py.

  .venv/bin/python scripts/create_fulltext_indexes.py
  .venv/bin/python scripts/create_fulltext_indexes.py --recreate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.algorithm.cypher.fulltext import ensure_query_graph_fulltext  # noqa: E402
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


class _SessionDb:
    def __init__(self, session: Any) -> None:
        self._session = session

    def query(self, cypher: str, **params: Any) -> list[dict]:
        return [dict(row) for row in self._session.run(cypher, **params)]

    def consume(self, cypher: str, **params: Any) -> None:
        self._session.run(cypher, **params).consume()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ensure query_graph Lucene indexes (DROP+CREATE on label/type drift)."
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="DROP and CREATE the Lucene indexes even if the label/type union matches.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    driver = _driver(cfg.neo4j.uri, cfg.neo4j.user, cfg.neo4j.password)
    try:
        with driver.session() as session:
            ensure_query_graph_fulltext(
                _SessionDb(session),
                recreate=bool(args.recreate),
                dry_run=bool(args.dry_run),
            )
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

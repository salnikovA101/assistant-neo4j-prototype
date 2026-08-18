#!/usr/bin/env python3
"""Drop/recreate relationship vector indexes with filterable run_id (Cypher 25).

Needed once so S2 can SEARCH … WHERE r.run_id = $run_id in-index.
Does not change L / L_raw_max; one index per relationship type, not per run_id.

  .venv/bin/python scripts/recreate_rel_vector_indexes.py
  .venv/bin/python scripts/recreate_rel_vector_indexes.py --dry-run
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.utils.config import load_config  # noqa: E402

_INDEX_NAME_RE_OK = __import__("re").compile(r"^[A-Za-z][A-Za-z0-9_]*$")


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


def _rel_indexes(session) -> list[dict]:
    rows = session.run(
        """
        SHOW VECTOR INDEXES
        YIELD name, entityType, labelsOrTypes, properties, state
        WHERE entityType = 'RELATIONSHIP'
          AND any(p IN properties WHERE p = 'evidence_embedding')
        RETURN name, labelsOrTypes, properties, state
        """
    )
    return [dict(r) for r in rows]


def _vector_dim(session) -> int:
    rec = session.run(
        """
        MATCH ()-[r]->()
        WHERE r.evidence_embedding IS NOT NULL
        RETURN size(r.evidence_embedding) AS dim
        LIMIT 1
        """
    ).single()
    if rec is None or rec["dim"] is None:
        raise SystemExit("No relationship.evidence_embedding found")
    return int(rec["dim"])


def _create_cypher(name: str, rel_type: str, dim: int) -> str:
    if not _INDEX_NAME_RE_OK.fullmatch(name):
        raise SystemExit(f"Unsafe index name: {name!r}")
    if not _INDEX_NAME_RE_OK.fullmatch(rel_type):
        raise SystemExit(f"Unsafe relationship type: {rel_type!r}")
    return f"""
CYPHER 25
CREATE VECTOR INDEX {name} IF NOT EXISTS
FOR ()-[r:{rel_type}]-() ON r.evidence_embedding
WITH [r.run_id]
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {int(dim)},
    `vector.similarity_function`: 'cosine'
  }}
}}
"""


def _wait_online(session, names: list[str], timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    pending = set(names)
    while pending and time.monotonic() < deadline:
        rows = session.run(
            """
            SHOW VECTOR INDEXES
            YIELD name, state
            WHERE name IN $names
            RETURN name, state
            """,
            names=list(pending),
        )
        states = {r["name"]: r["state"] for r in rows}
        still = set()
        for n in pending:
            st = str(states.get(n) or "")
            if st.upper() != "ONLINE":
                still.add(n)
        if not still:
            print("All relationship vector indexes ONLINE")
            return
        pending = still
        time.sleep(1.0)
    raise SystemExit(f"Indexes not ONLINE in time: {sorted(pending)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_config()
    drv = _driver(cfg.neo4j.uri, cfg.neo4j.user, cfg.neo4j.password)
    try:
        with drv.session() as session:
            dim = _vector_dim(session)
            indexes = _rel_indexes(session)
            if not indexes:
                raise SystemExit("No relationship vector indexes on evidence_embedding")
            print(f"vector.dimensions={dim}; indexes={len(indexes)}")
            plans: list[tuple[str, str, str]] = []
            for idx in indexes:
                name = idx["name"]
                types = list(idx.get("labelsOrTypes") or [])
                if len(types) != 1:
                    raise SystemExit(f"{name}: expected one rel type, got {types}")
                rel_type = str(types[0])
                cypher = _create_cypher(name, rel_type, dim)
                plans.append((name, rel_type, cypher))
                print(f"  {name} FOR ()-[r:{rel_type}]-() WITH [r.run_id]")

            if args.dry_run:
                print("dry-run: no DROP/CREATE")
                return

            for name, rel_type, cypher in plans:
                print(f"DROP INDEX {name}")
                session.run(f"DROP INDEX {name} IF EXISTS").consume()
                print(f"CREATE VECTOR INDEX {name} ({rel_type})")
                session.run(cypher).consume()
            _wait_online(session, [n for n, _, _ in plans])
    finally:
        drv.close()


if __name__ == "__main__":
    main()

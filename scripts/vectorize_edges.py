#!/usr/bin/env python3
"""Host script: embed relationship.evidence via OpenRouter, SET evidence_embedding.

Runs on the host (not the app container). Uses server.algorithm.embed_client
(nvidia/nemotron-3-embed-1b, no torch).

  .venv/bin/python scripts/vectorize_edges.py
  .venv/bin/python scripts/vectorize_edges.py --run-id full_corpus_20260713
  .venv/bin/python scripts/vectorize_edges.py --recreate-indexes
  .venv/bin/python scripts/vectorize_edges.py --force --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
import time
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.algorithm.embed_client import (  # noqa: E402
    OPENROUTER_DEFAULT_MODEL,
    get_embeddings_batch,
)
from server.utils.config import load_config  # noqa: E402

logger = logging.getLogger(__name__)

_INDEX_NAME_RE_OK = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_WRITE_BATCH = 32

_FETCH_CYPHER = """
MATCH ()-[r]->()
WHERE r.run_id = $run_id
  AND r.evidence IS NOT NULL
  AND trim(toString(r.evidence)) <> ''
RETURN elementId(r) AS eid,
       toString(r.evidence) AS evidence,
       r.evidence_embedding IS NOT NULL AS has_vec
"""

_SET_CYPHER = """
UNWIND $rows AS row
MATCH ()-[r]->()
WHERE elementId(r) = row.eid
SET r.evidence_embedding = row.embedding
"""


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


def recreate_rel_vector_indexes(session, *, dry_run: bool) -> None:
    """DROP/CREATE relationship vector indexes with filterable run_id (Cypher 25)."""
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

    if dry_run:
        print("dry-run: no DROP/CREATE")
        return

    for name, rel_type, cypher in plans:
        print(f"DROP INDEX {name}")
        session.run(f"DROP INDEX {name} IF EXISTS").consume()
        print(f"CREATE VECTOR INDEX {name} ({rel_type})")
        session.run(cypher).consume()
    _wait_online(session, [n for n, _, _ in plans])


def _pending_edges(session, run_id: str, *, force: bool) -> list[dict]:
    rows = [dict(r) for r in session.run(_FETCH_CYPHER, run_id=run_id)]
    with_evidence = len(rows)
    already = sum(1 for r in rows if r["has_vec"])
    pending = rows if force else [r for r in rows if not r["has_vec"]]
    print(
        f"run_id={run_id}: {with_evidence} edges with evidence, "
        f"{already} already embedded, {len(pending)} to embed"
        f"{' (--force)' if force else ''}"
    )
    return pending


async def _embed_and_write(session, pending: list[dict]) -> None:
    total = len(pending)
    if not total:
        print("Nothing to embed")
        return
    done = 0
    for i in range(0, total, _WRITE_BATCH):
        chunk = pending[i : i + _WRITE_BATCH]
        texts = [c["evidence"] for c in chunk]
        vecs = await get_embeddings_batch(
            texts,
            model_id=OPENROUTER_DEFAULT_MODEL,
        )
        if len(vecs) != len(chunk):
            raise SystemExit(
                f"embedding count mismatch: got {len(vecs)} want {len(chunk)}"
            )
        session.run(
            _SET_CYPHER,
            rows=[
                {"eid": c["eid"], "embedding": vec}
                for c, vec in zip(chunk, vecs)
            ],
        ).consume()
        done += len(chunk)
        print(f"  SET evidence_embedding {done}/{total}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Embed r.evidence via OpenRouter nemotron and SET r.evidence_embedding. "
            "Host-only; key from OPENROUTER_API_KEY or LLM__PROFILES__OTHER__API_KEY."
        )
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Relationship run_id (default: server/config.yaml run_id)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed edges that already have evidence_embedding",
    )
    parser.add_argument(
        "--recreate-indexes",
        action="store_true",
        help=(
            "After embedding, DROP/CREATE relationship vector indexes "
            "ON r.evidence_embedding WITH [r.run_id]"
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config()
    run_id = (args.run_id if args.run_id is not None else cfg.run_id) or ""
    run_id = str(run_id).strip()
    if not run_id:
        raise SystemExit(
            "run_id is empty: pass --run-id or set run_id in server/config.yaml"
        )

    drv = _driver(cfg.neo4j.uri, cfg.neo4j.user, cfg.neo4j.password)
    try:
        with drv.session() as session:
            pending = _pending_edges(session, run_id, force=args.force)
            if args.dry_run:
                print("dry-run: no OpenRouter calls, no SET")
            else:
                asyncio.run(_embed_and_write(session, pending))
            if args.recreate_indexes:
                recreate_rel_vector_indexes(session, dry_run=args.dry_run)
    finally:
        drv.close()


if __name__ == "__main__":
    main()

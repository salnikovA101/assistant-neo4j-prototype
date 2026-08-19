#!/usr/bin/env python3
"""Host script: embed relationship.evidence via OpenRouter, SET evidence_embedding.

Runs on the host (not the app container). Uses server.algorithm.embed_client
(nvidia/nemotron-3-embed-1b, no torch).

  .venv/bin/python scripts/vectorize_edges.py
  .venv/bin/python scripts/vectorize_edges.py --run-id full_corpus_20260713
  .venv/bin/python scripts/vectorize_edges.py --recreate-indexes --yes
  .venv/bin/python scripts/vectorize_edges.py --force --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

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
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", "host.docker.internal"}
T = TypeVar("T")

_COUNT_CYPHER = """
MATCH ()-[r]->()
WHERE r.run_id = $run_id
  AND r.evidence IS NOT NULL
  AND trim(toString(r.evidence)) <> ''
RETURN count(*) AS with_evidence,
       sum(
         CASE
           WHEN r.evidence_embedding IS NOT NULL
            AND size(r.evidence_embedding) > 0 THEN 1
           ELSE 0
         END
       ) AS already
"""

_FETCH_PENDING_CYPHER = """
MATCH ()-[r]->()
WHERE r.run_id = $run_id
  AND r.evidence IS NOT NULL
  AND trim(toString(r.evidence)) <> ''
  AND (
    $force
    OR r.evidence_embedding IS NULL
    OR size(r.evidence_embedding) = 0
  )
RETURN elementId(r) AS eid, toString(r.evidence) AS evidence
"""

_SET_CYPHER = """
UNWIND $rows AS row
MATCH ()-[r]->()
WHERE elementId(r) = row.eid
SET r.evidence_embedding = row.embedding
"""

_DIMS_CYPHER = """
MATCH ()-[r]->()
WHERE r.evidence_embedding IS NOT NULL
  AND size(r.evidence_embedding) > 0
  AND ($run_id = '' OR r.run_id = $run_id)
RETURN DISTINCT size(r.evidence_embedding) AS dim
"""

_REL_TYPES_WITH_VEC_CYPHER = """
MATCH ()-[r]->()
WHERE r.evidence_embedding IS NOT NULL
  AND size(r.evidence_embedding) > 0
RETURN DISTINCT type(r) AS rel_type
ORDER BY rel_type
"""


def _redact_uri(uri: str) -> str:
    if "@" not in uri or "://" not in uri:
        return uri
    scheme, rest = uri.split("://", 1)
    hostpart = rest.rsplit("@", 1)[-1]
    return f"{scheme}://***@{hostpart}"


def _uri_host(uri: str) -> str:
    parsed = urlparse(uri)
    if parsed.hostname:
        return parsed.hostname
    rest = uri.split("://", 1)[-1]
    rest = rest.rsplit("@", 1)[-1]
    if rest.startswith("["):
        return rest[1:].split("]", 1)[0]
    return rest.split("/", 1)[0].split(":", 1)[0]


def _warn_insecure_bolt(uri: str) -> None:
    scheme = uri.split("://", 1)[0].lower()
    if scheme in {"bolt+s", "bolt+ssc", "neo4j+s", "neo4j+ssc"}:
        return
    host = (_uri_host(uri) or "").lower()
    if host in _LOCAL_HOSTS:
        return
    print(
        f"WARNING: unencrypted Neo4j URI {_redact_uri(uri)} "
        "(use bolt+s:// or neo4j+s:// for a remote host)"
    )


def _confirm(prompt: str, *, yes: bool, dry_run: bool) -> None:
    if dry_run or yes:
        return
    if not sys.stdin.isatty():
        raise SystemExit(f"{prompt}  Refusing without a TTY; pass --yes.")
    ans = input(f"{prompt} [y/N] ").strip().lower()
    if ans not in {"y", "yes"}:
        raise SystemExit("aborted")


def _index_name_for_rel_type(rel_type: str) -> str:
    name = f"{rel_type}_evidence_index"
    if not _INDEX_NAME_RE_OK.fullmatch(rel_type):
        raise SystemExit(f"Unsafe relationship type: {rel_type!r}")
    if not _INDEX_NAME_RE_OK.fullmatch(name):
        raise SystemExit(f"Unsafe index name: {name!r}")
    return name


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


def _connect(uri: str, user: str, password: str):
    candidates = [uri]
    if "host.docker.internal" in uri:
        candidates.append(uri.replace("host.docker.internal", "localhost"))
        candidates.append(uri.replace("host.docker.internal", "127.0.0.1"))
    last: Exception | None = None
    for cand in candidates:
        drv = GraphDatabase.driver(cand, auth=(user, password))
        try:
            drv.verify_connectivity()
            print(f"Connected: {_redact_uri(cand)}")
            _warn_insecure_bolt(cand)
            return drv
        except AuthError:
            drv.close()
            raise SystemExit(f"Neo4j auth failed for {_redact_uri(cand)}") from None
        except (ServiceUnavailable, OSError) as e:
            last = e
            drv.close()
    raise SystemExit(f"Neo4j unreachable: {last}")


class _Neo4j:
    def __init__(self, uri: str, user: str, password: str):
        self._uri = uri
        self._user = user
        self._password = password
        self._drv = _connect(uri, user, password)

    def close(self) -> None:
        self._drv.close()

    def _reconnect(self) -> None:
        try:
            self._drv.close()
        except Exception:
            pass
        self._drv = _connect(self._uri, self._user, self._password)

    def _retry(self, fn: Callable[[Any], T]) -> T:
        try:
            return fn(self._drv)
        except (ServiceUnavailable, OSError):
            print("Neo4j connection dropped; reconnecting")
            self._reconnect()
            return fn(self._drv)

    def query(self, cypher: str, **params: Any) -> list[dict]:
        def _once(drv: Any) -> list[dict]:
            with drv.session() as session:
                return [dict(r) for r in session.run(cypher, **params)]

        return self._retry(_once)

    def consume(self, cypher: str, **params: Any) -> None:
        def _once(drv: Any) -> None:
            with drv.session() as session:
                session.run(cypher, **params).consume()

        self._retry(_once)


def _rel_indexes(db: _Neo4j) -> list[dict]:
    return db.query(
        """
        SHOW VECTOR INDEXES
        YIELD name, entityType, labelsOrTypes, properties, state
        WHERE entityType = 'RELATIONSHIP'
          AND any(p IN properties WHERE p = 'evidence_embedding')
        RETURN name, labelsOrTypes, properties, state
        """
    )


def _unique_dim(db: _Neo4j, run_id: str = "") -> int | None:
    rows = db.query(_DIMS_CYPHER, run_id=run_id)
    dims = sorted({int(r["dim"]) for r in rows if r.get("dim") is not None})
    if not dims:
        return None
    if len(dims) > 1:
        scope = f"run_id={run_id}" if run_id else "database"
        raise SystemExit(
            f"Mixed evidence_embedding dimensions in {scope}: {dims}. "
            "Re-embed every corpus with the same model before --recreate-indexes."
        )
    return dims[0]


def _wait_online(db: _Neo4j, names: list[str], timeout_s: float = 180.0) -> None:
    deadline = time.monotonic() + timeout_s
    pending = set(names)
    while pending and time.monotonic() < deadline:
        rows = db.query(
            """
            SHOW VECTOR INDEXES
            YIELD name, state
            WHERE name IN $names
            RETURN name, state
            """,
            names=list(pending),
        )
        states = {r["name"]: r["state"] for r in rows}
        still = {
            n
            for n in pending
            if str(states.get(n) or "").upper() != "ONLINE"
        }
        if not still:
            print("All relationship vector indexes ONLINE")
            return
        pending = still
        time.sleep(1.0)
    raise SystemExit(f"Indexes not ONLINE in time: {sorted(pending)}")


def recreate_rel_vector_indexes(db: _Neo4j, *, dry_run: bool) -> None:
    """DROP/CREATE relationship vector indexes with filterable run_id (Cypher 25)."""
    dim = _unique_dim(db)
    if dim is None:
        raise SystemExit("No relationship.evidence_embedding found")
    indexes = _rel_indexes(db)
    plans: list[tuple[str, str, str]] = []
    if indexes:
        for idx in indexes:
            name = idx["name"]
            types = list(idx.get("labelsOrTypes") or [])
            if len(types) != 1:
                raise SystemExit(f"{name}: expected one rel type, got {types}")
            rel_type = str(types[0])
            plans.append((name, rel_type, _create_cypher(name, rel_type, dim)))
    else:
        types = [str(r["rel_type"]) for r in db.query(_REL_TYPES_WITH_VEC_CYPHER)]
        if not types:
            raise SystemExit("No relationship types with evidence_embedding")
        print("No relationship vector indexes yet; creating from embedded types")
        for rel_type in types:
            name = _index_name_for_rel_type(rel_type)
            plans.append((name, rel_type, _create_cypher(name, rel_type, dim)))

    print(
        f"vector.dimensions={dim}; indexes={len(plans)} "
        "(per relationship type, ALL run_ids — not scoped to --run-id)"
    )
    for name, rel_type, _cypher in plans:
        print(f"  {name} FOR ()-[r:{rel_type}]-() WITH [r.run_id]")

    if dry_run:
        print("dry-run: no DROP/CREATE")
        return

    for name, rel_type, cypher in plans:
        print(f"DROP INDEX {name}")
        db.consume(f"DROP INDEX {name} IF EXISTS")
        print(f"CREATE VECTOR INDEX {name} ({rel_type})")
        db.consume(cypher)
    _wait_online(db, [n for n, _, _ in plans])


def _pending_edges(db: _Neo4j, run_id: str, *, force: bool) -> list[dict]:
    counts = db.query(_COUNT_CYPHER, run_id=run_id)
    row = counts[0] if counts else {}
    with_evidence = int(row.get("with_evidence") or 0)
    already = int(row.get("already") or 0)
    pending = db.query(_FETCH_PENDING_CYPHER, run_id=run_id, force=force)
    print(
        f"run_id={run_id}: {with_evidence} edges with evidence, "
        f"{already} already embedded, {len(pending)} to embed"
        f"{' (--force)' if force else ''}"
    )
    return pending


def _assert_batch_dim(
    vecs: list[list[float]],
    expected: int | None,
) -> int:
    if not vecs or not vecs[0]:
        raise SystemExit("embedding backend returned an empty vector")
    dim = len(vecs[0])
    for i, vec in enumerate(vecs):
        if not vec:
            raise SystemExit(f"empty embedding at batch offset {i}")
        if len(vec) != dim:
            raise SystemExit(
                f"in-batch dimension mismatch: {len(vec)} vs {dim}"
            )
    if expected is not None and dim != expected:
        raise SystemExit(
            f"embedding dim {dim} does not match existing evidence_embedding "
            f"dim {expected}. Same model as the rest of the graph is required "
            "(or re-embed every corpus, then --recreate-indexes)."
        )
    return dim


async def _embed_and_write(
    db: _Neo4j,
    pending: list[dict],
    *,
    expected_dim: int | None,
) -> int | None:
    total = len(pending)
    if not total:
        print("Nothing to embed")
        return expected_dim
    print(
        f"Sending {total} evidence texts to OpenRouter "
        f"({OPENROUTER_DEFAULT_MODEL})"
    )
    done = 0
    dim = expected_dim
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
        dim = _assert_batch_dim(vecs, dim)
        db.consume(
            _SET_CYPHER,
            rows=[
                {"eid": c["eid"], "embedding": vec}
                for c, vec in zip(chunk, vecs)
            ],
        )
        done += len(chunk)
        print(f"  SET evidence_embedding {done}/{total} (dim={dim})")
    return dim


def _load_env() -> None:
    load_dotenv(ROOT / ".env", override=False)


def _require_neo4j_password() -> None:
    if not (os.environ.get("NEO4J__PASSWORD") or "").strip():
        raise SystemExit(
            "NEO4J__PASSWORD is not set. Put it in the repo-root .env "
            "(see .env.example). Refusing the hardcoded Neo4jConfig default."
        )


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
            "ON r.evidence_embedding WITH [r.run_id] (all run_ids, every rel type)"
        ),
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Do not prompt for --force / --recreate-indexes",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    _load_env()
    _require_neo4j_password()
    cfg = load_config()
    run_id = (args.run_id if args.run_id is not None else cfg.run_id) or ""
    run_id = str(run_id).strip()
    if not run_id:
        raise SystemExit(
            "run_id is empty: pass --run-id or set run_id in server/config.yaml"
        )

    db = _Neo4j(cfg.neo4j.uri, cfg.neo4j.user, cfg.neo4j.password)
    try:
        pending = _pending_edges(db, run_id, force=args.force)
        if args.force and pending:
            _confirm(
                f"Overwrite evidence_embedding on {len(pending)} edges "
                f"(sends texts to OpenRouter).",
                yes=args.yes,
                dry_run=args.dry_run,
            )
        expected_dim = _unique_dim(db)  # whole graph; refuse mixed dims
        if expected_dim is None:
            expected_dim = _unique_dim(db, run_id)
        if args.dry_run:
            print("dry-run: no OpenRouter calls, no SET")
        else:
            asyncio.run(
                _embed_and_write(db, pending, expected_dim=expected_dim)
            )
        if args.recreate_indexes:
            n_idx = len(_rel_indexes(db))
            _confirm(
                "DROP/CREATE relationship vector indexes for ALL run_ids "
                f"(currently {n_idx} index(es); ANN is down until ONLINE).",
                yes=args.yes,
                dry_run=args.dry_run,
            )
            recreate_rel_vector_indexes(db, dry_run=args.dry_run)
    finally:
        db.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Host script: embed relationship.evidence via local Ollama, SET evidence_embedding.

Runs on the host (not the app container). Uses server.algorithm.embed_client
(embeddinggemma:300m-qat-q8_0, no torch). Stop the Docker app first on a 3 GB VM.

  .venv/bin/python scripts/vectorize_edges.py
  .venv/bin/python scripts/vectorize_edges.py --run-id full_corpus_20260713
  .venv/bin/python scripts/vectorize_edges.py --force --recreate-indexes --yes
  .venv/bin/python scripts/vectorize_edges.py --force --dry-run
  .venv/bin/python scripts/bench_embed.py
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import logging
import os
import re
import sys
import time
import unicodedata
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar
from urllib.parse import urlparse

from dotenv import load_dotenv
from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ServiceUnavailable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from server.algorithm.embed import format_document  # noqa: E402
from server.algorithm.embed_client import (  # noqa: E402
    DEFAULT_EMBED_MODEL,
    get_embeddings_batch,
    _resolve_embed_settings,
)
from server.utils.config import DEFAULT_WORKSPACE, load_config, workspace_run_id  # noqa: E402

logger = logging.getLogger(__name__)

_WRITE_BATCH = 8
_READ_BATCH = 256
_INDEX_PREFIX = "rel_ev_v1_"
_INDEX_SLUG_MAX = 40
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
       ) AS already,
       sum(
         CASE
           WHEN $force
             OR r.evidence_embedding IS NULL
             OR size(r.evidence_embedding) = 0 THEN 1
           ELSE 0
         END
       ) AS pending
"""

_FETCH_PENDING_BATCH_CYPHER = """
MATCH ()-[r]->()
WHERE r.run_id = $run_id
  AND r.evidence IS NOT NULL
  AND trim(toString(r.evidence)) <> ''
  AND (
    $force
    OR r.evidence_embedding IS NULL
    OR size(r.evidence_embedding) = 0
  )
  AND ($after_eid = '' OR elementId(r) > $after_eid)
RETURN elementId(r) AS eid, toString(r.evidence) AS evidence
ORDER BY eid
LIMIT $limit
"""

_SET_CYPHER = """
UNWIND $rows AS row
MATCH ()-[r]->()
WHERE elementId(r) = row.eid
  AND r.run_id = $run_id
SET r.evidence_embedding = row.embedding
"""

_COUNT_MISSING_CYPHER = """
MATCH ()-[r]->()
WHERE r.run_id = $run_id
  AND r.evidence IS NOT NULL
  AND trim(toString(r.evidence)) <> ''
  AND (
    r.evidence_embedding IS NULL
    OR size(r.evidence_embedding) = 0
  )
RETURN count(*) AS pending
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

_REL_TYPES_FOR_RUN_CYPHER = """
MATCH ()-[r]->()
WHERE r.run_id = $run_id
  AND r.evidence IS NOT NULL
  AND trim(toString(r.evidence)) <> ''
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


def _identifier(value: str, *, kind: str) -> str:
    raw = str(value or "")
    if not raw.strip() or any(ord(ch) < 32 or ord(ch) == 127 for ch in raw):
        raise SystemExit(f"Invalid {kind}: {value!r}")
    return raw


def _quote_identifier(value: str, *, kind: str) -> str:
    clean = _identifier(value, kind=kind)
    return f"`{clean.replace('`', '``')}`"


def _index_name_for_rel_type(rel_type: str) -> str:
    """Stable, collision-resistant name for any valid Neo4j relationship type."""
    clean = _identifier(rel_type, kind="relationship type")
    ascii_name = unicodedata.normalize("NFKD", clean).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.casefold()).strip("_")
    slug = (slug[:_INDEX_SLUG_MAX].rstrip("_") or "type")
    digest = hashlib.sha256(clean.encode("utf-8")).hexdigest()[:10]
    return f"{_INDEX_PREFIX}{slug}_{digest}"


def _create_cypher(name: str, rel_type: str, dim: int) -> str:
    quoted_name = _quote_identifier(name, kind="index name")
    quoted_rel_type = _quote_identifier(rel_type, kind="relationship type")
    if int(dim) <= 0:
        raise SystemExit(f"Invalid vector dimension: {dim!r}")
    return f"""
CYPHER 25
CREATE VECTOR INDEX {quoted_name} IF NOT EXISTS
FOR ()-[r:{quoted_rel_type}]-() ON (r.evidence_embedding)
WITH [r.run_id]
OPTIONS {{
  indexConfig: {{
    `vector.dimensions`: {int(dim)},
    `vector.similarity_function`: 'cosine'
  }}
}}
"""


def _drop_cypher(name: str) -> str:
    return f"DROP INDEX {_quote_identifier(name, kind='index name')} IF EXISTS"


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
        YIELD name, entityType, labelsOrTypes, properties, state, options
        WHERE entityType = 'RELATIONSHIP'
          AND any(p IN properties WHERE p = 'evidence_embedding')
        RETURN name, labelsOrTypes, properties, state, options
        """
    )


def _index_dimension(index: Mapping[str, Any]) -> int | None:
    options = index.get("options")
    if not isinstance(options, Mapping):
        return None
    config = options.get("indexConfig")
    if not isinstance(config, Mapping):
        return None
    value = config.get("vector.dimensions")
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _single_index_type(index: Mapping[str, Any]) -> str | None:
    types = [str(value) for value in (index.get("labelsOrTypes") or []) if str(value)]
    properties = [str(value) for value in (index.get("properties") or [])]
    if len(types) != 1 or "evidence_embedding" not in properties:
        return None
    return types[0]


def _required_rel_types(db: _Neo4j, run_id: str) -> list[str]:
    values = {
        str(row["rel_type"])
        for row in db.query(_REL_TYPES_WITH_VEC_CYPHER)
        if row.get("rel_type") is not None
    }
    values.update(
        str(row["rel_type"])
        for row in db.query(_REL_TYPES_FOR_RUN_CYPHER, run_id=run_id)
        if row.get("rel_type") is not None
    )
    return sorted(values, key=lambda value: (value.casefold(), value))


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


def ensure_rel_vector_indexes(
    db: _Neo4j,
    *,
    run_id: str,
    dim: int,
    dry_run: bool,
) -> list[str]:
    """Add missing per-type indexes without replacing compatible indexes."""
    types = _required_rel_types(db, run_id)
    if not types:
        print("No relationship types with evidence; no vector indexes needed")
        return []

    by_type: dict[str, list[dict]] = {}
    for index in _rel_indexes(db):
        rel_type = _single_index_type(index)
        if rel_type is not None:
            by_type.setdefault(rel_type, []).append(index)

    created: list[str] = []
    reused: list[str] = []
    wait_for: list[str] = []
    for rel_type in types:
        candidates = by_type.get(rel_type, [])
        filterable = [
            index
            for index in candidates
            if "run_id"
            in [str(value) for value in (index.get("properties") or [])]
        ]
        compatible: list[dict] = []
        for index in filterable:
            index_dim = _index_dimension(index)
            if index_dim is not None and index_dim != dim:
                continue
            compatible.append(index)
        if filterable and not compatible:
            found = sorted(
                {value for value in (_index_dimension(item) for item in filterable) if value is not None}
            )
            raise SystemExit(
                f"{rel_type}: existing run-filterable evidence vector index "
                f"dimension(s) {found} "
                f"do not match embedding dimension {dim}; use --force for every "
                "corpus and then --recreate-indexes --yes"
            )
        if compatible:
            preferred = next(
                (item for item in compatible if str(item.get("name") or "").startswith(_INDEX_PREFIX)),
                compatible[0],
            )
            name = str(preferred.get("name") or "")
            reused.append(name)
            if str(preferred.get("state") or "").upper() != "ONLINE":
                wait_for.append(name)
            continue

        name = _index_name_for_rel_type(rel_type)
        if any(str(item.get("name") or "") == name for item in candidates):
            raise SystemExit(
                f"{rel_type}: index {name} exists without filterable run_id; "
                "use --recreate-indexes --yes after vectorizing every corpus"
            )
        created.append(name)
        print(f"CREATE {name} FOR relationship type {rel_type!r} WITH [r.run_id]")
        if not dry_run:
            db.consume(_create_cypher(name, rel_type, dim))
            wait_for.append(name)

    if reused:
        print(f"Reusing {len(reused)} compatible relationship vector index(es)")
    if not created:
        print("All required relationship vector indexes already exist")
    elif dry_run:
        print(f"dry-run: would create {len(created)} missing index(es)")
    if wait_for and not dry_run:
        _wait_online(db, wait_for)
    return created


def recreate_rel_vector_indexes(db: _Neo4j, *, run_id: str, dry_run: bool) -> None:
    """DROP/CREATE relationship vector indexes with filterable run_id (Cypher 25)."""
    dim = _unique_dim(db)
    if dim is None:
        raise SystemExit("No relationship.evidence_embedding found")
    indexes = _rel_indexes(db)
    types = _required_rel_types(db, run_id)
    if not types:
        raise SystemExit("No relationship types with evidence_embedding")
    plans = [
        (
            _index_name_for_rel_type(rel_type),
            rel_type,
            _create_cypher(_index_name_for_rel_type(rel_type), rel_type, dim),
        )
        for rel_type in types
    ]

    print(
        f"vector.dimensions={dim}; indexes={len(plans)} "
        "(per relationship type, ALL run_ids — not scoped to --run-id)"
    )
    for name, rel_type, _cypher in plans:
        print(f"  {name} FOR relationship type {rel_type!r} WITH [r.run_id]")

    if dry_run:
        print("dry-run: no DROP/CREATE")
        return

    for index in indexes:
        name = str(index.get("name") or "")
        print(f"DROP INDEX {name}")
        db.consume(_drop_cypher(name))
    for name, rel_type, cypher in plans:
        print(f"CREATE VECTOR INDEX {name} ({rel_type})")
        db.consume(cypher)
    _wait_online(db, [n for n, _, _ in plans])


def _pending_counts(db: _Neo4j, run_id: str, *, force: bool) -> dict[str, int]:
    counts = db.query(_COUNT_CYPHER, run_id=run_id, force=force)
    row = counts[0] if counts else {}
    with_evidence = int(row.get("with_evidence") or 0)
    already = int(row.get("already") or 0)
    pending = int(row.get("pending") or 0)
    print(
        f"run_id={run_id}: {with_evidence} edges with evidence, "
        f"{already} already embedded, {pending} to embed"
        f"{' (--force)' if force else ''}"
    )
    return {"with_evidence": with_evidence, "already": already, "pending": pending}


def _pending_batch(
    db: _Neo4j,
    run_id: str,
    *,
    force: bool,
    after_eid: str,
    limit: int = _READ_BATCH,
) -> list[dict]:
    return db.query(
        _FETCH_PENDING_BATCH_CYPHER,
        run_id=run_id,
        force=force,
        after_eid=after_eid,
        limit=int(limit),
    )


def _assert_no_missing_vectors(db: _Neo4j, run_id: str) -> None:
    rows = db.query(_COUNT_MISSING_CYPHER, run_id=run_id)
    missing = int(rows[0].get("pending") or 0) if rows else 0
    if missing:
        raise SystemExit(
            f"run_id={run_id}: {missing} relationship(s) still have evidence "
            "but no evidence_embedding"
        )
    print(f"run_id={run_id}: every relationship with evidence has an embedding")


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
    *,
    run_id: str,
    force: bool,
    pending_count: int,
    expected_dim: int | None,
) -> int | None:
    total = int(pending_count)
    if not total:
        print("Nothing to embed")
        return expected_dim
    _backend, model, embeddings_url, _headers = _resolve_embed_settings()
    print(
        f"Sending {total} evidence texts to Ollama "
        f"model={model} url={embeddings_url}"
    )
    done = 0
    dim = expected_dim
    after_eid = ""
    while True:
        page = _pending_batch(
            db,
            run_id,
            force=force,
            after_eid=after_eid,
        )
        if not page:
            break
        for i in range(0, len(page), _WRITE_BATCH):
            chunk = page[i : i + _WRITE_BATCH]
            texts = [format_document(c["evidence"]) for c in chunk]
            vecs = await get_embeddings_batch(
                texts,
                model_id=model,
            )
            if len(vecs) != len(chunk):
                raise SystemExit(
                    f"embedding count mismatch: got {len(vecs)} want {len(chunk)}"
                )
            dim = _assert_batch_dim(vecs, dim)
            db.consume(
                _SET_CYPHER,
                run_id=run_id,
                rows=[
                    {"eid": c["eid"], "embedding": vec}
                    for c, vec in zip(chunk, vecs)
                ],
            )
            done += len(chunk)
            print(f"  SET evidence_embedding {done}/{total} (dim={dim})")
        after_eid = str(page[-1]["eid"])
        if len(page) < _READ_BATCH:
            break
    if done != total:
        raise SystemExit(
            f"relationship set changed during vectorization: embedded {done}, "
            f"initially planned {total}; rerun the command"
        )
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
            "Embed r.evidence via local Ollama (embeddinggemma:300m-qat-q8_0) "
            "and SET r.evidence_embedding. Host-only. Stop docker compose app "
            "on the 3 GB VM first. Missing per-relationship-type indexes are "
            "created additively. After a model/dim change: re-embed every run, "
            "then use --recreate-indexes --yes."
        )
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Relationship run_id (default: server/config.yaml workspaces.packaging)",
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
    run_id = (
        args.run_id
        if args.run_id is not None
        else workspace_run_id(cfg, DEFAULT_WORKSPACE)
    ) or ""
    run_id = str(run_id).strip()
    if not run_id:
        raise SystemExit(
            "run_id is empty: pass --run-id or configure workspaces.packaging"
        )

    db = _Neo4j(cfg.neo4j.uri, cfg.neo4j.user, cfg.neo4j.password)
    try:
        counts = _pending_counts(db, run_id, force=args.force)
        pending_count = counts["pending"]
        if args.force and pending_count:
            _confirm(
                f"Overwrite evidence_embedding on {pending_count} edges "
                f"(sends texts to local Ollama {DEFAULT_EMBED_MODEL}).",
                yes=args.yes,
                dry_run=args.dry_run,
            )
        # --force is also the recovery path from a database that temporarily
        # contains mixed dimensions while corpora are migrated one by one.
        existing_dim = None if args.force else _unique_dim(db)
        expected_dim = existing_dim
        if args.force:
            print("--force: vectors for this run will be replaced")
        if args.dry_run:
            print("dry-run: no Ollama calls, no SET")
            actual_dim = existing_dim
        else:
            actual_dim = asyncio.run(
                _embed_and_write(
                    db,
                    run_id=run_id,
                    force=args.force,
                    pending_count=pending_count,
                    expected_dim=expected_dim,
                )
            )
            _assert_no_missing_vectors(db, run_id)
            # A single index is shared by all run_ids of one relationship type.
            # Mixed dimensions therefore cannot be served safely.
            actual_dim = _unique_dim(db)
        if args.recreate_indexes:
            n_idx = len(_rel_indexes(db))
            _confirm(
                "DROP/CREATE relationship vector indexes for ALL run_ids "
                f"(currently {n_idx} index(es); ANN is down until ONLINE).",
                yes=args.yes,
                dry_run=args.dry_run,
            )
            recreate_rel_vector_indexes(
                db,
                run_id=run_id,
                dry_run=args.dry_run,
            )
        elif actual_dim is None:
            print("No vector dimension available; index creation is deferred")
        else:
            ensure_rel_vector_indexes(
                db,
                run_id=run_id,
                dim=actual_dim,
                dry_run=args.dry_run,
            )
    finally:
        db.close()


if __name__ == "__main__":
    main()

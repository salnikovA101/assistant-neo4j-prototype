"""Embedding HTTP client + Neo4j vector-index discovery."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
from neo4j import AsyncDriver

from server.utils.constants import EmbeddingBackend

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Embedding HTTP/backend failed after retries."""


DEFAULT_EMBED_MODEL = "embeddinggemma:300m-qat-q8_0"
DEFAULT_EMBED_API_KEY = "ollama"
_HOST_EMBED_BASE_URL = "http://127.0.0.1:11434/v1"
_DOCKER_EMBED_BASE_URL = "http://host.docker.internal:11434/v1"
_MAX_RETRIES = 5
_RETRY_BASE_SEC = 2.0
_BATCH_SIZE = 8
_MAX_INPUT_CHARS = 8192


def running_in_docker() -> bool:
    return Path("/.dockerenv").exists()


def default_embed_base_url() -> str:
    """Ollama on the VM host: Docker uses host-gateway, scripts use loopback."""
    if running_in_docker():
        return _DOCKER_EMBED_BASE_URL
    return _HOST_EMBED_BASE_URL


def _ensure_env_loaded() -> None:
    from dotenv import load_dotenv

    root = Path(__file__).resolve().parents[2]
    env_path = root / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)


def _resolve_embed_settings(
    model_id: str | None = None,
    url: str | None = None,
    backend: EmbeddingBackend | str | None = None,
) -> tuple[EmbeddingBackend, str, str, dict[str, str]]:
    """Resolve Ollama model, embeddings URL, and request headers."""
    resolved_backend = EmbeddingBackend(backend or EmbeddingBackend.OLLAMA)
    _ensure_env_loaded()
    model = (
        model_id
        or (os.environ.get("EMBED__MODEL") or "").strip()
        or DEFAULT_EMBED_MODEL
    )
    base = (
        url
        or (os.environ.get("EMBED__BASE_URL") or "").strip()
        or default_embed_base_url()
    ).rstrip("/")
    key = (os.environ.get("EMBED__API_KEY") or "").strip() or DEFAULT_EMBED_API_KEY
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }

    embeddings_url = base if base.endswith("/embeddings") else f"{base}/embeddings"
    return resolved_backend, model, embeddings_url, headers


def _truncate_input(text: str, max_chars: int) -> str:
    t = text or ""
    if max_chars <= 0 or len(t) <= max_chars:
        return t
    return t[: max_chars - 1].rstrip() + "…"


async def _post_embeddings_once(
    client: httpx.AsyncClient,
    *,
    embeddings_url: str,
    headers: dict[str, str],
    payload: dict,
    resolved_backend: EmbeddingBackend,
) -> list[list[float]]:
    last_err: Exception | str | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            response = await client.post(
                embeddings_url, headers=headers, json=payload, timeout=60.0
            )
            if response.status_code in (429, 500, 502, 503, 504):
                last_err = f"HTTP {response.status_code}"
                wait = _RETRY_BASE_SEC * (2**attempt)
                logger.warning(
                    "Embeddings HTTP %s (%s), retry in %.1fs (%s/%s)",
                    response.status_code,
                    resolved_backend,
                    wait,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                if attempt + 1 >= _MAX_RETRIES:
                    break
                await asyncio.sleep(wait)
                continue
            response.raise_for_status()
            body = response.json()
            data = body.get("data")
            if not isinstance(data, list):
                raise EmbeddingError("embedding response missing data[]")
            inputs = payload.get("input") if isinstance(payload.get("input"), list) else []
            parsed: list[tuple[int, list[float]]] = []
            missing_index = False
            for offset, item in enumerate(data):
                if not isinstance(item, dict) or "embedding" not in item:
                    raise EmbeddingError("embedding response missing embedding")
                if "index" not in item:
                    missing_index = True
                    parsed.append((offset, item["embedding"]))
                    continue
                parsed.append((int(item["index"]), item["embedding"]))
            if missing_index:
                if len(data) != len(inputs):
                    raise EmbeddingError("embedding response missing index")
            else:
                parsed.sort(key=lambda pair: pair[0])
                expected = list(range(len(inputs))) if inputs else [i for i, _ in parsed]
                got = [i for i, _ in parsed]
                if inputs and got != expected:
                    raise EmbeddingError(
                        f"embedding index mismatch: got {got} want {expected}"
                    )
            return [emb for _, emb in parsed]
        except EmbeddingError:
            raise
        except Exception as e:
            last_err = e
            wait = _RETRY_BASE_SEC * (2**attempt)
            logger.error(
                "Error fetching embeddings via %s (attempt %s/%s): %s",
                resolved_backend,
                attempt + 1,
                _MAX_RETRIES,
                e,
            )
            if attempt + 1 >= _MAX_RETRIES:
                break
            await asyncio.sleep(wait)
    raise EmbeddingError(
        f"embeddings failed after {_MAX_RETRIES} retries: {last_err}"
    )


async def get_embeddings_batch(
    texts: list[str],
    model_id: str | None = None,
    url: str | None = None,
    backend: EmbeddingBackend | str | None = None,
) -> list[list[float]]:
    """Fetch embeddings via local Ollama (embeddinggemma QAT Q8 by default).

    Raises EmbeddingError after retries instead of returning empty vectors.
    """
    if not texts:
        return []

    prepared = [_truncate_input(t, _MAX_INPUT_CHARS) for t in texts]
    resolved_backend, model, embeddings_url, headers = _resolve_embed_settings(
        model_id=model_id, url=url, backend=backend
    )
    logger.debug(
        "Embedding via %s model=%s url=%s n=%s batch_size=%s",
        resolved_backend,
        model,
        embeddings_url,
        len(prepared),
        _BATCH_SIZE,
    )

    out: list[list[float]] = []
    async with httpx.AsyncClient(trust_env=False) as client:
        for i in range(0, len(prepared), _BATCH_SIZE):
            chunk = prepared[i : i + _BATCH_SIZE]
            payload: dict = {
                "model": model,
                "input": chunk,
            }
            vectors = await _post_embeddings_once(
                client,
                embeddings_url=embeddings_url,
                headers=headers,
                payload=payload,
                resolved_backend=resolved_backend,
            )
            if len(vectors) != len(chunk):
                vectors = []
                for t in chunk:
                    one_payload: dict = {
                        "model": model,
                        "input": [t],
                    }
                    one = await _post_embeddings_once(
                        client,
                        embeddings_url=embeddings_url,
                        headers=headers,
                        payload=one_payload,
                        resolved_backend=resolved_backend,
                    )
                    if not one or not one[0]:
                        raise EmbeddingError(
                            "embedding backend returned an empty vector"
                        )
                    vectors.append(one[0])
                if len(vectors) != len(chunk):
                    raise EmbeddingError(
                        f"embedding count mismatch: got {len(vectors)} want {len(chunk)}"
                    )
            out.extend(vectors)
    return out


def _index_dimension(record: Any) -> int | None:
    options = record.get("options")
    if not isinstance(options, Mapping):
        return None
    config = options.get("indexConfig")
    if not isinstance(config, Mapping):
        return None
    raw = config.get("vector.dimensions") or config.get("`vector.dimensions`")
    try:
        return int(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


async def fetch_vector_indexes(
    driver: AsyncDriver, *, expected_dimension: int | None = None
) -> dict[str, list[str]]:
    """Find ONLINE vector indexes; relationship ANN uses evidence only."""
    indexes: dict[str, list[str]] = {"nodes": [], "relationships": []}
    relationship_candidates: dict[str, list[str]] = {}
    async with driver.session() as session:
        res = await session.run("SHOW VECTOR INDEXES")
        records = [r async for r in res]
        for r in records:
            idx_name = str(r["name"])
            entity_type = r.get("entityType", "").upper()
            state = str(r.get("state", "")).upper()
            if state != "ONLINE":
                logger.warning("Skipping vector index %s with state=%s", idx_name, state)
                continue
            dimension = _index_dimension(r)
            if (
                expected_dimension is not None
                and dimension is not None
                and dimension != expected_dimension
            ):
                logger.warning(
                    "Skipping vector index %s with dimension=%s (expected %s)",
                    idx_name,
                    dimension,
                    expected_dimension,
                )
                continue
            if entity_type == "NODE":
                indexes["nodes"].append(idx_name)
            elif entity_type == "RELATIONSHIP":
                properties = [str(value) for value in (r.get("properties") or [])]
                rel_types = [str(value) for value in (r.get("labelsOrTypes") or [])]
                required_properties = {"evidence_embedding", "run_id"}
                if not required_properties.issubset(properties) or len(rel_types) != 1:
                    logger.warning(
                        "Skipping relationship vector index %s with types=%s properties=%s",
                        idx_name,
                        rel_types,
                        properties,
                    )
                    continue
                relationship_candidates.setdefault(rel_types[0], []).append(idx_name)
            else:
                logger.error(
                    "Skipping vector index %s with unknown entityType=%r",
                    idx_name,
                    entity_type,
                )
    for rel_type, names in sorted(relationship_candidates.items()):
        ranked = sorted(
            set(names),
            key=lambda name: (not name.startswith("rel_ev_v1_"), name),
        )
        indexes["relationships"].append(ranked[0])
        if len(ranked) > 1:
            logger.warning(
                "Multiple evidence vector indexes for %s; using %s and ignoring %s",
                rel_type,
                ranked[0],
                ranked[1:],
            )
    return indexes

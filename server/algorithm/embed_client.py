"""Embedding HTTP client + Neo4j vector-index discovery for V6."""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

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
            data_sorted = sorted(data, key=lambda item: item.get("index", 0))
            return [item["embedding"] for item in data_sorted]
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


async def fetch_vector_indexes(driver: AsyncDriver) -> dict[str, list[str]]:
    """Dynamically find all node and relationship vector indexes in the database."""
    indexes: dict[str, list[str]] = {"nodes": [], "relationships": []}
    async with driver.session() as session:
        res = await session.run("SHOW VECTOR INDEXES")
        records = [r async for r in res]
        for r in records:
            idx_name = r["name"]
            entity_type = r.get("entityType", "").upper()
            if entity_type == "NODE":
                indexes["nodes"].append(idx_name)
            elif entity_type == "RELATIONSHIP":
                indexes["relationships"].append(idx_name)
            else:
                if "edge" in idx_name.lower():
                    indexes["relationships"].append(idx_name)
                else:
                    indexes["nodes"].append(idx_name)
    return indexes

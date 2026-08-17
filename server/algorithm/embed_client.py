"""Embedding HTTP client + Neo4j vector-index discovery for V6."""

from __future__ import annotations

import asyncio
import logging
import os

import httpx
from neo4j import AsyncDriver

from server.utils.constants import EmbeddingBackend

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Embedding HTTP/backend failed after retries."""


OPENROUTER_DEFAULT_MODEL = "nvidia/nemotron-3-embed-1b:free"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
TEI_DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"
TEI_DEFAULT_BASE_URL = "http://localhost:7998/v1"
_MAX_RETRIES = 5
_RETRY_BASE_SEC = 2.0
_BATCH_SIZE = 32
_MAX_INPUT_CHARS = 8192


def _ensure_env_loaded() -> None:
    if os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM__PROFILES__OTHER__API_KEY"):
        return
    from pathlib import Path

    from dotenv import load_dotenv

    root = Path(__file__).resolve().parents[2]
    for name in (".env.ragas-testing", ".env"):
        path = root / name
        if path.exists():
            load_dotenv(path, override=False)


def _resolve_embed_settings(
    model_id: str | None = None,
    url: str | None = None,
    backend: EmbeddingBackend | str | None = None,
) -> tuple[EmbeddingBackend, str, str, dict[str, str]]:
    """Resolve backend, model, embeddings URL, and request headers.

    Defaults match embed.py (OpenRouter nemotron). TEI only if caller passes it.
    """
    resolved_backend = EmbeddingBackend(backend or EmbeddingBackend.OPENROUTER)

    if resolved_backend == EmbeddingBackend.TEI:
        model = model_id or TEI_DEFAULT_MODEL
        base = (
            url or os.environ.get("EMBEDDING_URL") or TEI_DEFAULT_BASE_URL
        ).rstrip("/")
        headers = {"Content-Type": "application/json"}
        tei_key = os.environ.get("EMBEDDING_API_KEY")
        if tei_key:
            headers["Authorization"] = f"Bearer {tei_key}"
    else:
        _ensure_env_loaded()
        model = model_id or OPENROUTER_DEFAULT_MODEL
        base = (url or OPENROUTER_DEFAULT_BASE_URL).rstrip("/")
        key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get(
            "LLM__PROFILES__OTHER__API_KEY"
        )
        if not key:
            raise RuntimeError(
                "OpenRouter API key not found "
                "(OPENROUTER_API_KEY or LLM__PROFILES__OTHER__API_KEY)"
            )
        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        }

    embeddings_url = base if base.endswith("/embeddings") else f"{base}/embeddings"
    return resolved_backend, model, embeddings_url, headers


def _truncate_for_tei(text: str, max_chars: int) -> str:
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
            data = response.json()["data"]
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
    """Fetch embeddings (OpenRouter by default, or TEI if backend=tei).

    Raises EmbeddingError after retries instead of returning empty vectors.
    """
    if not texts:
        return []

    prepared = [_truncate_for_tei(t, _MAX_INPUT_CHARS) for t in texts]
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
            payload: dict = {"model": model, "input": chunk}
            if resolved_backend == EmbeddingBackend.OPENROUTER:
                payload["encoding_format"] = "float"
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
                    one_payload: dict = {"model": model, "input": [t]}
                    if resolved_backend == EmbeddingBackend.OPENROUTER:
                        one_payload["encoding_format"] = "float"
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

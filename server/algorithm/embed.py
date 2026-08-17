"""S1 helpers: cached text embeddings via embed_client."""

from __future__ import annotations

import logging

from server.algorithm.embed_client import EmbeddingError, get_embeddings_batch
from server.utils.constants import EmbeddingBackend

logger = logging.getLogger(__name__)

_V6_EMBED_BACKEND = EmbeddingBackend.OPENROUTER
_V6_EMBED_MODEL = "nvidia/nemotron-3-embed-1b:free"


async def embed_texts(
    texts: list[str],
    cache: dict[str, list[float]],
) -> list[list[float]]:
    out: list[list[float] | None] = [None] * len(texts)
    missing_idx: list[int] = []
    missing_texts: list[str] = []
    for i, t in enumerate(texts):
        key = t.strip()
        if key in cache:
            out[i] = cache[key]
        else:
            missing_idx.append(i)
            missing_texts.append(t)

    if missing_texts:
        vectors = await get_embeddings_batch(
            missing_texts,
            backend=_V6_EMBED_BACKEND,
            model_id=_V6_EMBED_MODEL,
        )
        if len(vectors) != len(missing_texts):
            raise EmbeddingError(
                f"embedding count mismatch: got {len(vectors)} "
                f"want {len(missing_texts)}"
            )
        for i, vec in zip(missing_idx, vectors):
            if not vec:
                raise EmbeddingError("embedding backend returned an empty vector")
            out[i] = vec
            cache[texts[i].strip()] = vec

    missing = [i for i, v in enumerate(out) if not v]
    if missing:
        raise EmbeddingError(
            f"missing embeddings for {len(missing)} text(s)"
        )
    filled: list[list[float]] = []
    for v in out:
        if not v:
            raise EmbeddingError("missing embeddings")
        filled.append(v)
    return filled

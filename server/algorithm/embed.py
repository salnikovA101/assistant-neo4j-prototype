"""S1 helpers: cached text embeddings via embed_client."""

from __future__ import annotations

import logging

from server.algorithm.embed_client import get_embeddings_batch
from server.utils.constants import EmbeddingBackend

logger = logging.getLogger(__name__)

_EMBED_BACKEND = EmbeddingBackend.OPENROUTER
_EMBED_MODEL = "nvidia/nemotron-3-embed-1b:free"


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
            backend=_EMBED_BACKEND,
            model_id=_EMBED_MODEL,
        )
        if len(vectors) != len(missing_texts):
            vectors = []
            for t in missing_texts:
                one = await get_embeddings_batch(
                    [t],
                    backend=_EMBED_BACKEND,
                    model_id=_EMBED_MODEL,
                )
                vectors.append(one[0] if one else [])
        for i, vec in zip(missing_idx, vectors):
            out[i] = vec
            if vec:
                cache[texts[i].strip()] = vec

    return [v or [] for v in out]

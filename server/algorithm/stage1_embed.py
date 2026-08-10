"""S1: embed subquestions into run dict (no SLM decompose)."""

from __future__ import annotations

import logging

from server.algorithm.embed import embed_texts
from server.algorithm.models import SubQuestion

logger = logging.getLogger(__name__)


async def embed_subquestions(
    subquestions: list[SubQuestion],
    embed_cache: dict[str, list[float]],
    *,
    query_fallback: str = "",
) -> dict[str, list[float]]:
    sqs = list(subquestions)
    if not sqs and query_fallback.strip():
        sqs = [SubQuestion(id="sq1", text=query_fallback.strip())]

    texts = [s.text for s in sqs]
    vectors = await embed_texts(texts, embed_cache)
    out: dict[str, list[float]] = {}
    for sq, vec in zip(sqs, vectors):
        if not vec:
            logger.warning("Empty embedding for sq %s", sq.id)
        out[sq.id] = list(vec or [])
    return out

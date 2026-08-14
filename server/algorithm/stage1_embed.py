"""S1: embed subquestions into run dict (no SLM decompose)."""

from __future__ import annotations

import logging

from server.algorithm.embed import embed_texts
from server.algorithm.models import SubQuestion

logger = logging.getLogger(__name__)


async def embed_subquestions(
    subquestions: list[SubQuestion],
    embed_cache: dict[str, list[float]],
) -> dict[str, list[float]]:
    texts = [s.text for s in subquestions]
    vectors = await embed_texts(texts, embed_cache)
    out: dict[str, list[float]] = {}
    for sq, vec in zip(subquestions, vectors):
        if not vec:
            logger.warning("Empty embedding for sq %s", sq.id)
        out[sq.id] = list(vec or [])
    return out

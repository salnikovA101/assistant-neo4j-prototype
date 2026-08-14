"""S2b: Ettin CE rerank — per-sq ANN top-pool → keep for S3 anchors."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from server.algorithm.models import EdgeRecord, SubQuestion
from server.algorithm.params import Params

logger = logging.getLogger(__name__)


def _doc_text(edge: EdgeRecord) -> str:
    ev = (edge.evidence or "").strip()
    if ev:
        return ev
    return f"{edge.start_ref()} {edge.rel_type} {edge.end_ref()}".strip()


def _pool_by_sim(hits: dict[str, EdgeRecord], pool: int) -> list[EdgeRecord]:
    ranked = sorted(hits.values(), key=lambda e: float(e.sim), reverse=True)
    return ranked[: max(0, pool)]


async def _score_texts(
    client: httpx.AsyncClient,
    url: str,
    query: str,
    texts: list[str],
    *,
    batch_size: int,
    timeout_s: float,
    raw_scores: bool = True,
) -> list[float]:
    """Return score per input text index (not sorted)."""
    scores = [0.0] * len(texts)
    if not texts:
        return scores
    base = url.rstrip("/")
    endpoint = f"{base}/rerank"
    bs = max(1, int(batch_size))
    for start in range(0, len(texts), bs):
        chunk = texts[start : start + bs]
        payload: dict[str, Any] = {
            "query": query,
            "texts": chunk,
            "raw_scores": raw_scores,
        }
        resp = await client.post(endpoint, json=payload, timeout=timeout_s)
        resp.raise_for_status()
        items = resp.json()
        if not isinstance(items, list):
            raise ValueError(f"unexpected rerank response type: {type(items)}")
        for item in items:
            if not isinstance(item, dict):
                continue
            rel = int(item.get("index", 0))
            abs_i = start + rel
            if 0 <= abs_i < len(scores):
                scores[abs_i] = float(item.get("score", 0.0))
    return scores


def _keep_top(pool: list[EdgeRecord], keep: int) -> dict[str, EdgeRecord]:
    # Already ordered by (rerank_score, sim) descending.
    out: dict[str, EdgeRecord] = {}
    for e in pool[: max(0, keep)]:
        out[e.edge_key] = e
    return out


async def rerank_ann_by_sq(
    sqs: list[SubQuestion],
    ann_by_sq: dict[str, dict[str, EdgeRecord]],
    params: Params,
) -> tuple[
    dict[str, dict[str, EdgeRecord]],
    dict[str, list[str]],
    dict[str, list[str]],
    dict[str, float],
]:
    """
    Per sq: take top ``L_raw_max`` by cosine (ANN already capped), CE-rerank
    vs sq.text, keep ``L``.

    Returns:
      truncated hits per sq,
      ann_keys (pool keys per sq),
      rerank_keys (kept keys per sq),
      ann_edge_sims (union pool: max cosine per edge_key, pre-CE).
    """
    pool_n = max(1, int(params.L_raw_max))
    keep_n = max(1, int(params.L))
    out: dict[str, dict[str, EdgeRecord]] = {}
    ann_keys: dict[str, list[str]] = {}
    rerank_keys: dict[str, list[str]] = {}
    ann_edge_sims: dict[str, float] = {}

    def _note_pool(pool: list[EdgeRecord]) -> None:
        for e in pool:
            sim = float(e.sim)
            prev = ann_edge_sims.get(e.edge_key)
            if prev is None or sim > prev:
                ann_edge_sims[e.edge_key] = sim

    if not params.rerank_enabled:
        for sq in sqs:
            hits = ann_by_sq.get(sq.id) or {}
            pool = _pool_by_sim(hits, pool_n)
            _note_pool(pool)
            ann_keys[sq.id] = [e.edge_key for e in pool]
            kept = _keep_top(pool, keep_n)
            rerank_keys[sq.id] = list(kept.keys())
            out[sq.id] = kept
            logger.info(
                "V6 S2b sq=%s rerank=off pool=%s keep=%s",
                sq.id,
                len(pool),
                len(kept),
            )
        return out, ann_keys, rerank_keys, ann_edge_sims

    timeout = float(params.rerank_timeout_s)
    batch_size = int(params.rerank_batch_size)
    url = (params.rerank_url or "http://127.0.0.1:7997").strip()

    async with httpx.AsyncClient() as client:
        for sq in sqs:
            hits = ann_by_sq.get(sq.id) or {}
            pool = _pool_by_sim(hits, pool_n)
            _note_pool(pool)
            ann_keys[sq.id] = [e.edge_key for e in pool]
            claim = (sq.text or "").strip()
            query = (
                f"Which evidence supports this claim?\nClaim: {claim}"
                if claim
                else ""
            )
            if not pool or not query:
                kept = _keep_top(pool, keep_n)
                rerank_keys[sq.id] = list(kept.keys())
                out[sq.id] = kept
                continue

            texts = [_doc_text(e) for e in pool]
            try:
                scores = await _score_texts(
                    client,
                    url,
                    query,
                    texts,
                    batch_size=batch_size,
                    timeout_s=timeout,
                    raw_scores=True,
                )
                for e, sc in zip(pool, scores):
                    e.rerank_score = float(sc)
                ordered = sorted(
                    pool,
                    key=lambda e: (float(e.rerank_score), float(e.sim)),
                    reverse=True,
                )
                kept = _keep_top(ordered, keep_n)
                logger.info(
                    "V6 S2b sq=%s pool=%s keep=%s top_ce=%.3f",
                    sq.id,
                    len(pool),
                    len(kept),
                    float(ordered[0].rerank_score) if ordered else 0.0,
                )
            except Exception as e:
                logger.warning(
                    "V6 S2b sq=%s rerank failed (%s); fallback top-%s by sim",
                    sq.id,
                    e,
                    keep_n,
                )
                kept = _keep_top(pool, keep_n)
            rerank_keys[sq.id] = list(kept.keys())
            out[sq.id] = kept

    return out, ann_keys, rerank_keys, ann_edge_sims

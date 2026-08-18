"""S2: per-subquestion relationship ANN (score = cosine from vector index)."""

from __future__ import annotations

import asyncio
import logging
import time

from neo4j import AsyncDriver

from server.algorithm.cypher.edges import fetch_edge_properties, query_relationship_ann
from server.algorithm.edge_keys import compute_edge_key
from server.algorithm.embed import embed_texts
from server.algorithm.embed_client import EmbeddingError, fetch_vector_indexes
from server.algorithm.models import EdgeRecord, SubQuestion
from server.algorithm.params import Params

logger = logging.getLogger(__name__)


class AnnError(RuntimeError):
    """Relationship ANN cannot run (no indexes, or every query failed)."""

_REL_INDEX_CACHE: list[str] | None = None
_REL_INDEX_CACHE_TS: float = 0.0
_REL_INDEX_TTL_SEC = 300.0


async def _get_rel_indexes(driver: AsyncDriver) -> list[str]:
    global _REL_INDEX_CACHE, _REL_INDEX_CACHE_TS
    now = time.monotonic()
    if _REL_INDEX_CACHE is not None and (now - _REL_INDEX_CACHE_TS) < _REL_INDEX_TTL_SEC:
        return _REL_INDEX_CACHE
    indexes = await fetch_vector_indexes(driver)
    rel_indexes = list(indexes.get("relationships") or [])
    _REL_INDEX_CACHE = rel_indexes
    _REL_INDEX_CACHE_TS = now
    return rel_indexes


def _upsert_hit(
    raw: dict[str, EdgeRecord],
    *,
    rid: str,
    rel_type: str,
    start_id: str,
    end_id: str,
    start_name: str,
    end_name: str,
    start_label: str,
    end_label: str,
    chunk_id: str,
    evidence: str,
    score: float,
) -> None:
    edge_key = compute_edge_key(start_name, rel_type, end_name, chunk_id, evidence)
    existing = raw.get(edge_key)
    if existing is None:
        raw[edge_key] = EdgeRecord(
            edge_key=edge_key,
            element_id=rid,
            rel_type=rel_type,
            start_id=start_id,
            end_id=end_id,
            start_name=start_name,
            end_name=end_name,
            start_label=start_label or "",
            end_label=end_label or "",
            sim=float(score),
            chunk_id=chunk_id or "",
            evidence=evidence or "",
            source="ann",
        )
        return
    if float(score) > existing.sim:
        existing.sim = float(score)
        existing.element_id = rid
        existing.start_id = start_id
        existing.end_id = end_id
        if start_label:
            existing.start_label = start_label
        if end_label:
            existing.end_label = end_label
    if evidence and not existing.evidence:
        existing.evidence = evidence
    if chunk_id and not existing.chunk_id:
        existing.chunk_id = chunk_id
    if start_label and not existing.start_label:
        existing.start_label = start_label
    if end_label and not existing.end_label:
        existing.end_label = end_label


async def edge_ann_search(
    driver: AsyncDriver,
    search_texts: list[str],
    params: Params,
    embedding_cache: dict[str, list[float]],
) -> dict[str, EdgeRecord]:
    if not search_texts:
        return {}
    ann_texts = list(search_texts[: max(1, params.max_ann_texts)])
    vectors = await embed_texts(ann_texts, embedding_cache)
    rel_indexes = await _get_rel_indexes(driver)
    if not rel_indexes:
        raise AnnError("No relationship vector indexes found")

    raw: dict[str, EdgeRecord] = {}
    sem = asyncio.Semaphore(max(1, params.ann_concurrency))

    async def _query_one(emb: list[float], index_name: str):
        async with sem:
            try:
                hits = await query_relationship_ann(
                    driver,
                    index_name,
                    emb,
                    top_k=params.L,
                    run_id=(params.run_id or "").strip(),
                )
                return hits, None
            except Exception as e:
                return [], e

    tasks = []
    for text, emb in zip(ann_texts, vectors):
        if not emb:
            raise EmbeddingError("ANN skipped: empty embedding vector")
        for index_name in rel_indexes:
            tasks.append(_query_one(emb, index_name))

    if not tasks:
        raise EmbeddingError("ANN skipped: no embeddings to query")

    results = await asyncio.gather(*tasks)
    n_err = sum(1 for _, err in results if err is not None)
    if n_err == len(results):
        first_err = next(err for _, err in results if err is not None)
        raise AnnError(f"all ANN queries failed: {first_err}")
    for hits, err in results:
        if err is not None:
            logger.error("ANN query failed: %s", err)
            continue
        for h in hits:
            rid = h.get("rid")
            if not rid:
                continue
            _upsert_hit(
                raw,
                rid=rid,
                rel_type=h.get("rel_type") or "",
                start_id=h.get("start_id") or "",
                end_id=h.get("end_id") or "",
                start_name=h.get("start_name") or "",
                end_name=h.get("end_name") or "",
                start_label=h.get("start_label") or "",
                end_label=h.get("end_label") or "",
                chunk_id=h.get("chunk_id") or "",
                evidence=h.get("evidence") or "",
                score=float(h.get("score") or 0.0),
            )

    n_merged = len(raw)
    if len(raw) > params.L_raw_max:
        ranked = sorted(raw.values(), key=lambda h: h.sim, reverse=True)
        raw = {h.edge_key: h for h in ranked[: params.L_raw_max]}
    rid = (params.run_id or "").strip()
    logger.info(
        "V6 S2 ANN indexes=%s per_index_L=%s merged_unique=%s after_L_raw_max=%s run_id=%s",
        len(rel_indexes),
        params.L,
        n_merged,
        len(raw),
        rid or "*",
    )

    prop_ids = [h.element_id for h in raw.values() if h.element_id]
    if prop_ids:
        props = await fetch_edge_properties(driver, prop_ids)
        by_id = {p["rid"]: p for p in props}
        remapped: dict[str, EdgeRecord] = {}
        for hit in list(raw.values()):
            p = by_id.get(hit.element_id)
            if p:
                if p.get("evidence") and not hit.evidence:
                    hit.evidence = p["evidence"]
                if p.get("chunk_id") and not hit.chunk_id:
                    hit.chunk_id = p["chunk_id"]
                if p.get("source_file"):
                    hit.source_file = p["source_file"] or ""
                if p.get("confidence") is not None:
                    hit.confidence = float(p.get("confidence") or hit.confidence)
                if p.get("start_label") and not hit.start_label:
                    hit.start_label = p["start_label"] or ""
                if p.get("end_label") and not hit.end_label:
                    hit.end_label = p["end_label"] or ""
                if p.get("start_name"):
                    hit.start_name = p["start_name"] or hit.start_name
                if p.get("end_name"):
                    hit.end_name = p["end_name"] or hit.end_name
                new_key = compute_edge_key(
                    hit.start_name,
                    hit.rel_type,
                    hit.end_name,
                    hit.chunk_id,
                    hit.evidence,
                )
                hit.edge_key = new_key
            prev = remapped.get(hit.edge_key)
            if prev is None or hit.sim > prev.sim:
                remapped[hit.edge_key] = hit
        raw = remapped

    return raw


async def ann_for_subquestions(
    driver: AsyncDriver,
    sqs: list[SubQuestion],
    sq_embeddings: dict[str, list[float]],
    embed_cache: dict[str, list[float]],
    params: Params,
) -> dict[str, dict[str, EdgeRecord]]:
    out: dict[str, dict[str, EdgeRecord]] = {}
    for sq in sqs:
        emb = sq_embeddings.get(sq.id) or []
        if emb and sq.text.strip() not in embed_cache:
            embed_cache[sq.text.strip()] = emb
        hits = await edge_ann_search(driver, [sq.text], params, embed_cache)
        out[sq.id] = hits
        logger.info("V6 S2 sq=%s hits=%s", sq.id, len(hits))
    return out

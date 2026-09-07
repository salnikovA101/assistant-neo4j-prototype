"""S2: per-subquestion relationship ANN (score = cosine from vector index)."""

from __future__ import annotations

import asyncio
import logging

from neo4j import AsyncDriver

from server.algorithm.cypher.edges import fetch_edge_properties, query_relationship_ann
from server.algorithm.edge_keys import compute_edge_key
from server.algorithm.embed import embed_texts
from server.algorithm.embed_client import EmbeddingError, fetch_vector_indexes
from server.algorithm.models import EdgeRecord, SubQuestion, normalize_labels, parse_confidence
from server.algorithm.params import Params

logger = logging.getLogger(__name__)


class AnnError(RuntimeError):
    """Relationship ANN cannot run (no indexes, or every query failed)."""

def _upsert_hit(
    raw: dict[str, EdgeRecord],
    *,
    rid: str,
    rel_type: str,
    start_id: str,
    end_id: str,
    start_name: str,
    end_name: str,
    start_labels: list[str],
    end_labels: list[str],
    chunk_id: str,
    evidence: str,
    score: float,
    run_id: str,
) -> None:
    start_labels = normalize_labels(start_labels)
    end_labels = normalize_labels(end_labels)
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
            start_labels=start_labels,
            end_labels=end_labels,
            sim=float(score),
            chunk_id=chunk_id or "",
            evidence=evidence or "",
            source="ann",
            run_id=run_id,
        )
        return
    if float(score) > existing.sim:
        existing.sim = float(score)
        existing.element_id = rid
        existing.start_id = start_id
        existing.end_id = end_id
        if start_labels:
            existing.start_labels = start_labels
            existing.start_label = " ".join(start_labels)
        if end_labels:
            existing.end_labels = end_labels
            existing.end_label = " ".join(end_labels)
    if evidence and not existing.evidence:
        existing.evidence = evidence
    if chunk_id and not existing.chunk_id:
        existing.chunk_id = chunk_id
    if start_labels and not existing.start_labels:
        existing.start_labels = start_labels
        existing.start_label = " ".join(start_labels)
    if end_labels and not existing.end_labels:
        existing.end_labels = end_labels
        existing.end_label = " ".join(end_labels)
    if run_id and not existing.run_id:
        existing.run_id = run_id


async def edge_ann_search(
    driver: AsyncDriver,
    search_texts: list[str],
    params: Params,
    embedding_cache: dict[str, list[float]],
) -> dict[str, EdgeRecord]:
    if not search_texts:
        return {}
    run_id = (params.run_id or "").strip()
    if not run_id:
        raise AnnError("run_id is required for relationship ANN")
    ann_texts = list(search_texts[: max(1, params.max_ann_texts)])
    vectors = await embed_texts(ann_texts, embedding_cache)
    first_dimension = len(next((vector for vector in vectors if vector), []))
    indexes = await fetch_vector_indexes(
        driver, expected_dimension=first_dimension or None
    )
    rel_indexes = list(indexes.get("relationships") or [])
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
                    run_id=run_id,
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
    if n_err:
        first_err = next(err for _, err in results if err is not None)
        raise AnnError(f"{n_err}/{len(results)} ANN queries failed: {first_err}")
    for hits, err in results:
        if err is not None:
            continue
        for h in hits:
            rid = h.get("rid")
            if not rid:
                continue
            hit_run_id = str(h.get("run_id") or "").strip()
            if hit_run_id != run_id:
                raise AnnError(
                    f"ANN returned edge from run_id={hit_run_id!r}, expected {run_id!r}"
                )
            _upsert_hit(
                raw,
                rid=rid,
                rel_type=h.get("rel_type") or "",
                start_id=h.get("start_id") or "",
                end_id=h.get("end_id") or "",
                start_name=h.get("start_name") or "",
                end_name=h.get("end_name") or "",
                start_labels=h.get("start_labels") or [],
                end_labels=h.get("end_labels") or [],
                chunk_id=h.get("chunk_id") or "",
                evidence=h.get("evidence") or "",
                score=float(h.get("score") or 0.0),
                run_id=hit_run_id,
            )

    n_merged = len(raw)
    if len(raw) > params.L_raw_max:
        ranked = sorted(raw.values(), key=lambda h: h.sim, reverse=True)
        raw = {h.edge_key: h for h in ranked[: params.L_raw_max]}
    logger.info(
        "S2 ANN indexes=%s per_index_L=%s merged_unique=%s after_L_raw_max=%s run_id=%s",
        len(rel_indexes),
        params.L,
        n_merged,
        len(raw),
        run_id,
    )

    prop_ids = [h.element_id for h in raw.values() if h.element_id]
    if prop_ids:
        props = await fetch_edge_properties(driver, prop_ids, run_id=run_id)
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
                    hit.confidence = parse_confidence(p.get("confidence"))
                if p.get("start_labels") and not hit.start_labels:
                    hit.start_labels = normalize_labels(p["start_labels"])
                    hit.start_label = " ".join(hit.start_labels)
                if p.get("end_labels") and not hit.end_labels:
                    hit.end_labels = normalize_labels(p["end_labels"])
                    hit.end_label = " ".join(hit.end_labels)
                if p.get("run_id"):
                    if str(p["run_id"]) != run_id:
                        raise AnnError("edge hydration crossed the requested run_id")
                    hit.run_id = str(p["run_id"])
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
        logger.info("S2 sq=%s hits=%s", sq.id, len(hits))
    return out

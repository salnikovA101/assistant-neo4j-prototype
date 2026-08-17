"""S3: N+1 graphs — L ANN/CE anchors + induced bridges ranked by cosine."""

from __future__ import annotations

import logging

from neo4j import AsyncDriver

from server.algorithm.cypher.edges import fetch_induced_bridges_by_sim
from server.algorithm.edge_keys import compute_edge_key
from server.algorithm.models import CandidateGraph, EdgeRecord, SubQuestion
from server.algorithm.params import Params

logger = logging.getLogger(__name__)


def _rebuild_node_index(graph: CandidateGraph) -> None:
    graph.node_to_edges = {}
    for key, edge in graph.edges.items():
        for nid in (edge.start_id, edge.end_id):
            graph.node_to_edges.setdefault(nid, []).append(key)


def transition_allowed(e: EdgeRecord, o: EdgeRecord) -> bool:
    """
    Line-graph step e → o: walk continuation via exactly one shared endpoint.

    Star walks (co-outgoing / co-incoming) are allowed; hop-DP may traverse a
    hub, then reshape collapses entry/rays/exit into SPINE+FANS. Evidence
    anti-dupe is enforced inside a single unit during S4, not here.
    """
    a = {e.start_id, e.end_id}
    b = {o.start_id, o.end_id}
    if len(a) < 2 or len(b) < 2:
        return False
    shared = a & b
    return len(shared) == 1


def _build_transitions(graph: CandidateGraph, branch_cap: int) -> None:
    graph.transition_adj = {}
    for key, edge in graph.edges.items():
        neighbors: list[tuple[str, float]] = []
        seen: set[str] = set()
        for nid in (edge.start_id, edge.end_id):
            for other in graph.node_to_edges.get(nid, []):
                if other == key or other in seen:
                    continue
                seen.add(other)
                o = graph.edges[other]
                if not transition_allowed(edge, o):
                    continue
                neighbors.append((other, float(o.sim)))
        neighbors.sort(key=lambda x: x[1], reverse=True)
        graph.transition_adj[key] = [k for k, _ in neighbors[:branch_cap]]


def _finalize_graph(source: str, edges: dict[str, EdgeRecord], branch_cap: int) -> CandidateGraph:
    g = CandidateGraph(source_graph=source, edges=dict(edges))
    _rebuild_node_index(g)
    _build_transitions(g, branch_cap)
    return g


def _row_to_bridge(b: dict) -> EdgeRecord:
    evidence = b.get("evidence") or ""
    chunk_id = b.get("chunk_id") or ""
    start_name = b.get("start_name") or ""
    end_name = b.get("end_name") or ""
    rel_type = b.get("rel_type") or ""
    key = compute_edge_key(start_name, rel_type, end_name, chunk_id, evidence)
    return EdgeRecord(
        edge_key=key,
        element_id=b.get("rid") or "",
        rel_type=rel_type,
        start_id=b.get("start_id") or "",
        end_id=b.get("end_id") or "",
        start_name=start_name,
        end_name=end_name,
        start_label=b.get("start_label") or "",
        end_label=b.get("end_label") or "",
        sim=float(b.get("score") or 0.0),
        chunk_id=chunk_id,
        evidence=evidence,
        source_file=b.get("source_file") or "",
        source="bridge",
        confidence=float(b.get("confidence") or 1.0),
    )


async def _add_bridges(
    driver: AsyncDriver,
    anchors: dict[str, EdgeRecord],
    sq_vec: list[float],
    params: Params,
) -> dict[str, EdgeRecord]:
    if not anchors or not sq_vec:
        return {}
    nodes: set[str] = set()
    for e in anchors.values():
        if e.start_id:
            nodes.add(e.start_id)
        if e.end_id:
            nodes.add(e.end_id)
    exclude = [e.element_id for e in anchors.values() if e.element_id]
    rows = await fetch_induced_bridges_by_sim(
        driver,
        nodes,
        sq_vec,
        exclude_ids=exclude,
        limit=params.bridge_top,
    )
    out: dict[str, EdgeRecord] = {}
    for b in rows:
        e = _row_to_bridge(b)
        if e.edge_key in anchors:
            continue
        prev = out.get(e.edge_key)
        if prev is None or e.sim > prev.sim:
            out[e.edge_key] = e
    return out


def _anchor_sort_key(edge: EdgeRecord) -> tuple[float, float]:
    """Prefer CE when present; missing CE sorts below any scored logit."""
    ce = float("-inf") if edge.rerank_score is None else float(edge.rerank_score)
    return (ce, float(edge.sim))


async def build_sq_graph(
    driver: AsyncDriver,
    sq: SubQuestion,
    ann_hits: dict[str, EdgeRecord],
    sq_vec: list[float],
    params: Params,
) -> CandidateGraph:
    # Anchors: prefer CE rerank_score, then cosine; cap at L
    ranked = sorted(
        ann_hits.values(),
        key=_anchor_sort_key,
        reverse=True,
    )
    anchors = {e.edge_key: e for e in ranked[: params.L]}
    bridges = await _add_bridges(driver, anchors, sq_vec, params)
    merged = dict(anchors)
    merged.update(bridges)
    logger.info(
        "V6 S3 sq=%s anchors=%s bridges=%s total=%s",
        sq.id,
        len(anchors),
        len(bridges),
        len(merged),
    )
    return _finalize_graph(sq.id, merged, params.branch_cap)


async def build_global_graph(
    driver: AsyncDriver,
    per_sq: dict[str, CandidateGraph],
    sq_embeddings: dict[str, list[float]],
    params: Params,
) -> CandidateGraph:
    # Union with max(sim) already stored from per-sq ANN/bridges
    union: dict[str, EdgeRecord] = {}
    for g in per_sq.values():
        for k, e in g.edges.items():
            prev = union.get(k)
            if prev is None or e.sim > prev.sim:
                union[k] = EdgeRecord(
                    edge_key=e.edge_key,
                    element_id=e.element_id,
                    rel_type=e.rel_type,
                    start_id=e.start_id,
                    end_id=e.end_id,
                    start_name=e.start_name,
                    end_name=e.end_name,
                    start_label=e.start_label,
                    end_label=e.end_label,
                    sim=e.sim,
                    rerank_score=e.rerank_score,
                    chunk_id=e.chunk_id,
                    evidence=e.evidence,
                    source_file=e.source_file,
                    source=e.source,
                    confidence=e.confidence,
                )
            elif (
                e.rerank_score is not None
                and (
                    prev.rerank_score is None
                    or float(e.rerank_score) > float(prev.rerank_score)
                )
            ):
                prev.rerank_score = float(e.rerank_score)

    ranked = sorted(
        union.values(),
        key=_anchor_sort_key,
        reverse=True,
    )
    anchors = {e.edge_key: e for e in ranked[: params.L]}

    # Pooled sq vector for DB-side bridge ranking (no Python cosine)
    sq_vecs = [v for v in sq_embeddings.values() if v]
    if sq_vecs:
        dim = len(sq_vecs[0])
        pooled = [0.0] * dim
        for v in sq_vecs:
            for i, x in enumerate(v):
                pooled[i] += x
        n = len(sq_vecs)
        pooled = [x / n for x in pooled]
    else:
        pooled = []

    bridges = await _add_bridges(driver, anchors, pooled, params)
    merged = dict(anchors)
    for k, e in bridges.items():
        if k not in merged:
            merged[k] = e
    logger.info(
        "V6 S3 global anchors=%s bridges=%s total=%s",
        len(anchors),
        len(bridges),
        len(merged),
    )
    return _finalize_graph("global", merged, params.branch_cap)


async def build_all_graphs(
    driver: AsyncDriver,
    sqs: list[SubQuestion],
    ann_by_sq: dict[str, dict[str, EdgeRecord]],
    sq_embeddings: dict[str, list[float]],
    params: Params,
) -> dict[str, CandidateGraph]:
    graphs: dict[str, CandidateGraph] = {}
    for sq in sqs:
        g = await build_sq_graph(
            driver,
            sq,
            ann_by_sq.get(sq.id) or {},
            sq_embeddings.get(sq.id) or [],
            params,
        )
        graphs[sq.id] = g
    graphs["global"] = await build_global_graph(driver, graphs, sq_embeddings, params)
    return graphs

"""Retrieval orchestration: S1→S3 once → S4 carousel → S5 spine dedup."""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncDriver

from server.algorithm.embed_client import EmbeddingError
from server.algorithm.graph_cache import build_s3_bundle, load_s3_bundle_graphs
from server.algorithm.models import CandidateGraph, Chain, SessionState, SubQuestion
from server.algorithm.params import Params, merge_params
from server.algorithm.stage1_embed import embed_subquestions
from server.algorithm.stage2_ann import AnnError, ann_for_subquestions
from server.algorithm.stage2b_rerank import RerankError, rerank_ann_by_sq
from server.algorithm.stage3_graphs import build_all_graphs
from server.algorithm.stage4_hop_dp import CarouselState, continue_s4_carousel
from server.algorithm.stage5_select import hydrate_chains, prepare_s5_batch

logger = logging.getLogger(__name__)


def _normalize_subquestions(
    raw: list[dict[str, Any]] | list[SubQuestion] | None,
    query: str = "",
) -> list[SubQuestion]:
    del query
    if not raw:
        return []
    out: list[SubQuestion] = []
    for i, item in enumerate(raw):
        if isinstance(item, SubQuestion):
            out.append(item)
            continue
        text = str(item.get("text") or item.get("query") or "").strip()
        sid = str(item.get("id") or item.get("sq_id") or f"sq{i+1}")
        if text:
            out.append(SubQuestion(id=sid, text=text))
    return out


def _s3_edge_keys(graphs: dict) -> dict[str, set[str]]:
    return {src: set(g.edges.keys()) for src, g in graphs.items()}


def _s3_edge_sims_union(graphs: dict) -> dict[str, float]:
    """Union of all S3 graph edges → max sim per edge_key."""
    out: dict[str, float] = {}
    for g in graphs.values():
        for key, edge in g.edges.items():
            sim = float(edge.sim)
            prev = out.get(key)
            if prev is None or sim > prev:
                out[key] = sim
    return out


def _graphs_for_sqs(
    graphs: dict[str, CandidateGraph],
    sqs: list[SubQuestion],
) -> dict[str, CandidateGraph]:
    """Keep cached S3 graphs whose ids match current subquestions; ignore id `global`."""
    out: dict[str, CandidateGraph] = {}
    for sq in sqs:
        if sq.id == "global":
            continue
        g = graphs.get(sq.id)
        if g is not None:
            out[sq.id] = g
    return out


def label_chains_for_assistant(chains: list[Chain]) -> list[Chain]:
    """Assign a1.. ids and UNIT text; keep carousel order."""
    out: list[Chain] = []
    for i, c in enumerate(chains, start=1):
        c.chain_id = f"a{i}"
        if c.edges or c.fans:
            c.text = c.format_unit(c.chain_id)
        out.append(c)
    return out


def _empty_run_result(
    params: Params,
    *,
    error: str,
    subquestions: list[SubQuestion] | None = None,
    error_detail: str = "",
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "accepted": [],
        "accepted_all": [],
        "subquestions": [s.to_dict() for s in (subquestions or [])],
        "trace": {},
        "s3_keys": {},
        "s3_keys_union": [],
        "s3_edge_sims": {},
        "ann_keys": {},
        "ann_keys_union": [],
        "ann_edge_sims": {},
        "rerank_keys": {},
        "rerank_keys_union": [],
        "effort": params.effort,
        "from_graph_cache": False,
        "error": error,
    }
    if error_detail:
        out["error_detail"] = error_detail
    return out


async def run(
    driver: AsyncDriver,
    *,
    subquestions: list[dict[str, Any]] | list[SubQuestion] | None = None,
    query: str = "",
    effort: str = "medium",
    params: Params | None = None,
    s3_bundle: dict[str, Any] | None = None,
    emit_s3_bundle: bool = False,
    cache_qid: str = "",
    carousel_state: dict[str, Any] | CarouselState | None = None,
    prior_signatures: list[str] | set[str] | None = None,
    manual_round: bool = False,
    budget_override: int | None = None,
) -> dict[str, Any]:
    """
    Run the retrieval pipeline (wired by ask_subgraph).

    S1–S3 once (embed → ANN → CE → per-sq graphs); S4 carousel until the
    path budget; S5 drops duplicate spines and returns the pool.

    s3_bundle: optional cached S3 payload (graphs + ann/rerank keys); skips S1–S3.
    emit_s3_bundle: include serializable S3 bundle in result for graph-cache writes.
    """
    p = (params or merge_params()).with_effort(effort)
    state = SessionState(subquestions=_normalize_subquestions(subquestions, query))

    if not state.subquestions:
        return _empty_run_result(p, error="no_subquestions")
    if not (p.run_id or "").strip():
        return _empty_run_result(
            p,
            error="run_id_required",
            subquestions=state.subquestions,
            error_detail="Unscoped Neo4j retrieval is disabled",
        )

    from_graph_cache = False
    s3_bundle_out: dict[str, Any] | None = None
    graphs: dict[str, CandidateGraph]
    ann_keys: dict[str, list[str]]
    rerank_keys: dict[str, list[str]]
    ann_edge_sims: dict[str, float]

    if s3_bundle is not None:
        graphs = load_s3_bundle_graphs(s3_bundle, branch_cap=p.branch_cap)
        for graph in graphs.values():
            for edge in graph.edges.values():
                if edge.run_id and edge.run_id != p.run_id:
                    return _empty_run_result(
                        p,
                        error="cache_run_id_mismatch",
                        subquestions=state.subquestions,
                        error_detail="Cached graph belongs to another run_id",
                    )
                if not edge.run_id:
                    edge.run_id = p.run_id
        ann_keys = {
            str(k): list(v) for k, v in (s3_bundle.get("ann_keys") or {}).items()
        }
        rerank_keys = {
            str(k): list(v) for k, v in (s3_bundle.get("rerank_keys") or {}).items()
        }
        ann_edge_sims = {
            str(k): float(v)
            for k, v in (s3_bundle.get("ann_edge_sims") or {}).items()
        }
        from_graph_cache = True
        logger.info(
            "S1–S3 from graph-cache (%s graphs, %s edges)",
            len(graphs),
            sum(len(g.edges) for g in graphs.values()),
        )
    else:
        try:
            sq_emb = await embed_subquestions(state.subquestions, state.embed_cache)
            if any(not v for v in sq_emb.values()):
                return _empty_run_result(
                    p,
                    error="embed_failed",
                    subquestions=state.subquestions,
                    error_detail="empty embedding for one or more subquestions",
                )
            ann_by_sq = await ann_for_subquestions(
                driver, state.subquestions, sq_emb, state.embed_cache, p
            )
            ann_by_sq, ann_keys_map, rerank_keys_map, ann_edge_sims = await rerank_ann_by_sq(
                state.subquestions, ann_by_sq, p
            )
            ann_keys = {k: list(v) for k, v in ann_keys_map.items()}
            rerank_keys = {k: list(v) for k, v in rerank_keys_map.items()}

            graphs = await build_all_graphs(
                driver, state.subquestions, ann_by_sq, sq_emb, p
            )
            if emit_s3_bundle:
                s3_bundle_out = build_s3_bundle(
                    qid=cache_qid or "",
                    question=query,
                    params=p,
                    sqs=state.subquestions,
                    graphs=graphs,
                    ann_keys=ann_keys,
                    rerank_keys=rerank_keys,
                    ann_edge_sims=ann_edge_sims,
                )
        except EmbeddingError as e:
            logger.exception("S1 embed failed")
            return _empty_run_result(
                p,
                error="embed_failed",
                subquestions=state.subquestions,
                error_detail=str(e),
            )
        except AnnError as e:
            logger.exception("S2 ANN failed")
            return _empty_run_result(
                p,
                error="ann_failed",
                subquestions=state.subquestions,
                error_detail=str(e),
            )
        except RerankError as e:
            logger.exception("S2b rerank failed")
            return _empty_run_result(
                p,
                error="rerank_failed",
                subquestions=state.subquestions,
                error_detail=str(e),
            )

    s3_keys_sets = _s3_edge_keys(graphs)
    s3_edge_sims = _s3_edge_sims_union(graphs)

    graphs_s4 = _graphs_for_sqs(graphs, state.subquestions)
    s3_keys_s4 = _s3_edge_keys(graphs_s4)
    budget = (
        max(0, int(budget_override))
        if budget_override is not None
        else (len(graphs_s4) if manual_round else p.effort_max_paths())
    )

    carousel = continue_s4_carousel(
        graphs_s4,
        params=p,
        budget=budget,
        state=carousel_state,
        one_per_graph=manual_round,
        prior_signatures=set(prior_signatures or []),
    )
    s4_pool = carousel.chains
    batch = prepare_s5_batch(s4_pool)
    stop_reason = ""

    if not batch:
        logger.info("S5 empty batch; nothing to accept")
        stop_reason = "empty_batch"
    else:
        await hydrate_chains(driver, batch, run_id=p.run_id)
        state.accepted = label_chains_for_assistant(list(batch))
        logger.info(
            "S5 accept %s chains (carousel order, budget=%s)",
            len(state.accepted),
            budget,
        )

    trace: dict[str, Any] = {
        "s3_sizes": {k: len(v) for k, v in s3_keys_s4.items()},
        "s4_pool": len(s4_pool),
        "s4_mined": carousel.mined,
        "s4_duplicate_mined": carousel.duplicate_mined,
        "batch": [c.to_dict() for c in batch],
        "accepted": [c.to_dict() for c in state.accepted],
        "s3_keys": {k: sorted(v) for k, v in s3_keys_s4.items()},
        **({"stop_reason": stop_reason} if stop_reason else {}),
    }

    union_s3: set[str] = set()
    for ks in s3_keys_sets.values():
        union_s3 |= ks

    ann_union: set[str] = set()
    for ks in ann_keys.values():
        ann_union.update(ks)
    rerank_union: set[str] = set()
    for ks in rerank_keys.values():
        rerank_union.update(ks)

    accepted_dicts = [c.to_dict() for c in state.accepted]

    out: dict[str, Any] = {
        "accepted": accepted_dicts,
        "accepted_all": accepted_dicts,
        "subquestions": [s.to_dict() for s in state.subquestions],
        "trace": trace,
        "s3_keys": {k: sorted(v) for k, v in s3_keys_sets.items()},
        "s3_keys_union": sorted(union_s3),
        "s3_edge_sims": dict(s3_edge_sims),
        "ann_keys": {k: list(v) for k, v in ann_keys.items()},
        "ann_keys_union": sorted(ann_union),
        "ann_edge_sims": dict(ann_edge_sims),
        "rerank_keys": {k: list(v) for k, v in rerank_keys.items()},
        "rerank_keys_union": sorted(rerank_union),
        "effort": p.effort,
        "params": p.to_dict(),
        "from_graph_cache": from_graph_cache,
        "carousel_state": carousel.state.to_dict(),
        "manual_round": manual_round,
    }
    if s3_bundle_out is not None:
        out["s3_bundle"] = s3_bundle_out
    return out

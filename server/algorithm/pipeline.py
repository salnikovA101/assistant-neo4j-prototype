"""Pipeline orchestration: S1→S3 once → [S4→S5→S6 effort loop] → S7."""

from __future__ import annotations

import logging
from typing import Any

from neo4j import AsyncDriver

from server.algorithm.graph_cache import build_s3_bundle, load_s3_bundle_graphs
from server.algorithm.models import CandidateGraph, SessionState, SubQuestion
from server.algorithm.params import Params, merge_params
from server.algorithm.pheromone import apply_s7
from server.algorithm.stage1_embed import embed_subquestions
from server.algorithm.stage2_ann import ann_for_subquestions
from server.algorithm.stage2b_rerank import rerank_ann_by_sq
from server.algorithm.stage3_graphs import build_all_graphs
from server.algorithm.stage4_hop_dp import run_s4_all_graphs
from server.algorithm.stage5_select import hydrate_chains, prepare_s5_batch
from server.algorithm.stage6_judge import judge_flat, postprocess_and_update_p

logger = logging.getLogger(__name__)


def _normalize_subquestions(
    raw: list[dict[str, Any]] | list[SubQuestion] | None,
    query: str = "",
) -> list[SubQuestion]:
    if not raw:
        if query.strip():
            return [SubQuestion(id="sq1", text=query.strip())]
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
    if not out and query.strip():
        out = [SubQuestion(id="sq1", text=query.strip())]
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


def _graph_edge_sims(graph) -> dict[str, float]:
    if graph is None:
        return {}
    return {k: float(e.sim) for k, e in graph.edges.items()}


def _graphs_for_open(
    graphs: dict[str, CandidateGraph],
    open_sqs: list[SubQuestion],
) -> dict[str, CandidateGraph]:
    """Slice cached S3 graphs to open sq ids + fixed global."""
    out: dict[str, CandidateGraph] = {}
    for sq in open_sqs:
        g = graphs.get(sq.id)
        if g is not None:
            out[sq.id] = g
    global_g = graphs.get("global")
    if global_g is not None:
        out["global"] = global_g
    return out


async def run(
    driver: AsyncDriver,
    *,
    subquestions: list[dict[str, Any]] | list[SubQuestion] | None = None,
    query: str = "",
    effort: str = "medium",
    params: Params | None = None,
    tau_store: dict[str, float] | None = None,
    s3_bundle: dict[str, Any] | None = None,
    emit_s3_bundle: bool = False,
    cache_qid: str = "",
) -> dict[str, Any]:
    """
    Run the retrieval pipeline.

    S1–S3 once (embed → ANN → CE → N+1 graphs); effort loop is S4→S5→S6 only;
    S7 τ update after the loop. tau_store: optional dict (mutated at S7);
    if None, ephemeral for this call only.

    s3_bundle: optional cached S3 payload (graphs + ann/rerank keys); skips S1–S3.
    emit_s3_bundle: include serializable S3 bundle in result for graph-cache writes.
    """
    p = (params or merge_params()).with_effort(effort)
    state = SessionState(subquestions=_normalize_subquestions(subquestions, query))
    tau = tau_store if tau_store is not None else {}

    if not state.subquestions:
        return {
            "accepted": [],
            "subquestions": [],
            "iters": [],
            "s3_keys": {},
            "s3_keys_union": [],
            "s3_edge_sims": {},
            "s3_global_edge_sims": {},
            "ann_keys": {},
            "ann_keys_union": [],
            "ann_edge_sims": {},
            "rerank_keys": {},
            "rerank_keys_union": [],
            "tau": dict(tau),
            "effort": p.effort,
            "from_graph_cache": False,
            "error": "no_subquestions",
        }

    from_graph_cache = False
    s3_bundle_out: dict[str, Any] | None = None

    if s3_bundle is not None:
        # Replay: S4–S6 only
        graphs = load_s3_bundle_graphs(s3_bundle, branch_cap=p.branch_cap)
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
        # S1 — once
        sq_emb = await embed_subquestions(
            state.subquestions, state.embed_cache, query_fallback=query
        )

        # S2 — once (all sq)
        ann_by_sq = await ann_for_subquestions(
            driver, state.subquestions, sq_emb, state.embed_cache, p
        )
        # S2b — once
        ann_by_sq, ann_keys_map, rerank_keys_map, ann_edge_sims = await rerank_ann_by_sq(
            state.subquestions, ann_by_sq, p
        )
        ann_keys = {k: list(v) for k, v in ann_keys_map.items()}
        rerank_keys = {k: list(v) for k, v in rerank_keys_map.items()}

        # S3 — once (N+1 graphs)
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

    s3_keys_sets = _s3_edge_keys(graphs)
    s3_edge_sims = _s3_edge_sims_union(graphs)
    s3_global_edge_sims = _graph_edge_sims(graphs.get("global"))

    max_iters = p.effort_max_iters()
    iter_traces: list[dict[str, Any]] = []

    # Effort loop: S4 → S5 → S6 only
    for it in range(1, max_iters + 1):
        open_sqs = state.open_sqs()
        if not open_sqs:
            logger.info("early-stop all sq closed at iter %s", it)
            break

        graphs_it = _graphs_for_open(graphs, open_sqs)
        iter_s3 = _s3_edge_keys(graphs_it)

        # S4
        s4_pool = run_s4_all_graphs(
            graphs_it, tau_store=tau, p_store=state.p_edges, params=p
        )
        # S5
        batch = prepare_s5_batch(
            s4_pool,
            seen_spine_seqs=state.seen_spine_seqs,
            params=p,
            graph_ids=list(graphs_it.keys()),
        )
        if not batch:
            logger.info("iter %s empty judge batch; stopping", it)
            iter_traces.append(
                {
                    "iter": it,
                    "open_sq": [s.to_dict() for s in open_sqs],
                    "s3_sizes": {k: len(v) for k, v in iter_s3.items()},
                    "s4_pool": len(s4_pool),
                    "batch": [],
                    "sq_closed": {},
                    "chain_needed": {},
                    "needed_ids": [],
                    "rejected_ids": [],
                    "newly_accepted": [],
                    "accepted_so_far": [c.to_dict() for c in state.accepted],
                    "remaining_open": [s.to_dict() for s in state.open_sqs()],
                    "judge_raw": "",
                    "s3_keys": {k: sorted(v) for k, v in iter_s3.items()},
                    "stop_reason": "empty_batch",
                }
            )
            break

        await hydrate_chains(driver, batch)

        # S6
        if p.skip_judge:
            sq_closed = {s.id: False for s in open_sqs}
            chain_needed = {c.chain_id: True for c in batch}
            raw_judge = "SKIP_JUDGE"
            judge_ok = True
            logger.info(
                "iter %s skip_judge: accept all %s chains, sq stay open",
                it,
                len(batch),
            )
        else:
            sq_closed, chain_needed, raw_judge, judge_ok = await judge_flat(
                open_sqs, batch, p
            )

        newly = []
        rejected = []
        if not judge_ok:
            # API/parse fail: no accept, no p_reject, no evidence burn; sq stay open
            logger.warning(
                "iter %s judge fail; skipping postprocess (batch=%s)",
                it,
                len(batch),
            )
        else:
            newly, rejected = postprocess_and_update_p(
                open_sqs=open_sqs,
                all_sqs=state.subquestions,
                chains=batch,
                sq_closed=sq_closed,
                chain_needed=chain_needed,
                accepted=state.accepted,
                used_edges=state.used_edges,
                seen_spine_seqs=state.seen_spine_seqs,
                p_store=state.p_edges,
                params=p,
            )

        iter_traces.append(
            {
                "iter": it,
                "open_sq": [s.to_dict() for s in open_sqs],
                "s3_sizes": {k: len(v) for k, v in iter_s3.items()},
                "s4_pool": len(s4_pool),
                "batch": [c.to_dict() for c in batch],
                "sq_closed": dict(sq_closed),
                "chain_needed": dict(chain_needed),
                "needed_ids": [c.chain_id for c in batch if chain_needed.get(c.chain_id)],
                "rejected_ids": [c.chain_id for c in rejected],
                "newly_accepted": [c.to_dict() for c in newly],
                "accepted_so_far": [c.to_dict() for c in state.accepted],
                "remaining_open": [s.to_dict() for s in state.open_sqs()],
                "judge_raw": (raw_judge or "")[:2000],
                "judge_ok": bool(judge_ok),
                "s3_keys": {k: sorted(v) for k, v in iter_s3.items()},
                "skip_judge": bool(p.skip_judge),
                **({"stop_reason": "judge_fail"} if not judge_ok else {}),
            }
        )

        if not state.open_sqs():
            break

    # S7
    accepted_keys: list[str] = []
    for c in state.accepted:
        accepted_keys.extend(c.all_edge_keys())
    apply_s7(tau, accepted_keys, p)

    union_s3: set[str] = set()
    for ks in s3_keys_sets.values():
        union_s3 |= ks

    ann_union: set[str] = set()
    for ks in ann_keys.values():
        ann_union.update(ks)
    rerank_union: set[str] = set()
    for ks in rerank_keys.values():
        rerank_union.update(ks)

    out: dict[str, Any] = {
        "accepted": [c.to_dict() for c in state.accepted],
        "subquestions": [s.to_dict() for s in state.subquestions],
        "iters": iter_traces,
        "s3_keys": {k: sorted(v) for k, v in s3_keys_sets.items()},
        "s3_keys_union": sorted(union_s3),
        "s3_edge_sims": dict(s3_edge_sims),
        "s3_global_edge_sims": dict(s3_global_edge_sims),
        "ann_keys": {k: list(v) for k, v in ann_keys.items()},
        "ann_keys_union": sorted(ann_union),
        "ann_edge_sims": dict(ann_edge_sims),
        "rerank_keys": {k: list(v) for k, v in rerank_keys.items()},
        "rerank_keys_union": sorted(rerank_union),
        "tau": dict(tau),
        "effort": p.effort,
        "params": p.to_dict(),
        "p_edges": dict(state.p_edges),
        "from_graph_cache": from_graph_cache,
    }
    if s3_bundle_out is not None:
        out["s3_bundle"] = s3_bundle_out
    return out

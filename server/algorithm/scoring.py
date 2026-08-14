"""Edge weight helpers: PCST-style rank prize/cost for S4 path DP."""

from __future__ import annotations

from collections.abc import Mapping

from server.algorithm.models import EdgeRecord
from server.algorithm.params import Params


def anchor_prize(
    sim: float,
    edge_key: str,
    *,
    p_store: Mapping[str, float],
    params: Params,
) -> float:
    """Ranking weight = sim · p, clipped to [s_floor, 1]."""
    p = float(p_store.get(edge_key, 1.0))
    prize = max(0.0, float(sim)) * p
    if prize <= 0.0:
        return float(params.s_floor)
    return float(min(1.0, max(params.s_floor, prize)))


def edge_prize_weight(
    edge: EdgeRecord,
    *,
    p_store: Mapping[str, float],
    params: Params,
) -> float:
    """Rank weight: relevance · p (CE if set, else sim)."""
    rel = float(edge.rerank_score) if float(edge.rerank_score) > 0.0 else float(edge.sim)
    return anchor_prize(
        rel,
        edge.edge_key,
        p_store=p_store,
        params=params,
    )


def rank_contribs(
    edges: Mapping[str, EdgeRecord],
    *,
    p_store: Mapping[str, float],
    params: Params,
) -> tuple[dict[str, float], set[str]]:
    """
    Rank economics for a profitable tour (G-Retriever order, not a tree):
    trust the ranker only on order.

    Non-bridge edges ranked by (rerank_score|sim)·p_rank, r=1 best.
    p≤0 marks an arc already collected: it keeps its original rank (p_rank=1)
    so leftover prize_top seats are *not* given to demoted ANN, and the arc
    itself pays struct cost (TOARP: prize at most once, no promotion).
    - r ≤ prize_top and not collected: prize = prize_rank_max·(K−r+1)/K · p;
    - r > prize_top (demoted ANN): cost = c0·(1+γ·x²)·(2−p),
      x = (r−K)/(N−K) — flat mid-range, steep garbage tail;
    - source="bridge" (structural, no honest ANN rank): flat
      bridge_struct_cost·(2−p).

    Returns (contrib per edge, prize key set).
    """
    ranked = [
        e
        for e in edges.values()
        if (e.source or "").strip().lower() != "bridge"
    ]
    collected = {
        e.edge_key
        for e in edges.values()
        if float(p_store.get(e.edge_key, 1.0)) <= 0.0
    }
    rank_p: dict[str, float] = dict(p_store)
    for ek in collected:
        rank_p[ek] = 1.0
    ranked.sort(
        key=lambda e: edge_prize_weight(e, p_store=rank_p, params=params),
        reverse=True,
    )
    k = max(0, int(params.prize_top))
    n = len(ranked)
    span = max(1, n - k)
    p_max = float(params.prize_rank_max)
    c0 = float(params.bridge_cost_c0)
    gamma = float(params.bridge_cost_gamma)
    c_struct = float(params.bridge_struct_cost)

    contrib: dict[str, float] = {}
    prize_keys: set[str] = set()
    for r, e in enumerate(ranked, start=1):
        if e.edge_key in collected:
            contrib[e.edge_key] = -float(c_struct * 2.0)
            continue
        p = float(p_store.get(e.edge_key, 1.0))
        p = max(0.0, min(1.0, p))
        if r <= k:
            prize_keys.add(e.edge_key)
            frac = (k - r + 1) / k
            floor = max(0.0, min(1.0, float(params.prize_floor)))
            prize = p_max * (floor + (1.0 - floor) * frac) * p
            contrib[e.edge_key] = float(min(1.0, max(params.s_floor, prize)))
        else:
            x = (r - k) / span
            contrib[e.edge_key] = -float(c0 * (1.0 + gamma * x * x) * (2.0 - p))
    for e in edges.values():
        if (e.source or "").strip().lower() == "bridge":
            p = float(p_store.get(e.edge_key, 1.0))
            p = max(0.0, min(1.0, p))
            contrib[e.edge_key] = -float(c_struct * (2.0 - p))
    return contrib, prize_keys

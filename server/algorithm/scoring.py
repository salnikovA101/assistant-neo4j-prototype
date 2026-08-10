"""Edge weight helpers: PCST-style rank prize/cost for S4 path DP."""

from __future__ import annotations

from collections.abc import Mapping

from server.algorithm.models import EdgeRecord
from server.algorithm.params import Params
from server.algorithm.pheromone import get_tau


def anchor_prize(
    sim: float,
    edge_key: str,
    *,
    tau_store: Mapping[str, float],
    p_store: Mapping[str, float],
    params: Params,
) -> float:
    """Ranking weight = sim · τ · p, clipped to [s_floor, 1]."""
    tau = get_tau(tau_store, edge_key)
    p = float(p_store.get(edge_key, 1.0))
    prize = max(0.0, float(sim)) * tau * p
    if prize <= 0.0:
        return float(params.s_floor)
    return float(min(1.0, max(params.s_floor, prize)))


def edge_prize_weight(
    edge: EdgeRecord,
    *,
    tau_store: Mapping[str, float],
    p_store: Mapping[str, float],
    params: Params,
) -> float:
    """Rank weight: relevance · τ · p (CE if set, else sim)."""
    rel = float(edge.rerank_score) if float(edge.rerank_score) > 0.0 else float(edge.sim)
    return anchor_prize(
        rel,
        edge.edge_key,
        tau_store=tau_store,
        p_store=p_store,
        params=params,
    )


def rank_contribs(
    edges: Mapping[str, EdgeRecord],
    *,
    tau_store: Mapping[str, float],
    p_store: Mapping[str, float],
    params: Params,
) -> tuple[dict[str, float], set[str]]:
    """
    PCST-style rank economics (G-Retriever): trust the ranker only on order.

    Non-bridge edges ranked by (rerank_score|sim)·τ·p, r=1 best:
    - r ≤ prize_top: prize = prize_rank_max·(K−r+1)/K, scaled by τ·p;
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
    ranked.sort(
        key=lambda e: edge_prize_weight(
            e, tau_store=tau_store, p_store=p_store, params=params
        ),
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
        p = float(p_store.get(e.edge_key, 1.0))
        p = max(0.0, min(1.0, p))
        if r <= k:
            prize_keys.add(e.edge_key)
            tau = get_tau(tau_store, e.edge_key)
            prize = p_max * (k - r + 1) / k * tau * p
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

"""Edge weight helpers: rank prize/cost for S4 path DP."""

from __future__ import annotations

from collections.abc import Mapping

from server.algorithm.models import EdgeRecord
from server.algorithm.params import Params


def ranking_relevance(edge: EdgeRecord) -> float:
    """CE logit if scored, else ANN cosine. Missing CE is None, not 0.0."""
    if edge.rerank_score is None:
        return float(edge.sim)
    return float(edge.rerank_score)


def edge_prize_weight(
    edge: EdgeRecord,
    *,
    p_store: Mapping[str, float],
    params: Params,
) -> float:
    """Sort key: raw CE (or cosine if CE was not run) · p. Not clipped to [0, 1]."""
    del params
    p = float(p_store.get(edge.edge_key, 1.0))
    return ranking_relevance(edge) * p


def rank_contribs(
    edges: Mapping[str, EdgeRecord],
    *,
    p_store: Mapping[str, float],
    params: Params,
) -> tuple[dict[str, float], set[str]]:
    """
    One ranked list of every edge on the graph (anchors and bridges).

    Sort by (CE|sim)·p from the shared overlay. p only moves order; prize
    and cost amounts come from rank, not from p again.
    - r ≤ prize_top: prize = prize_rank_max·(K−r+1)/K
    - r > prize_top: cost = prize_rank_max·x^s4_cost_power,
      x = (r−K)/(N−K); last rank pays prize_rank_max.
    """
    ranked = list(edges.values())
    ranked.sort(
        key=lambda e: edge_prize_weight(e, p_store=p_store, params=params),
        reverse=True,
    )
    k = max(0, int(params.prize_top))
    n = len(ranked)
    span = max(1, n - k)
    p_max = float(params.prize_rank_max)
    power = max(0.0, float(params.s4_cost_power))
    floor = float(params.s_floor)

    contrib: dict[str, float] = {}
    prize_keys: set[str] = set()
    if k <= 0:
        for r, e in enumerate(ranked, start=1):
            x = r / max(1, n)
            contrib[e.edge_key] = -float(p_max * (x**power))
        return contrib, prize_keys

    for r, e in enumerate(ranked, start=1):
        if r <= k:
            prize_keys.add(e.edge_key)
            frac = (k - r + 1) / k
            prize = p_max * frac
            contrib[e.edge_key] = float(min(p_max, max(floor, prize)))
        else:
            x = (r - k) / span
            contrib[e.edge_key] = -float(p_max * (x**power))
    return contrib, prize_keys

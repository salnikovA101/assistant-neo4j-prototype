"""Algorithm hyperparameters, grouped by pipeline stage."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any

# Docker Compose sets RERANK_URL=http://host.docker.internal:7997 for the app service.
_DEFAULT_RERANK_URL = "http://127.0.0.1:7997"


def _rerank_url_default() -> str:
    return (os.environ.get("RERANK_URL") or _DEFAULT_RERANK_URL).strip().rstrip("/")


@dataclass
class Params:
    # S2 ANN: per relationship vector index take L, merge all, then cut to L_raw_max.
    # L is also CE keep (S2b) and S3 anchor budget — one shared top-K.
    L: int = 100
    L_raw_max: int = 300
    ann_concurrency: int = 30
    max_ann_texts: int = 16

    # S2b Ettin CE: score full ANN pool (≤ L_raw_max), keep top L
    # Override URL via env RERANK_URL (compose → host.docker.internal:7997).
    rerank_enabled: bool = True
    rerank_url: str = _DEFAULT_RERANK_URL
    rerank_timeout_s: float = 120.0
    rerank_batch_size: int = 50

    # S3 N+1 graphs: anchors = top L (by CE|sim); bridges; line-graph branch cap
    bridge_top: int = 4000
    branch_cap: int = 20

    # S4: Team Arc Orienteering on the line-graph — k profitable tours / graph.
    # Rank-based prize/cost: rank edges by (rerank|sim)·p;
    # prize(r) = prize_rank_max·(K−r+1)/K for r≤prize_top;
    # demoted ANN cost(r) = bridge_cost_c0·(1+γ·x²), x=(r−K)/(N−K);
    # structural bridges (source="bridge") pay flat bridge_struct_cost.
    # After each tour, collected arcs get local p=0 so the next tour must
    # pick leftover prize (TOARP: prize at most once).
    prize_top: int = 80
    max_hops: int = 10
    min_path_len: int = 6
    s4_paths_per_graph: int = 3
    s4_min_prize_edges: int = 2
    # Prize collected on one graph is gone for the next (TOARP globally).
    s4_share_prize_across_graphs: bool = True
    prize_rank_max: float = 1.0
    # Floor as a fraction of prize_rank_max so mid-ranks in prize_top stay
    # worth collecting (0 = linear to ~0 at rank K; 0.25 ≈ bridge cost).
    prize_floor: float = 0.0
    bridge_cost_c0: float = 0.25
    bridge_cost_gamma: float = 0.3
    bridge_struct_cost: float = 0.30
    s_floor: float = 1e-6

    # S5: pick up to path budget (effort_max_paths)
    max_paths_low: int = 10
    max_paths_medium: int = 15
    max_paths_high: int = 20
    effort: str = "medium"
    # What the assistant actually sees: sort by S4 score, then cap.
    # Retrieval budget stays max_paths_*; emit is smaller: easy 5 / medium 10 / hard 15.
    # emit_top_k>0 overrides the effort-specific cap (sweep). 0 → use emit_top_k_*.
    # emit_score_frac still drops a score cliff inside that cap.
    emit_top_k_low: int = 5
    emit_top_k_medium: int = 10
    emit_top_k_high: int = 15
    emit_top_k: int = 0
    emit_score_frac: float = 0.25

    def effort_max_paths(self) -> int:
        """Hard cap on accepted units per question (low≈10, medium≈15, hard≈20)."""
        e = (self.effort or "medium").strip().lower()
        if e == "low":
            return max(0, int(self.max_paths_low))
        if e == "high":
            return max(0, int(self.max_paths_high))
        return max(0, int(self.max_paths_medium))

    def effort_emit_top_k(self) -> int:
        """How many score-sorted units the assistant sees (low=5, medium=10, hard=15)."""
        if int(self.emit_top_k or 0) > 0:
            return int(self.emit_top_k)
        e = (self.effort or "medium").strip().lower()
        if e == "low":
            return max(0, int(self.emit_top_k_low))
        if e == "high":
            return max(0, int(self.emit_top_k_high))
        return max(0, int(self.emit_top_k_medium))

    def with_effort(self, effort: str | None) -> Params:
        if not effort:
            return self
        data = asdict(self)
        data["effort"] = str(effort).strip().lower()
        return Params(**data)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_params() -> Params:
    return Params(rerank_url=_rerank_url_default())


def merge_params(overrides: dict[str, Any] | None = None) -> Params:
    base = default_params()
    if not overrides:
        return base
    allowed = {f.name for f in fields(Params)}
    data = asdict(base)
    for k, v in overrides.items():
        if k in allowed:
            data[k] = v
    return Params(**data)

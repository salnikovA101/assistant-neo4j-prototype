"""Algorithm hyperparameters, grouped by pipeline stage."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any

# Docker Compose sets RERANK_URL=http://reranker:7997 for the app service.
_DEFAULT_RERANK_URL = "http://127.0.0.1:7997"


def _rerank_url_default() -> str:
    return (os.environ.get("RERANK_URL") or _DEFAULT_RERANK_URL).strip().rstrip("/")


@dataclass
class Params:
    # One Neo4j relationship.run_id for in-index ANN + S3 bridges. Empty = all.
    run_id: str = ""

    # S2 ANN: per relationship vector index take L, merge all, then cut to L_raw_max.
    # L is also CE keep (S2b) and S3 anchor budget — one shared top-K.
    L: int = 100
    L_raw_max: int = 300
    ann_concurrency: int = 30
    max_ann_texts: int = 16

    # S2b Ettin CE: score full ANN pool (≤ L_raw_max), keep top L
    # Override URL via env RERANK_URL (compose → http://reranker:7997).
    rerank_enabled: bool = True
    rerank_url: str = _DEFAULT_RERANK_URL
    rerank_timeout_s: float = 120.0
    rerank_batch_size: int = 50

    # S3 per-sq graphs: anchors = top L (by CE|sim); bridges; line-graph branch cap
    bridge_top: int = 4000
    branch_cap: int = 20

    # S4 carousel: one tour / sq / round, shared p on UNION edge_keys.
    # Rank all edges on the graph by (CE|sim)·p; top prize_top get linear
    # prizes prize_rank_max·(K−r+1)/K; tail cost prize_rank_max·x^s4_cost_power,
    # x=(r−K)/(N−K). After a tour, p *= s4_p_decay on walk keys (0 → zero).
    prize_top: int = 50
    max_hops: int = 10
    min_path_len: int = 1
    s4_min_prize_edges: int = 2
    prize_rank_max: float = 1.0
    s4_p_decay: float = 0.7
    s4_cost_power: float = 1.5
    s_floor: float = 1e-6

    # How many units to mine and show (low / medium / high).
    max_paths_low: int = 5
    max_paths_medium: int = 10
    max_paths_high: int = 15
    effort: str = "medium"

    def effort_max_paths(self) -> int:
        """Hard cap on accepted units per question (max_paths_* by effort)."""
        e = (self.effort or "medium").strip().lower()
        if e == "low":
            return max(0, int(self.max_paths_low))
        if e == "high":
            return max(0, int(self.max_paths_high))
        return max(0, int(self.max_paths_medium))

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

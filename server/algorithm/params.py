"""Algorithm hyperparameters (defaults), grouped by pipeline stage."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, fields
from typing import Any

# Legacy override keys → canonical field (L = ANN / CE keep / S3 anchors).
_PARAM_ALIASES: dict[str, str] = {
    "anchor_top": "L",
    "rerank_keep": "L",
}

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

    # S4: global best-path DP on line-graph (1 path / graph)
    # Rank-based prize/cost (G-Retriever PCST-style): rank edges by
    # (rerank|sim)·τ·p; prize(r) = prize_rank_max·(K−r+1)/K for r≤prize_top;
    # demoted ANN cost(r) = bridge_cost_c0·(1+γ·x²), x=(r−K)/(N−K);
    # structural bridges (source="bridge") pay flat bridge_struct_cost.
    prize_top: int = 25
    max_hops: int = 10
    min_path_len: int = 5
    prize_rank_max: float = 1.0
    bridge_cost_c0: float = 0.25
    bridge_cost_gamma: float = 0.3
    bridge_struct_cost: float = 0.30
    s_floor: float = 1e-6

    # S5 / effort K
    k_tool_low: int = 10
    k_tool_medium: int = 10
    k_tool_high: int = 10
    effort: str = "medium"

    # S6 session p + judge SLM
    p_accept: float = 0.90
    p_reject: float = 0.50
    p_floor: float = 0.10
    judge_temperature: float = 0.0
    judge_max_tokens: int = 8192
    # A/B: skip SLM judge — accept every batch chain, never close sq
    skip_judge: bool = True

    # S7 pheromone
    tau_rho: float = 0.9
    tau_delta: float = 0.08
    tau_max: float = 1.25

    def effort_max_iters(self) -> int:
        e = (self.effort or "medium").strip().lower()
        if e == "low":
            return 1
        if e == "high":
            return 3
        return 2

    def effort_k_tool(self) -> int:
        e = (self.effort or "medium").strip().lower()
        if e == "low":
            return int(self.k_tool_low)
        if e == "high":
            return int(self.k_tool_high)
        return int(self.k_tool_medium)

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
        key = _PARAM_ALIASES.get(k, k)
        if key in allowed:
            data[key] = v
    return Params(**data)

"""S4: carousel over sq graphs with a shared p overlay.

Each tour is a best-path DP: maximize sum of rank contribs over
L ∈ [min_path_len, max_hops], no edge/evidence revisit. After a tour,
walk keys get p *= s4_p_decay in a UNION store shared across graphs.

DP with path_so_far revisit ban is a practical optimum under no-revisit
(not color-coding exact). Sufficient for n≤300, L≤10.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from dataclasses import replace
from typing import Any

from server.algorithm.evidence import edge_evidence_key
from server.algorithm.models import CandidateGraph, Chain, EdgeRecord
from server.algorithm.params import Params
from server.algorithm.scoring import rank_contribs
from server.algorithm.unit_reshape import reshape_star_walk


@dataclass
class CarouselState:
    """Serializable continuation state for a deterministic S4 sequence."""

    p_store: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    accepted_signatures: list[str] = field(default_factory=list)
    rounds: int = 0

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> CarouselState:
        data = raw or {}
        return cls(
            p_store={str(k): float(v) for k, v in (data.get("p_store") or {}).items()},
            counts={str(k): int(v) for k, v in (data.get("counts") or {}).items()},
            accepted_signatures=[str(v) for v in (data.get("accepted_signatures") or [])],
            rounds=int(data.get("rounds") or 0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "p_store": dict(self.p_store),
            "counts": dict(self.counts),
            "accepted_signatures": list(self.accepted_signatures),
            "rounds": self.rounds,
        }


@dataclass
class CarouselResult:
    chains: list[Chain]
    state: CarouselState
    mined: int = 0
    duplicate_mined: int = 0


def _chain_signature(chain: Chain) -> str:
    seq = chain.spine_evidence_seq()
    if seq:
        return "\x1f".join(seq)
    return "\x1f".join(chain.all_edge_keys())


def _path_evidence_keys(graph: CandidateGraph, path: list[str]) -> set[str]:
    out: set[str] = set()
    for k in path:
        e = graph.edges.get(k)
        if not e:
            continue
        ek = edge_evidence_key(e)
        if ek:
            out.add(ek)
    return out


def _tag_s4_role(edge: EdgeRecord, prize_keys: set[str]) -> EdgeRecord:
    """Serialize S4 role: prize if in prize set, else bridge (tail rank)."""
    role = "prize" if edge.edge_key in prize_keys else "bridge"
    return replace(edge, source=role)


def _contribs_for_graph(
    graph: CandidateGraph,
    p_store: Mapping[str, float],
    params: Params,
) -> tuple[dict[str, float], set[str]]:
    return rank_contribs(graph.edges, p_store=p_store, params=params)


def _reconstruct(
    end: str,
    h: int,
    prev: dict[tuple[str, int], str | None],
) -> list[str]:
    path: list[str] = []
    cur: str | None = end
    layer = h
    while cur is not None and layer >= 1:
        path.append(cur)
        cur = prev.get((cur, layer))
        layer -= 1
    path.reverse()
    return path


def best_path_for_graph(
    graph: CandidateGraph,
    *,
    p_store: Mapping[str, float] | None = None,
    params: Params,
    id_prefix: str = "c",
    path_index: int = 1,
) -> Chain | None:
    """
    One profitable tour: start from every edge; maximize sum of rank contribs
    over L ∈ [min_path_len, max_hops]. All graph edges share one rank list.
    Returns one Chain (walk-ordered tour; spine+fans for viz) or None.
    """
    if not graph.edges:
        return None

    contrib, prize_keys = _contribs_for_graph(graph, p_store or {}, params)
    max_h = int(params.max_hops)
    min_h = max(1, min(int(params.min_path_len), max_h))
    min_prize = max(0, int(params.s4_min_prize_edges))

    # best_sum[(e,h)] = max sum contrib along path of length h ending at e
    best_sum: dict[tuple[str, int], float] = {}
    prev: dict[tuple[str, int], str | None] = {}

    for e in graph.edges:
        best_sum[(e, 1)] = float(contrib.get(e, 0.0))
        prev[(e, 1)] = None

    for h in range(2, max_h + 1):
        fronts = [e for (e, hh) in best_sum if hh == h - 1]
        for e in fronts:
            path_so_far = _reconstruct(e, h - 1, prev)
            path_evs = _path_evidence_keys(graph, path_so_far)
            for nxt in graph.transition_adj.get(e, []):
                if nxt in path_so_far:
                    continue
                nxt_edge = graph.edges.get(nxt)
                if nxt_edge is not None:
                    nxt_ev = edge_evidence_key(nxt_edge)
                    if nxt_ev and nxt_ev in path_evs:
                        continue
                cand = best_sum[(e, h - 1)] + float(contrib.get(nxt, 0.0))
                key = (nxt, h)
                if key not in best_sum or cand > best_sum[key]:
                    best_sum[key] = cand
                    prev[key] = e

    best_path: list[str] | None = None
    best_score = float("-inf")
    for h in range(min_h, max_h + 1):
        for (e, hh), sumc in best_sum.items():
            if hh != h:
                continue
            path = _reconstruct(e, h, prev)
            if not path or len(path) != h:
                continue
            if sumc <= 0.0:
                continue
            n_prize = sum(1 for k in path if k in prize_keys)
            if n_prize < min_prize:
                continue
            if sumc > best_score:
                best_path, best_score = path, float(sumc)

    if not best_path:
        return None

    raw_edges = [
        _tag_s4_role(graph.edges[k], prize_keys)
        for k in best_path
        if k in graph.edges
    ]
    spine, fans, hub_names = reshape_star_walk(raw_edges)
    return Chain(
        chain_id=f"{id_prefix}{path_index}",
        edge_keys=[e.edge_key for e in spine],
        score=float(best_score),
        source_graph=graph.source_graph,
        source_graphs=[graph.source_graph],
        edges=spine,
        fans=fans,
        fan_hub_names=hub_names,
        walk=raw_edges,
    )


def _decay_keys(p_store: dict[str, float], keys: list[str], decay: float) -> None:
    d = float(decay)
    for ek in keys:
        if d <= 0.0:
            p_store[ek] = 0.0
        else:
            p_store[ek] = float(p_store.get(ek, 1.0)) * d


def run_s4_carousel(
    graphs: dict[str, CandidateGraph],
    *,
    params: Params,
    budget: int,
) -> list[Chain]:
    """Round-robin tours across sq graphs until ``budget`` (first round full).

    Shared p starts at 1 on the UNION of all graph edge keys. After each
    accepted tour, walk keys are multiplied by ``s4_p_decay``.
    """
    return continue_s4_carousel(
        graphs,
        params=params,
        budget=budget,
    ).chains


def continue_s4_carousel(
    graphs: dict[str, CandidateGraph],
    *,
    params: Params,
    budget: int,
    state: CarouselState | Mapping[str, Any] | None = None,
    one_per_graph: bool = False,
    prior_signatures: set[str] | None = None,
) -> CarouselResult:
    """Continue S4 without resetting diversity state.

    Auto preserves the legacy first-round-full behavior. Manual mode mines at
    most one *new* UNIT per SQ; duplicate tours are decayed and audited but not
    returned.
    """
    cap = max(0, int(budget))
    persistent_call = bool(prior_signatures)
    if isinstance(state, CarouselState):
        persistent_call = persistent_call or bool(
            state.p_store or state.counts or state.accepted_signatures or state.rounds
        )
    elif state is not None:
        persistent_call = persistent_call or bool(state)
    # A checkpoint snapshot is immutable: continuation always works on a copy.
    current = (
        CarouselState.from_dict(state.to_dict())
        if isinstance(state, CarouselState)
        else CarouselState.from_dict(state)
    )
    if cap <= 0 or not graphs:
        return CarouselResult([], current)

    for src, graph in graphs.items():
        current.counts.setdefault(src, 0)
        for edge_key in graph.edges:
            current.p_store.setdefault(edge_key, 1.0)

    known = set(current.accepted_signatures)
    known.update(prior_signatures or set())
    pool: list[Chain] = []
    mined = 0
    duplicate_mined = 0
    decay = float(params.s4_p_decay)
    items = list(graphs.items())

    def mine_for_graph(src: str, graph: CandidateGraph, *, seek_novel: bool) -> Chain | None:
        nonlocal mined, duplicate_mined
        max_attempts = min(64, max(1, len(graph.edges))) if seek_novel else 1
        for _ in range(max_attempts):
            current.counts[src] = current.counts.get(src, 0) + 1
            chain = best_path_for_graph(
                graph,
                p_store=current.p_store,
                params=params,
                id_prefix=f"{src}_",
                path_index=current.counts[src],
            )
            if chain is None:
                return None
            mined += 1
            _decay_keys(current.p_store, chain.all_edge_keys(), decay)
            signature = _chain_signature(chain)
            if signature in known:
                duplicate_mined += 1
                if seek_novel:
                    continue
                return chain
            known.add(signature)
            current.accepted_signatures.append(signature)
            return chain
        return None

    if one_per_graph:
        for src, graph in items:
            chain = mine_for_graph(src, graph, seek_novel=True)
            if chain is not None:
                pool.append(chain)
        current.rounds += 1
        return CarouselResult(pool, current, mined=mined, duplicate_mined=duplicate_mined)

    def one_round(*, stop_at_cap: bool) -> int:
        before = len(pool)
        for src, graph in items:
            if stop_at_cap and len(pool) >= cap:
                break
            chain = mine_for_graph(src, graph, seek_novel=persistent_call)
            if chain is not None:
                pool.append(chain)
        current.rounds += 1
        return len(pool) - before

    one_round(stop_at_cap=False)
    while len(pool) < cap:
        if one_round(stop_at_cap=True) == 0:
            break
    return CarouselResult(pool, current, mined=mined, duplicate_mined=duplicate_mined)

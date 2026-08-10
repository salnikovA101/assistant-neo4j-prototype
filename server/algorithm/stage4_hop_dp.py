"""S4: global best-path DP on line-graph; prize/cost; reshape star → SPINE+FANS.

DP with path_so_far revisit ban is a practical optimum under no-revisit
(not color-coding exact). Sufficient for n≤300, L≤10.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from server.algorithm.evidence import edge_evidence_key
from server.algorithm.models import CandidateGraph, Chain, EdgeRecord
from server.algorithm.params import Params
from server.algorithm.scoring import rank_contribs
from server.algorithm.unit_reshape import reshape_star_walk


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
    """Serialize S4 role: prize if in prize set, else bridge (demoted ANN or glue)."""
    role = "prize" if edge.edge_key in prize_keys else "bridge"
    return replace(edge, source=role)


def _contribs_for_graph(
    graph: CandidateGraph,
    tau_store: Mapping[str, float],
    p_store: Mapping[str, float],
    params: Params,
) -> tuple[dict[str, float], set[str]]:
    return rank_contribs(graph.edges, tau_store=tau_store, p_store=p_store, params=params)


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
    tau_store: Mapping[str, float],
    p_store: Mapping[str, float],
    params: Params,
    id_prefix: str = "c",
) -> Chain | None:
    """
    Global DP: start from every edge; maximize sum of rank contribs over
    L ∈ [min_path_len, max_hops]. Prize on top prize_top ranked non-bridge
    edges; demoted ANN + structural bridges pay rank costs. Returns one
    Chain (SPINE+FANS) or None.
    """
    if not graph.edges:
        return None

    contrib, prize_keys = _contribs_for_graph(graph, tau_store, p_store, params)
    max_h = int(params.max_hops)
    min_h = max(1, min(int(params.min_path_len), max_h))

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
            # Maximize sum contrib only (bridge cost never zero → no free padding).
            if sumc > best_score:
                best_path, best_score = path, float(sumc)

    if not best_path:
        return None

    raw_edges = [graph.edges[k] for k in best_path if k in graph.edges]
    spine, fans, hub_names = reshape_star_walk(raw_edges)
    spine = [_tag_s4_role(e, prize_keys) for e in spine]
    fans = {hub: [_tag_s4_role(e, prize_keys) for e in flist] for hub, flist in fans.items()}
    return Chain(
        chain_id=f"{id_prefix}1",
        edge_keys=[e.edge_key for e in spine],
        score=float(best_score),
        source_graph=graph.source_graph,
        source_graphs=[graph.source_graph],
        edges=spine,
        fans=fans,
        fan_hub_names=hub_names,
    )


def hop_dp_paths(
    graph: CandidateGraph,
    *,
    tau_store: Mapping[str, float],
    p_store: Mapping[str, float],
    params: Params,
    id_prefix: str = "c",
) -> list[Chain]:
    """Compat wrapper: 0 or 1 best path per graph."""
    chain = best_path_for_graph(
        graph,
        tau_store=tau_store,
        p_store=p_store,
        params=params,
        id_prefix=id_prefix,
    )
    return [chain] if chain is not None else []


def run_s4_all_graphs(
    graphs: dict[str, CandidateGraph],
    *,
    tau_store: Mapping[str, float],
    p_store: Mapping[str, float],
    params: Params,
) -> list[Chain]:
    pool: list[Chain] = []
    for src, g in graphs.items():
        chain = best_path_for_graph(
            g,
            tau_store=tau_store,
            p_store=p_store,
            params=params,
            id_prefix=f"{src}_",
        )
        if chain is not None:
            pool.append(chain)
    min_len = max(1, int(params.min_path_len))
    if min_len > 1:
        # Spine may shrink after reshape; require raw path length via spine+fans
        pool = [c for c in pool if len(c.all_edge_keys()) >= min_len]
    return pool

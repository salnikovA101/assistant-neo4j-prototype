"""S4: k profitable tours per graph with local prize-once overlay."""

from __future__ import annotations

from server.algorithm.models import CandidateGraph, Chain, EdgeRecord
from server.algorithm.params import Params
from server.algorithm.pipeline import emit_cut_chains, rank_chains_for_emit
from server.algorithm.stage4_hop_dp import hop_dp_paths, run_s4_all_graphs, run_s4_fill_budget


def _linear_chain(prefix: str, n: int, sim0: float) -> dict[str, EdgeRecord]:
    edges: dict[str, EdgeRecord] = {}
    for i in range(n):
        k = f"{prefix}e{i}"
        edges[k] = EdgeRecord(
            edge_key=k,
            element_id=k,
            rel_type="R",
            start_id=f"{prefix}n{i}",
            end_id=f"{prefix}n{i + 1}",
            start_name=f"{prefix}n{i}",
            end_name=f"{prefix}n{i + 1}",
            sim=sim0 - i * 0.01,
            rerank_score=sim0 - i * 0.01,
            evidence=f"ev-{prefix}-{i}",
            source="ann",
        )
    return edges


def _adj_for_linears(*edge_maps: dict[str, EdgeRecord]) -> dict[str, list[str]]:
    adj: dict[str, list[str]] = {}
    for emap in edge_maps:
        keys = list(emap)
        for i, k in enumerate(keys):
            nxt: list[str] = []
            if i + 1 < len(keys):
                nxt.append(keys[i + 1])
            if i - 1 >= 0:
                nxt.append(keys[i - 1])
            adj[k] = nxt
    return adj


def _graph_two_chains() -> CandidateGraph:
    a = _linear_chain("a", 6, 1.0)
    b = _linear_chain("b", 6, 0.80)
    edges = {**a, **b}
    node_to: dict[str, list[str]] = {}
    for e in edges.values():
        node_to.setdefault(e.start_id, []).append(e.edge_key)
        node_to.setdefault(e.end_id, []).append(e.edge_key)
    return CandidateGraph(
        source_graph="g",
        edges=edges,
        node_to_edges=node_to,
        transition_adj=_adj_for_linears(a, b),
    )


def test_k_paths_cover_disjoint_prize_clusters() -> None:
    g = _graph_two_chains()
    params = Params(
        prize_top=25,
        max_hops=10,
        min_path_len=6,
        s4_paths_per_graph=2,
    )
    one = hop_dp_paths(g, p_store={}, params=params, id_prefix="g_")
    params_k1 = Params(
        prize_top=25,
        max_hops=10,
        min_path_len=6,
        s4_paths_per_graph=1,
    )
    only_one = hop_dp_paths(g, p_store={}, params=params_k1, id_prefix="g_")
    assert len(only_one) == 1
    assert len(one) == 2
    keys0 = set(one[0].all_edge_keys())
    keys1 = set(one[1].all_edge_keys())
    assert not keys0 & keys1
    prefixes = {k[0] for k in keys0} | {k[0] for k in keys1}
    assert prefixes == {"a", "b"}
    # Local overlay must not mutate the caller's p dict.
    session: dict[str, float] = {}
    hop_dp_paths(g, p_store=session, params=params, id_prefix="g_")
    assert session == {}


def test_run_s4_all_graphs_prefixes_chain_ids() -> None:
    g = _graph_two_chains()
    params = Params(s4_paths_per_graph=2, min_path_len=6, max_hops=10, prize_top=25)
    pool = run_s4_all_graphs({"sq1": g}, p_store={}, params=params)
    assert [c.chain_id for c in pool] == ["sq1_1", "sq1_2"]
    assert all(c.score > 0 for c in pool)
    assert all(
        sum(1 for e in c.all_edges() if e.source == "prize") >= 2 for c in pool
    )


def test_collected_keys_skip_first_cluster() -> None:
    g = _graph_two_chains()
    params = Params(
        prize_top=25,
        max_hops=10,
        min_path_len=6,
        s4_paths_per_graph=1,
        s4_min_prize_edges=2,
    )
    first = hop_dp_paths(g, p_store={}, params=params, id_prefix="g_")
    assert len(first) == 1
    taken = first[0].all_edge_keys()
    rest = run_s4_all_graphs(
        {"g": g},
        p_store={},
        params=params,
        collected_keys=taken,
    )
    assert len(rest) == 1
    assert not set(rest[0].all_edge_keys()) & set(taken)


def test_frozen_prize_top_does_not_promote_tail() -> None:
    g = _graph_two_chains()
    params = Params(
        prize_top=6,
        max_hops=10,
        min_path_len=6,
        s4_paths_per_graph=2,
        s4_min_prize_edges=2,
    )
    paths = hop_dp_paths(g, p_store={}, params=params, id_prefix="g_")
    assert len(paths) == 1
    assert all(k.startswith("a") for k in paths[0].all_edge_keys())


def test_shared_overlay_across_graphs() -> None:
    g = _graph_two_chains()
    params = Params(
        prize_top=25,
        max_hops=10,
        min_path_len=6,
        s4_paths_per_graph=2,
        s4_share_prize_across_graphs=True,
    )
    pool = run_s4_all_graphs(
        {"sq1": g, "sq2": g},
        p_store={},
        params=params,
    )
    assert [c.chain_id for c in pool] == ["sq1_1", "sq1_2"]
    indep = Params(
        prize_top=25,
        max_hops=10,
        min_path_len=6,
        s4_paths_per_graph=2,
        s4_share_prize_across_graphs=False,
    )
    pool_i = run_s4_all_graphs(
        {"sq1": g, "sq2": g}, p_store={}, params=indep
    )
    assert {c.chain_id for c in pool_i} == {"sq1_1", "sq1_2", "sq2_1", "sq2_2"}


def test_effort_path_budget() -> None:
    assert Params(effort="low").effort_max_paths() == 10
    assert Params(effort="medium").effort_max_paths() == 15
    assert Params(effort="high").effort_max_paths() == 20
    assert Params(effort="low").effort_emit_top_k() == 5
    assert Params(effort="medium").effort_emit_top_k() == 10
    assert Params(effort="high").effort_emit_top_k() == 15
    assert Params(effort="high", emit_top_k=7).effort_emit_top_k() == 7


def test_s4_fill_budget_mines_until_cap() -> None:
    g = _graph_two_chains()
    params = Params(
        prize_top=25,
        max_hops=10,
        min_path_len=6,
        s4_paths_per_graph=1,
        s4_min_prize_edges=2,
    )
    one = run_s4_fill_budget(
        {"g": g}, p_store={}, params=params, budget=1
    )
    assert len(one) == 1
    two = run_s4_fill_budget(
        {"g": g}, p_store={}, params=params, budget=2
    )
    assert len(two) == 2
    assert not set(two[0].all_edge_keys()) & set(two[1].all_edge_keys())


def _emit_chain(cid: str, score: float, n_keys: int = 1) -> Chain:
    return Chain(
        chain_id=cid,
        edge_keys=[f"{cid}_{i}" for i in range(n_keys)],
        score=score,
        source_graph="g",
    )


def test_rank_chains_best_score_first() -> None:
    ranked = rank_chains_for_emit(
        [_emit_chain("low", 1.0, 5), _emit_chain("high", 3.0, 1), _emit_chain("mid", 2.0, 2)]
    )
    assert [c.chain_id for c in ranked] == ["high", "mid", "low"]


def test_rank_chains_tie_break_longer() -> None:
    ranked = rank_chains_for_emit(
        [_emit_chain("short", 2.0, 1), _emit_chain("long", 2.0, 4)]
    )
    assert [c.chain_id for c in ranked] == ["long", "short"]


def test_emit_cut_score_frac_drops_tail() -> None:
    ranked = [_emit_chain("a", 4.0), _emit_chain("b", 2.0), _emit_chain("c", 0.5)]
    kept = emit_cut_chains(
        ranked,
        Params(effort="medium", emit_score_frac=0.25, emit_top_k_medium=10),
    )
    # floor = 1.0 → drop 0.5
    assert [c.score for c in kept] == [4.0, 2.0]
    assert [c.chain_id for c in kept] == ["a1", "a2"]


def test_emit_cut_effort_top_k() -> None:
    ranked = [_emit_chain(str(i), 10.0 - i) for i in range(12)]
    kept = emit_cut_chains(
        ranked,
        Params(effort="low", emit_score_frac=0.0, emit_top_k_low=5),
    )
    assert len(kept) == 5
    assert [c.score for c in kept] == [10.0, 9.0, 8.0, 7.0, 6.0]


def test_emit_cut_top_k() -> None:
    ranked = [_emit_chain("a", 4.0), _emit_chain("b", 3.0), _emit_chain("c", 2.0)]
    kept = emit_cut_chains(ranked, Params(emit_score_frac=0.0, emit_top_k=2))
    assert [c.score for c in kept] == [4.0, 3.0]


def test_emit_cut_never_empty() -> None:
    kept = emit_cut_chains(
        [_emit_chain("only", 0.01)],
        Params(emit_score_frac=0.25, emit_top_k=0),
    )
    assert len(kept) == 1
    assert kept[0].chain_id == "a1"


def test_metrics_at_k_prefixes() -> None:
    from tests.evaluate_v6 import RECALL_AT_KS, metrics_at_k

    ranked = [
        {"edges": [{"evidence": "g1"}], "score": 3},
        {"edges": [{"evidence": "g2"}], "score": 2},
        {"edges": [{"evidence": "noise"}], "score": 1},
    ]
    gold = {"g1", "g2"}
    out = metrics_at_k(ranked, gold, ks=(1, 2, 5))
    assert out["1"]["recall"] == 0.5
    assert abs(float(out["1"]["precision"]) - 1.0) < 1e-9
    assert out["2"]["recall"] == 1.0
    assert out["5"]["recall"] == 1.0
    assert out["5"]["n_paths"] == 3
    assert RECALL_AT_KS[:3] == (1, 2, 3)

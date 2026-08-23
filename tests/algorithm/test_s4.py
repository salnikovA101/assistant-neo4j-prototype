"""S4 carousel: shared p, one tour / sq / round, first round full."""

from __future__ import annotations

from dataclasses import replace

from server.algorithm.models import CandidateGraph, Chain, EdgeRecord
from server.algorithm.params import Params
from server.algorithm.pipeline import label_chains_for_assistant
from server.algorithm.scoring import rank_contribs
from server.algorithm.stage4_hop_dp import run_s4_carousel


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


def _graph_two_chains(
    *,
    source: str = "g",
    sim_a: float = 1.0,
    sim_b: float = 0.80,
) -> CandidateGraph:
    a = _linear_chain("a", 6, sim_a)
    b = _linear_chain("b", 6, sim_b)
    edges = {**a, **b}
    node_to: dict[str, list[str]] = {}
    for e in edges.values():
        node_to.setdefault(e.start_id, []).append(e.edge_key)
        node_to.setdefault(e.end_id, []).append(e.edge_key)
    return CandidateGraph(
        source_graph=source,
        edges=edges,
        node_to_edges=node_to,
        transition_adj=_adj_for_linears(a, b),
    )


def _as_source(g: CandidateGraph, source: str) -> CandidateGraph:
    return replace(g, source_graph=source)


def _carousel_params(**kwargs) -> Params:
    data = dict(
        prize_top=25,
        max_hops=10,
        min_path_len=6,
        s4_min_prize_edges=2,
        s4_p_decay=0.7,
        s4_cost_power=1.5,
    )
    data.update(kwargs)
    return Params(**data)


def test_carousel_first_round_all_graphs_before_second_sq1() -> None:
    g1 = _graph_two_chains(source="sq1")
    g2 = _as_source(g1, "sq2")
    pool = run_s4_carousel(
        {"sq1": g1, "sq2": g2},
        params=_carousel_params(),
        budget=4,
    )
    assert [c.chain_id for c in pool[:2]] == ["sq1_1", "sq2_1"]
    assert [c.chain_id for c in pool] == ["sq1_1", "sq2_1", "sq1_2", "sq2_2"]


def test_carousel_shared_p_decay_resorts() -> None:
    """After sq1 takes the a-cluster, p*=0.7 drops a below b on the shared list."""
    g1 = _graph_two_chains(source="sq1", sim_a=0.90, sim_b=0.85)
    g2 = _as_source(g1, "sq2")
    contrib0, _ = rank_contribs(g1.edges, p_store={}, params=_carousel_params())
    assert contrib0["ae0"] > contrib0["be0"]

    pool = run_s4_carousel(
        {"sq1": g1, "sq2": g2},
        params=_carousel_params(s4_p_decay=0.7),
        budget=2,
    )
    assert len(pool) == 2
    assert all(k.startswith("a") for k in pool[0].all_edge_keys())
    assert all(k.startswith("b") for k in pool[1].all_edge_keys())


def test_carousel_p_decay_zero_zeros_walk() -> None:
    g = _graph_two_chains(source="sq1", sim_a=0.90, sim_b=0.85)
    pool = run_s4_carousel(
        {"sq1": g},
        params=_carousel_params(s4_p_decay=0.0),
        budget=2,
    )
    assert len(pool) == 2
    assert all(k.startswith("a") for k in pool[0].all_edge_keys())
    assert all(k.startswith("b") for k in pool[1].all_edge_keys())
    assert not set(pool[0].all_edge_keys()) & set(pool[1].all_edge_keys())


def test_carousel_partial_last_round() -> None:
    graphs = {
        "sq1": _graph_two_chains(source="sq1"),
        "sq2": _as_source(_graph_two_chains(source="sq2"), "sq2"),
        "sq3": _as_source(_graph_two_chains(source="sq3"), "sq3"),
    }
    pool = run_s4_carousel(graphs, params=_carousel_params(), budget=4)
    assert [c.chain_id for c in pool] == ["sq1_1", "sq2_1", "sq3_1", "sq1_2"]


def test_effort_path_budget() -> None:
    assert Params(effort="low").effort_max_paths() == Params().max_paths_low
    assert Params(effort="medium").effort_max_paths() == Params().max_paths_medium
    assert Params(effort="high").effort_max_paths() == Params().max_paths_high


def test_label_chains_keeps_carousel_order() -> None:
    labeled = label_chains_for_assistant(
        [
            Chain("sq2_1", ["b"], 1.0, source_graph="sq2"),
            Chain("sq1_1", ["a"], 3.0, source_graph="sq1"),
        ]
    )
    assert [c.chain_id for c in labeled] == ["a1", "a2"]
    assert [c.source_graph for c in labeled] == ["sq2", "sq1"]


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


def test_metrics_at_k_empty_pool_is_zero() -> None:
    from tests.evaluate_v6 import metrics_at_k

    out = metrics_at_k([], {"g1"}, ks=(1, 5, 20))
    assert out["1"]["recall"] == 0.0
    assert out["20"]["n_paths"] == 0
    assert out["5"]["precision"] == 0.0


def test_mean_at_k_counts_empty_reports() -> None:
    from tests.evaluate_v6 import RECALL_AT_KS, mean_at_k, metrics_at_k

    hit = {"recall_at_k": metrics_at_k(
        [{"edges": [{"evidence": "g1"}], "score": 1}],
        {"g1"},
        ks=RECALL_AT_KS,
    )}
    miss = {"recall_at_k": metrics_at_k([], {"g1"}, ks=RECALL_AT_KS)}
    avg = mean_at_k([hit, miss])
    assert abs(avg["1"]["mean_recall"] - 0.5) < 1e-9
    empty_row = {"recall_at_k": {}}
    avg2 = mean_at_k([hit, empty_row])
    assert abs(avg2["1"]["mean_recall"] - 0.5) < 1e-9

from server.algorithm.models import CandidateGraph, EdgeRecord
from server.algorithm.params import Params
from server.algorithm.stage4_hop_dp import continue_s4_carousel


def edge(key: str, sim: float, start: str, end: str) -> EdgeRecord:
    return EdgeRecord(
        edge_key=key,
        element_id=key,
        rel_type="R",
        start_id=start,
        end_id=end,
        start_name=start,
        end_name=end,
        sim=sim,
        evidence=f"evidence {key}",
        chunk_id=f"chunk-{key}",
    )


def graph(name: str, values: list[tuple[str, float]]) -> CandidateGraph:
    edges = {key: edge(key, sim, f"s-{key}", f"e-{key}") for key, sim in values}
    return CandidateGraph(
        source_graph=name,
        edges=edges,
        node_to_edges={},
        transition_adj={key: [] for key in edges},
    )


def params() -> Params:
    return Params(
        rerank_enabled=False,
        prize_top=8,
        max_hops=1,
        min_path_len=1,
        s4_min_prize_edges=1,
        s4_p_decay=0.0,
    )


def signatures(result) -> list[tuple[str, ...]]:
    return [tuple(chain.all_edge_keys()) for chain in result.chains]


def test_stepwise_continuation_is_prefix_consistent_with_long_run():
    graphs = {"sq-a": graph("sq-a", [("a", 0.9), ("b", 0.8), ("c", 0.7)])}
    long = continue_s4_carousel(graphs, params=params(), budget=3)

    first = continue_s4_carousel(graphs, params=params(), budget=1)
    second = continue_s4_carousel(graphs, params=params(), budget=2, state=first.state)

    assert signatures(first) + signatures(second) == signatures(long)
    assert len(first.state.accepted_signatures) == 1
    assert len(second.state.accepted_signatures) == 3


def test_new_sq_adds_edges_without_resetting_existing_shared_p():
    first_graphs = {"sq-a": graph("sq-a", [("shared", 0.9), ("a", 0.8)])}
    first = continue_s4_carousel(first_graphs, params=params(), budget=1)
    shared_after_first = first.state.p_store["shared"]

    expanded = {
        **first_graphs,
        "sq-b": graph("sq-b", [("new", 0.95), ("shared", 0.7)]),
    }
    continued = continue_s4_carousel(
        expanded,
        params=params(),
        budget=2,
        state=first.state,
        one_per_graph=True,
    )

    assert continued.state.p_store["shared"] <= shared_after_first
    assert "new" in continued.state.p_store
    assert len([chain for chain in continued.chains if chain.source_graph == "sq-b"]) <= 1


def test_manual_round_returns_at_most_one_novel_unit_per_sq():
    graphs = {
        "sq-a": graph("sq-a", [("a1", 0.9), ("a2", 0.8)]),
        "sq-b": graph("sq-b", [("b1", 0.9), ("b2", 0.8)]),
    }
    result = continue_s4_carousel(
        graphs,
        params=params(),
        budget=20,
        one_per_graph=True,
    )
    assert len(result.chains) == 2
    assert {chain.source_graph for chain in result.chains} == {"sq-a", "sq-b"}

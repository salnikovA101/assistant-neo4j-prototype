"""Unit tests for retrieval scoring / S4 / S5 / format (no Neo4j)."""

from __future__ import annotations

import pytest

from server.algorithm.models import CandidateGraph, Chain, EdgeRecord, SubQuestion
from server.algorithm.params import Params
from server.algorithm.scoring import (
    edge_prize_weight,
    rank_contribs,
)
from server.algorithm.stage3_graphs import _finalize_graph, transition_allowed
from server.algorithm.stage4_hop_dp import best_path_for_graph
from server.algorithm.stage5_select import dedup_s4_pool, prepare_s5_batch
from tests.algorithm.mock_decompose import _fallback_statements, _parse_sq


def test_decompose_parse_rejects_questions():
    raw = """
    {"subquestions":[
      {"id":"sq1","text":"What peptides form from casein?"},
      {"id":"sq2","text":"Antimicrobial peptides derived from casein hydrolysis including isracidin."},
      {"id":"sq3","text":"Which pathogens are inhibited?"}
    ]}
    """
    sqs = _parse_sq(raw)
    assert len(sqs) == 1
    assert sqs[0]["id"] == "sq2"
    assert "?" not in sqs[0]["text"]


def test_decompose_fallback_is_declarative():
    sqs = _fallback_statements("What AMPs from casein?")
    assert len(sqs) >= 2
    assert all("?" not in s["text"] for s in sqs)
    assert all(not s["text"].lower().startswith("what ") for s in sqs)


def test_edge_prize_weight_uses_ce():
    p = Params(s_floor=1e-6)
    e = EdgeRecord("a", "id-a", "R", "x", "y", "X", "Y", sim=0.5, rerank_score=0.8, source="ann")
    w = edge_prize_weight(e, p_store={}, params=p)
    assert abs(w - 0.8) < 1e-9


def _ce_edge(key: str, *, sim: float, ce: float | None) -> EdgeRecord:
    return EdgeRecord(
        edge_key=key,
        element_id=key,
        rel_type="R",
        start_id=f"{key}_s",
        end_id=f"{key}_e",
        start_name=f"{key}_s",
        end_name=f"{key}_e",
        sim=sim,
        rerank_score=ce,
        source="ann",
    )


def test_s4_ranks_raw_ce_not_clipped():
    params = Params()
    hi = _ce_edge("hi", sim=0.1, ce=5.0)
    mid = _ce_edge("mid", sim=0.99, ce=0.4)
    assert edge_prize_weight(hi, p_store={}, params=params) > edge_prize_weight(
        mid, p_store={}, params=params
    )


def test_s4_negative_ce_not_replaced_by_cosine():
    params = Params()
    neg = _ce_edge("neg", sim=0.95, ce=-1.5)
    w = edge_prize_weight(neg, p_store={}, params=params)
    assert w == -1.5


def test_s4_missing_ce_uses_cosine():
    params = Params()
    e = _ce_edge("ann", sim=0.7, ce=None)
    assert abs(edge_prize_weight(e, p_store={}, params=params) - 0.7) < 1e-9


def test_s4_p_discount_moves_order_when_ce_unclipped():
    params = Params()
    a = _ce_edge("a", sim=0.2, ce=5.0)
    b = _ce_edge("b", sim=0.2, ce=2.0)
    wa = edge_prize_weight(a, p_store={"a": 0.3}, params=params)
    wb = edge_prize_weight(b, p_store={}, params=params)
    assert wa < wb


def test_dedup_keeps_first_spine():
    e12a = EdgeRecord("e1", "1", "R", "a", "b", "A", "B", evidence="ev1")
    e12b = EdgeRecord("e2", "2", "R", "b", "c", "B", "C", evidence="ev2")
    e3 = EdgeRecord("e3", "3", "R", "x", "y", "X", "Y", evidence="ev3")
    a = Chain(
        "x",
        ["e1", "e2"],
        0.9,
        source_graph="sq1",
        edges=[e12a, e12b],
    )
    b = Chain(
        "y",
        ["e1", "e2"],
        1.5,
        source_graph="sq3",
        edges=[
            EdgeRecord("e1", "1", "R", "a", "b", "A", "B", evidence="ev1"),
            EdgeRecord("e2", "2", "R", "b", "c", "B", "C", evidence="ev2"),
        ],
    )
    c = Chain("z", ["e3"], 0.7, source_graph="sq2", edges=[e3])
    uniq = dedup_s4_pool([a, b, c])
    assert [u.chain_id for u in uniq] == ["x", "z"]
    assert uniq[0].score == 0.9
    assert uniq[0].source_graphs == ["sq1", "sq3"]
    batch = prepare_s5_batch(uniq)
    assert [u.chain_id for u in batch] == ["c1", "c2"]


def test_s5_batch_copies_walk():
    e1 = EdgeRecord("e1", "1", "R", "a", "h", "A", "H", evidence="enter")
    e2 = EdgeRecord("e2", "2", "R", "h", "d", "H", "D", evidence="ray")
    e3 = EdgeRecord("e3", "3", "R", "h", "c", "H", "C", evidence="exit")
    src = Chain(
        "x",
        ["e1", "e3"],
        0.9,
        source_graph="sq1",
        edges=[e1, e3],
        fans={"h": [e2]},
        walk=[e1, e2, e3],
    )
    batch = prepare_s5_batch([src])
    assert len(batch) == 1
    assert [e.edge_key for e in batch[0].walk] == ["e1", "e2", "e3"]


def test_dedup_allows_partial_evidence_overlap():
    """S5 only cuts exact spine copies; shared single evidence is OK."""
    e1 = EdgeRecord("ek1", "id1", "INHIBITS", "a", "b", "A", "B", sim=0.9, evidence="shared")
    e3 = EdgeRecord("ek3", "id3", "INHIBITS", "x", "y", "X", "Y", sim=0.8, evidence="other")
    c1 = Chain("x", ["ek1"], 0.9, edges=[e1])
    c2 = Chain(
        "y",
        ["ek2", "ek3"],
        0.85,
        edges=[
            EdgeRecord("ek2", "id2", "INHIBITS", "a", "c", "A", "C", evidence="shared"),
            e3,
        ],
    )
    uniq = dedup_s4_pool([c1, c2])
    assert [u.chain_id for u in uniq] == ["x", "y"]


def test_s4_blocks_same_evidence_reentry_via_hub():
    """Same evidence cannot appear twice in one unit path (S4), even if adj allows star."""
    ev1 = "Lactobacillus helveticus decrypting antimicrobial peptide"
    ev2 = "Hydrolysis of casein produced peptides against E. coli"
    e_yer = EdgeRecord("e_yer", "1", "INHIBITS", "AMP", "Yer", "AMP", "Yer", sim=0.91, evidence=ev1)
    e_eco = EdgeRecord("e_eco", "2", "STIMULATES", "Eco", "AMP", "Eco", "AMP", sim=0.90, evidence=ev2)
    e_sal = EdgeRecord("e_sal", "3", "INHIBITS", "AMP", "Sal", "AMP", "Sal", sim=0.91, evidence=ev1)
    # Star / shared-vertex transitions allowed at S3
    assert transition_allowed(e_yer, e_eco)
    assert transition_allowed(e_eco, e_sal)
    assert transition_allowed(e_yer, e_sal)

    edges = {"e_yer": e_yer, "e_eco": e_eco, "e_sal": e_sal}
    g = _finalize_graph("sq1", edges, branch_cap=20)
    assert "e_eco" in g.transition_adj["e_yer"]
    assert "e_sal" in g.transition_adj["e_eco"]

    p = Params(min_path_len=1, max_hops=5)
    c = best_path_for_graph(g, p_store={}, params=p)
    assert c is not None
    keys = set(c.all_edge_keys())
    assert not ({"e_yer", "e_sal"} <= keys)


def test_hop_dp_single_and_two_hop():
    e1 = EdgeRecord(
        "e1",
        "id1",
        "REL",
        "n1",
        "n2",
        "A",
        "B",
        sim=0.9,
        embedding=[1.0, 0.0],
        evidence="ev-a",
        source="ann",
    )
    e2 = EdgeRecord(
        "e2",
        "id2",
        "REL",
        "n2",
        "n3",
        "B",
        "C",
        sim=0.85,
        embedding=[1.0, 0.0],
        evidence="ev-b",
        source="ann",
    )
    g = CandidateGraph(
        source_graph="sq1",
        edges={"e1": e1, "e2": e2},
        node_to_edges={"n1": ["e1"], "n2": ["e1", "e2"], "n3": ["e2"]},
        transition_adj={"e1": ["e2"], "e2": ["e1"]},
    )
    p = Params(min_path_len=1, max_hops=3)
    c = best_path_for_graph(g, p_store={}, params=p)
    assert c is not None
    assert 1 <= len(c.all_edge_keys()) <= 3
    # Prefer longer high-prize path: e1+e2 score > either alone
    assert set(c.all_edge_keys()) == {"e1", "e2"}


def test_hop_dp_skips_isolated_prize_continues_cluster():
    iso = EdgeRecord(
        "iso",
        "iso",
        "R",
        "x",
        "y",
        "X",
        "Y",
        sim=0.99,
        rerank_score=5.0,
        evidence="isolated",
        source="ann",
    )
    a = EdgeRecord(
        "a",
        "a",
        "R",
        "n1",
        "n2",
        "A",
        "B",
        sim=0.5,
        rerank_score=0.8,
        evidence="ev-a",
        source="ann",
    )
    b = EdgeRecord(
        "b",
        "b",
        "R",
        "n2",
        "n3",
        "B",
        "C",
        sim=0.49,
        rerank_score=0.7,
        evidence="ev-b",
        source="ann",
    )
    g = CandidateGraph(
        source_graph="g",
        edges={"iso": iso, "a": a, "b": b},
        node_to_edges={
            "x": ["iso"],
            "y": ["iso"],
            "n1": ["a"],
            "n2": ["a", "b"],
            "n3": ["b"],
        },
        transition_adj={"iso": [], "a": ["b"], "b": ["a"]},
    )
    params = Params(
        min_path_len=1,
        max_hops=5,
        prize_top=10,
        s4_min_prize_edges=2,
    )
    c = best_path_for_graph(g, p_store={}, params=params)
    assert c is not None
    assert set(c.all_edge_keys()) == {"a", "b"}


def test_s4_min_prize_edges_zero_allows_single():
    iso = EdgeRecord(
        "iso",
        "iso",
        "R",
        "x",
        "y",
        "X",
        "Y",
        sim=0.9,
        rerank_score=2.0,
        evidence="only",
        source="ann",
    )
    g = CandidateGraph(
        source_graph="g",
        edges={"iso": iso},
        node_to_edges={"x": ["iso"], "y": ["iso"]},
        transition_adj={"iso": []},
    )
    params = Params(
        min_path_len=1,
        max_hops=3,
        prize_top=10,
        s4_min_prize_edges=0,
    )
    c = best_path_for_graph(g, p_store={}, params=params)
    assert c is not None
    assert c.all_edge_keys() == ["iso"]


def test_s4_one_path_per_graph_global_start():
    """Global DP returns at most one path; need not start at max-sim edge."""
    e_low = EdgeRecord("e_low", "1", "R", "a", "b", "A", "B", sim=0.4, evidence="low", source="ann")
    e_mid = EdgeRecord("e_mid", "2", "R", "b", "c", "B", "C", sim=0.95, evidence="mid", source="ann")
    e_hi = EdgeRecord("e_hi", "3", "R", "c", "d", "C", "D", sim=0.9, evidence="hi", source="ann")
    edges = {"e_low": e_low, "e_mid": e_mid, "e_hi": e_hi}
    g = _finalize_graph("sq1", edges, branch_cap=20)
    # prize_top covers all 3 → linear ranks by sim: mid, hi, low
    p = Params(
        min_path_len=2,
        max_hops=5,
        prize_top=3,
        prize_rank_max=1.0,
    )
    c = best_path_for_graph(g, p_store={}, params=p)
    assert c is not None
    keys = c.all_edge_keys()
    assert set(keys) == {"e_low", "e_mid", "e_hi"}
    # ranks: mid=1 → 1.0, hi=2 → 2/3, low=3 → 1/3
    assert abs(c.score - (1.0 + 2.0 / 3.0 + 1.0 / 3.0)) < 1e-6


def test_s4_bridge_in_same_rank_list():
    """Bridges share the sort with anchors; high sim can take a prize seat."""
    a1 = EdgeRecord("a1", "1", "R", "x", "y", "X", "Y", sim=0.8, evidence="a1", source="ann")
    br = EdgeRecord("br", "2", "R", "y", "z", "Y", "Z", sim=0.99, evidence="br", source="bridge")
    a2 = EdgeRecord("a2", "3", "R", "z", "w", "Z", "W", sim=0.5, evidence="a2", source="ann")
    g = _finalize_graph("sq1", {"a1": a1, "br": br, "a2": a2}, branch_cap=20)
    p = Params(
        min_path_len=3,
        max_hops=5,
        prize_top=2,
        prize_rank_max=1.0,
        s4_cost_power=1.5,
        s4_min_prize_edges=2,
    )
    c = best_path_for_graph(g, p_store={}, params=p)
    assert c is not None
    roles = {e.edge_key: e.source for e in c.all_edges()}
    assert roles["br"] == "prize"
    assert roles["a1"] == "prize"
    assert roles["a2"] == "bridge"
    # br r1=1.0, a1 r2=0.5, a2 last x=1 → −1
    assert abs(c.score - (1.5 - 1.0)) < 1e-6


def test_s4_p_only_moves_sort_not_prize_amount():
    """p discounts ranking weight only; rank-1 still pays prize_rank_max."""
    a = EdgeRecord("a", "1", "R", "x", "y", "X", "Y", sim=0.9, evidence="a", source="ann")
    b = EdgeRecord("b", "2", "R", "y", "z", "Y", "Z", sim=0.8, evidence="b", source="ann")
    edges = {"a": a, "b": b}
    p = Params(prize_top=2, prize_rank_max=1.0, s4_cost_power=1.5)
    contrib, prize_keys = rank_contribs(edges, p_store={"a": 0.5}, params=p)
    assert prize_keys == {"a", "b"}
    # 0.9·0.5 = 0.45 < 0.8 → b is rank 1
    assert abs(contrib["b"] - 1.0) < 1e-6
    assert abs(contrib["a"] - 0.5) < 1e-6


def test_s4_prize_top_demotes_weak_ann():
    """Only top prize_top ANN edges get prize; weaker ANN pay demoted cost."""
    e_hi = EdgeRecord("e_hi", "1", "R", "a", "b", "A", "B", sim=0.9, evidence="hi", source="ann")
    e_weak = EdgeRecord("e_weak", "2", "R", "b", "c", "B", "C", sim=0.4, evidence="wk", source="ann")
    e_mid = EdgeRecord("e_mid", "3", "R", "c", "d", "C", "D", sim=0.85, evidence="md", source="ann")
    g = _finalize_graph("sq1", {"e_hi": e_hi, "e_weak": e_weak, "e_mid": e_mid}, branch_cap=20)
    p = Params(
        min_path_len=3,
        max_hops=5,
        prize_top=2,
        prize_rank_max=1.0,
        s4_cost_power=1.5,
    )
    c = best_path_for_graph(g, p_store={}, params=p)
    assert c is not None
    assert set(c.all_edge_keys()) == {"e_hi", "e_weak", "e_mid"}
    # hi r1=1.0, mid r2=0.5, weak demoted r3 x=1 → −1.0
    assert abs(c.score - (1.5 - 1.0)) < 1e-6
    by_key = {e.edge_key: e.source for e in c.all_edges()}
    assert by_key["e_hi"] == "prize"
    assert by_key["e_mid"] == "prize"
    assert by_key["e_weak"] == "bridge"


def test_s4_no_free_bridge_padding_to_max_hops():
    """Tail-rank cost > 0: do not pad a prize pair with unused low-rank edges."""
    e1 = EdgeRecord("e1", "1", "R", "a", "b", "A", "B", sim=0.9, evidence="p1", source="ann")
    e2 = EdgeRecord("e2", "2", "R", "b", "c", "B", "C", sim=0.8, evidence="p2", source="ann")
    b1 = EdgeRecord("b1", "3", "R", "c", "d", "C", "D", sim=0.1, evidence="g1", source="bridge")
    b2 = EdgeRecord("b2", "4", "R", "d", "e", "D", "E", sim=0.05, evidence="g2", source="bridge")
    g = _finalize_graph("sq1", {"e1": e1, "e2": e2, "b1": b1, "b2": b2}, branch_cap=20)
    p = Params(
        min_path_len=2,
        max_hops=4,
        prize_top=2,
        prize_rank_max=1.0,
        s4_cost_power=1.5,
    )
    c = best_path_for_graph(g, p_store={}, params=p)
    assert c is not None
    assert set(c.all_edge_keys()) == {"e1", "e2"}
    assert len(c.all_edge_keys()) == 2
    assert abs(c.score - 1.5) < 1e-6


def test_s4_rank_prizes_economics():
    """One list: linear prizes over top-K, x^1.5 tail; p only in sort."""
    edges = {}
    for i in range(14):
        key = f"a{i:02d}"
        edges[key] = EdgeRecord(
            key,
            f"id-{key}",
            "R",
            f"s{i}",
            f"e{i}",
            f"S{i}",
            f"E{i}",
            sim=0.9 - 0.05 * i,
            evidence=key,
            source="ann",
        )
    edges["br"] = EdgeRecord(
        "br",
        "id-br",
        "R",
        "sx",
        "ex",
        "SX",
        "EX",
        sim=0.0,
        evidence="br",
        source="bridge",
    )
    p = Params(prize_top=4, prize_rank_max=1.0, s4_cost_power=1.5)
    contrib, prize_keys = rank_contribs(edges, p_store={}, params=p)

    assert prize_keys == {"a00", "a01", "a02", "a03"}
    assert [round(contrib[f"a{i:02d}"], 3) for i in range(4)] == [1.0, 0.75, 0.5, 0.25]
    # N=15, K=4, span=11; cost = −x^1.5
    n, k = 15, 4
    span = n - k
    assert abs(contrib["a04"] - (-((5 - k) / span) ** 1.5)) < 1e-6
    assert abs(contrib["a13"] - (-((14 - k) / span) ** 1.5)) < 1e-6
    assert abs(contrib["br"] - (-1.0)) < 1e-6
    contrib2, _ = rank_contribs(edges, p_store={"a00": 0.9}, params=p)
    assert abs(contrib2["a01"] - 1.0) < 1e-6
    assert abs(contrib2["a00"] - 0.75) < 1e-6


def _edge(key: str, start: str, end: str, sim: float = 0.9, evidence: str = "") -> EdgeRecord:
    return EdgeRecord(
        key,
        f"id-{key}",
        "INHIBITS",
        start,
        end,
        start,
        end,
        sim=sim,
        embedding=[1.0, 0.0],
        evidence=evidence or f"evidence-{key}",
        chunk_id=f"chunk-{key}",
    )


def test_transition_allows_star_and_same_evidence_adj():
    """S3: any shared vertex OK; evidence anti-dupe is S4-only."""
    ab = _edge("ab", "A", "B")
    ac = _edge("ac", "A", "C")
    bc = _edge("bc", "B", "C")
    xb = _edge("xb", "X", "B")
    assert transition_allowed(ab, ac)
    assert transition_allowed(ac, ab)
    assert transition_allowed(xb, ab)
    assert transition_allowed(ab, bc)
    ab.evidence = "identical blob"
    bc.evidence = "identical blob"
    assert transition_allowed(ab, bc)


def test_reshape_star_walk_to_spine_fans():
    from server.algorithm.unit_reshape import reshape_star_walk

    # A→H, H→D, H→E, H→C  → spine A→H, H→C + fans D,E
    ah = _edge("ah", "A", "H", evidence="enter")
    hd = _edge("hd", "H", "D", evidence="ray-d")
    he = _edge("he", "H", "E", evidence="ray-e")
    hc = _edge("hc", "H", "C", evidence="exit")
    spine, fans, names = reshape_star_walk([ah, hd, he, hc])
    assert [e.edge_key for e in spine] == ["ah", "hc"]
    assert "H" in fans
    assert {e.edge_key for e in fans["H"]} == {"hd", "he"}
    assert names["H"] == "H"


def test_linger_hubs_tags_rays_and_exit():
    from server.algorithm.unit_reshape import linger_hubs

    ah = _edge("ah", "A", "H", evidence="enter")
    hd = _edge("hd", "H", "D", evidence="ray-d")
    he = _edge("he", "H", "E", evidence="ray-e")
    hc = _edge("hc", "H", "C", evidence="exit")
    ck = _edge("ck", "C", "K", evidence="onward")
    assert linger_hubs([ah, hd, he, hc, ck]) == ["", "H", "H", "H", ""]


def test_linger_hubs_through_pair_unmarked():
    from server.algorithm.unit_reshape import linger_hubs

    e1 = _edge("e1", "A", "B")
    e2 = _edge("e2", "B", "C")
    assert linger_hubs([e1, e2]) == ["", ""]
    assert linger_hubs([e1]) == [""]


def test_reconstruct_walk_inserts_fans_between_spine():
    from server.algorithm.unit_reshape import reconstruct_walk, reshape_star_walk

    ah = _edge("ah", "A", "H", evidence="enter")
    hd = _edge("hd", "H", "D", evidence="ray-d")
    he = _edge("he", "H", "E", evidence="ray-e")
    hc = _edge("hc", "H", "C", evidence="exit")
    walk = [ah, hd, he, hc]
    spine, fans, _ = reshape_star_walk(walk)
    assert [e.edge_key for e in reconstruct_walk(spine, fans)] == [
        "ah",
        "hd",
        "he",
        "hc",
    ]


# ---------------------------------------------------------------------------
# Unit format: walk order, @Hub linger, source/conf on the quote line
# ---------------------------------------------------------------------------


def _parse_triple_joints(text: str) -> list[tuple[str, str]]:
    """Extract (left, right) entity pairs from directed triple lines."""
    import re

    pairs: list[tuple[str, str]] = []
    triple_re = re.compile(
        r"^(?:@.+(?:  ))?(.+?) —[A-Za-z0-9_ ]+→ (.+?)\s*$"
    )
    for line in text.splitlines():
        if line.startswith("Chain ") or line.startswith("  "):
            continue
        m = triple_re.match(line)
        if m:
            pairs.append((m.group(1), m.group(2)))
    return pairs


def test_format_single_edge_spine_neo4j_direction():
    """Edge case: one-edge spine uses Neo4j direction; no fans."""
    e = _edge("e1", "A", "B", evidence='quote with "quotes"')
    c = Chain("c1", ["e1"], 1.0, edges=[e])
    text = c.format_unit()
    assert "A —inhibits→ B" in text
    assert '  "quote with \'quotes\'"  (source:None; conf=None)' in text
    assert " (source:None; conf=None)" not in text.split("\n")[1]
    assert "FANS" not in text
    assert "SPINE:" not in text
    assert "(score=" not in text


def test_format_edge_appends_source_file():
    e = _edge("e1", "A", "B", evidence="ab")
    e.source_file = "PMC123.pdf"
    e.confidence = 0.87
    ray = _edge("e2", "B", "C", evidence="bc")
    ray.source_file = "Other.pdf"
    ray.confidence = 0.5
    text = Chain(
        "c1",
        ["e1"],
        1.0,
        edges=[e],
        fans={"B": [ray]},
        fan_hub_names={"B": "B"},
    ).format_unit()
    assert "A —inhibits→ B" in text
    assert '  "ab"  (PMC123.pdf; conf=0.87)' in text
    assert "B —inhibits→ C" in text
    assert '  "bc"  (Other.pdf; conf=0.50)' in text
    # missing source_file → source:None; missing confidence → conf=None
    bare = _edge("e3", "X", "Y", evidence="xy")
    bare.confidence = None
    bare_text = Chain("c2", ["e3"], 1.0, edges=[bare]).format_unit()
    assert "X —inhibits→ Y" in bare_text
    assert '  "xy"  (source:None; conf=None)' in bare_text


def test_format_spine_uses_names_not_node_labels():
    """Neo4j labels never leak into the assistant's edge cards."""
    e = EdgeRecord(
        "e1",
        "id-e1",
        "PRODUCES",
        "m1",
        "met1",
        "Lactobacillus",
        "L-lactic acid",
        start_label="Microbe",
        end_label="Metabolite",
        evidence="makes acid",
    )
    ray = EdgeRecord(
        "e2",
        "id-e2",
        "INHIBITS",
        "met1",
        "p1",
        "L-lactic acid",
        "E. coli",
        start_label="Metabolite",
        end_label="Microbe",
        evidence="kills",
    )
    text = Chain(
        "c1",
        ["e1"],
        1.0,
        edges=[e],
        fans={"met1": [ray]},
        fan_hub_names={"met1": "Metabolite: L-lactic acid"},
    ).format_unit()
    assert "Lactobacillus —produces→ L-lactic acid" in text
    assert "Metabolite:" not in text
    assert '  "makes acid"' in text
    assert "FANS" not in text
    assert "L-lactic acid —inhibits→ E. coli" in text
    assert '  "kills"' in text


def test_labels_are_unbounded_metadata_but_node_ref_is_name_only():
    from server.algorithm.models import format_node_ref, normalize_labels

    assert normalize_labels(["Thing", "Entity", "Thing"]) == ["Entity", "Thing"]
    assert normalize_labels(None) == []
    assert format_node_ref(["NewClass", "AnotherClass"], "Node name") == "Node name"


def test_format_spine_arrows_forward():
    """A->B, B->C prints A -> B then B -> C."""
    e1 = _edge("e1", "A", "B", evidence="ab")
    e2 = _edge("e2", "B", "C", evidence="bc")
    text = Chain("c1", ["e1", "e2"], 1.0, edges=[e1, e2]).format_unit()
    joints = _parse_triple_joints(text)
    assert joints == [("A", "B"), ("B", "C")]


def test_format_spine_arrows_incoming_kept():
    """A->B then C->B prints both Neo4j directions (C -> B, not B -> C)."""
    e1 = _edge("e1", "A", "B", evidence="ab")
    e2 = _edge("e2", "C", "B", evidence="cb")
    text = Chain("c1", ["e1", "e2"], 1.0, edges=[e1, e2]).format_unit()
    joints = _parse_triple_joints(text)
    assert joints == [("A", "B"), ("C", "B")]


def test_format_spine_first_edge_direction_kept():
    """X->A after A->B: both keep Neo4j start->end, no reorientation."""
    e1 = _edge("e1", "A", "B", evidence="ab")
    e2 = _edge("e2", "X", "A", evidence="xa")
    text = Chain("c1", ["e1", "e2"], 1.0, edges=[e1, e2]).format_unit()
    joints = _parse_triple_joints(text)
    assert joints == [("A", "B"), ("X", "A")]


def test_format_spine_spur_directions_and_fans():
    """Hub/reshape: spur stays in walk; linger tags rays and exit."""
    from server.algorithm.unit_reshape import reshape_star_walk

    # Walk: E.coli - L-lactic - Pathogen(spur) - L-lactic - Water kefir
    e1 = _edge("e1", "Ecoli", "Llactic", evidence="el")
    e2 = _edge("e2", "Llactic", "Pathogen", evidence="lp")
    e3 = _edge("e3", "Wkefir", "Llactic", evidence="wl")
    spine, fans, names = reshape_star_walk([e1, e2, e3])
    # Entry+exit on spine, middle ray in fans
    assert [e.edge_key for e in spine] == ["e1", "e3"]
    assert [e.edge_key for e in fans["Llactic"]] == ["e2"]
    text = Chain(
        "c1",
        ["e1", "e3"],
        1.0,
        edges=spine,
        fans=fans,
        fan_hub_names=names,
        walk=[e1, e2, e3],
    ).format_unit()
    joints = _parse_triple_joints(text)
    assert joints == [
        ("Ecoli", "Llactic"),
        ("Llactic", "Pathogen"),
        ("Wkefir", "Llactic"),
    ]
    assert "@Llactic  Llactic —inhibits→ Pathogen" in text
    assert "@Llactic  Wkefir —inhibits→ Llactic" in text
    assert '  "lp"' in text


def test_format_spine_broken_still_prints_direction():
    """Edge case: disconnected edges still print Neo4j direction."""
    e1 = _edge("e1", "A", "B", evidence="ab")
    e2 = _edge("e2", "X", "Y", evidence="xy")
    text = Chain("c1", ["e1", "e2"], 1.0, edges=[e1, e2]).format_unit()
    joints = _parse_triple_joints(text)
    assert joints == [("A", "B"), ("X", "Y")]


def test_format_empty_walk_prints_label_only():
    """Empty walk does not dump raw edge_keys as if they were evidence."""
    c = Chain("c1", ["key-only-1", "key-only-2"], 0.5, edges=[])
    text = c.format_unit()
    assert text.strip() == "Chain c1"
    assert "key-only-1" not in text


def test_format_empty_fans_dict_omitted():
    """Edge case: empty fan lists are skipped."""
    e = _edge("e1", "A", "B", evidence="ab")
    c = Chain("c1", ["e1"], 1.0, edges=[e], fans={"H": []}, fan_hub_names={"H": "Hub"})
    text = c.format_unit()
    assert "FANS" not in text
    assert "@Hub" not in text


def test_format_fans_out_star_direction():
    """Co-outgoing rays stay in walk order with @H, including exit."""
    ah = _edge("ah", "A", "H", evidence="enter")
    hd = _edge("hd", "H", "D", evidence="ray-d")
    he = _edge("he", "H", "E", evidence="ray-e")
    hc = _edge("hc", "H", "C", evidence="exit")
    text = Chain(
        "c1",
        ["ah", "hc"],
        1.0,
        edges=[ah, hc],
        fans={"H": [hd, he]},
        fan_hub_names={"H": "H"},
        walk=[ah, hd, he, hc],
    ).format_unit()
    assert "FANS" not in text
    assert "SPINE:" not in text
    joints = _parse_triple_joints(text)
    assert joints == [("A", "H"), ("H", "D"), ("H", "E"), ("H", "C")]
    assert not text.splitlines()[1].startswith("@")
    assert "@H  H —inhibits→ D" in text
    assert '  "ray-d"' in text
    assert "@H  H —inhibits→ E" in text
    assert "@H  H —inhibits→ C" in text


def test_format_fans_in_star_direction():
    """Co-incoming rays: Leaf —rel→ Hub as a full triple with @H."""
    ah = _edge("ah", "A", "H", evidence="enter")
    dh = _edge("dh", "D", "H", evidence="in-d")
    eh = _edge("eh", "E", "H", evidence="in-e")
    hc = _edge("hc", "H", "C", evidence="exit")
    text = Chain(
        "c1",
        ["ah", "hc"],
        1.0,
        edges=[ah, hc],
        fans={"H": [dh, eh]},
        fan_hub_names={"H": "H"},
        walk=[ah, dh, eh, hc],
    ).format_unit()
    assert "@H  D —inhibits→ H" in text
    assert '  "in-d"' in text
    assert "@H  E —inhibits→ H" in text
    assert '  "in-e"' in text


def test_format_spine_after_reshape_hub_walk():
    """Walk order: enter, @ rays, @ exit; arrows kept."""
    from server.algorithm.unit_reshape import reshape_star_walk

    ah = _edge("ah", "A", "H", evidence="enter")
    hd = _edge("hd", "H", "D", evidence="ray-d")
    he = _edge("he", "H", "E", evidence="ray-e")
    hc = _edge("hc", "H", "C", evidence="exit")
    walk = [ah, hd, he, hc]
    spine, fans, names = reshape_star_walk(walk)
    assert [e.edge_key for e in spine] == ["ah", "hc"]
    assert {e.edge_key for e in fans["H"]} == {"hd", "he"}
    text = Chain(
        "c1",
        [e.edge_key for e in spine],
        2.0,
        edges=spine,
        fans=fans,
        fan_hub_names=names,
        walk=walk,
    ).format_unit()
    joints = _parse_triple_joints(text)
    assert joints == [("A", "H"), ("H", "D"), ("H", "E"), ("H", "C")]
    assert "FANS" not in text
    assert "@H  H —inhibits→ D" in text
    assert '  "ray-d"' in text
    assert "@H  H —inhibits→ E" in text
    assert "@H  H —inhibits→ C" in text


def test_format_two_hubs_keeps_walk_order():
    """Two hub stays in one tour: @ tags follow walk, not dumped FANS blocks."""
    xh = _edge("xh", "X", "H1", evidence="enter")
    h1a = _edge("h1a", "H1", "A", evidence="ray-a")
    h1h2 = _edge("h1h2", "H1", "H2", evidence="bridge")
    h2b = _edge("h2b", "H2", "B", evidence="ray-b")
    h2y = _edge("h2y", "H2", "Y", evidence="exit")
    walk = [xh, h1a, h1h2, h2b, h2y]
    text = Chain(
        "c1",
        ["xh", "h1h2", "h2y"],
        1.0,
        walk=walk,
        fan_hub_names={"H1": "H1", "H2": "H2"},
    ).format_unit()
    joints = _parse_triple_joints(text)
    assert joints == [
        ("X", "H1"),
        ("H1", "A"),
        ("H1", "H2"),
        ("H2", "B"),
        ("H2", "Y"),
    ]
    assert "FANS" not in text
    assert "@H1  H1 —inhibits→ A" in text
    assert "@H1  H1 —inhibits→ H2" in text
    assert "@H2  H2 —inhibits→ B" in text
    assert "@H2  H2 —inhibits→ Y" in text


def test_format_unit_always_uses_arrows():
    """Cypher -[REL]- / <-[REL]- never emitted; cards use —rel→."""
    e1 = _edge("e1", "A", "B", evidence="ab")
    e2 = _edge("e2", "C", "B", evidence="cb")
    fans = {"B": [_edge("bd", "B", "D", evidence="bd"), _edge("xb", "X", "B", evidence="xb")]}
    text = Chain(
        "c1",
        ["e1", "e2"],
        1.0,
        edges=[e1, e2],
        fans=fans,
        fan_hub_names={"B": "B"},
    ).format_unit()
    assert "—" in text and "→" in text
    assert "-[" not in text
    assert "<-[" not in text
    assert "B —inhibits→ D" in text
    assert "X —inhibits→ B" in text
    assert _parse_triple_joints(text) == [
        ("A", "B"),
        ("B", "D"),
        ("X", "B"),
        ("C", "B"),
    ]


def test_format_unit_card_layout_walk_and_quote_meta():
    """Quote line holds source/conf; no SPINE/FANS; no score on UNIT header."""
    spine = [
        EdgeRecord(
            "e1",
            "id-e1",
            "REQUIRES",
            "d1",
            "pH",
            "Ph-sensitive dyes",
            "pH",
            start_label="Metabolite",
            end_label="EnvironmentCondition",
            evidence="Colorimetric indicators, such as pH-sensitive dyes",
            source_file="a.pdf",
            confidence=1.0,
        ),
    ]
    fan = EdgeRecord(
        "e2",
        "id-e2",
        "REQUIRES",
        "al",
        "pH",
        "Alizarin",
        "pH",
        start_label="Metabolite",
        end_label="EnvironmentCondition",
        evidence=(
            "plant-based natural pigments, such as anthocyanins, curcumin, "
            "and alizarin"
        ),
        source_file="b.pdf",
        confidence=1.0,
    )
    text = Chain(
        "c1",
        ["e1"],
        8.04,
        edges=spine,
        fans={"pH": [fan]},
        fan_hub_names={"pH": "EnvironmentCondition: pH"},
    ).format_unit()
    assert text == (
        "Chain c1\n"
        "Ph-sensitive dyes —requires→ pH\n"
        '  "Colorimetric indicators, such as pH-sensitive dyes"'
        "  (a.pdf; conf=1.00)\n"
        "Alizarin —requires→ pH\n"
        '  "plant-based natural pigments, such as anthocyanins, curcumin, '
        'and alizarin"  (b.pdf; conf=1.00)'
    )
    assert "(score=" not in text


def test_star_walk_forms_unit_with_fans():
    """AMP star walk is allowed; reshape collapses rays into FANS."""
    e_in = _edge("e_in", "Lab", "AMP", evidence="enter amp")
    e_sal = _edge("e_sal", "AMP", "Sal", evidence="ray sal")
    e_yer = _edge("e_yer", "AMP", "Yer", evidence="ray yer")
    e_out = _edge("e_out", "AMP", "Eco", evidence="exit eco")
    edges = {
        "e_in": e_in,
        "e_sal": e_sal,
        "e_yer": e_yer,
        "e_out": e_out,
    }
    g = _finalize_graph("sq1", edges, branch_cap=20)
    assert "e_sal" in g.transition_adj["e_in"]
    assert "e_yer" in g.transition_adj["e_sal"] or "e_out" in g.transition_adj["e_sal"]
    assert transition_allowed(e_sal, e_yer)

    p = Params(
        min_path_len=2,
        max_hops=5,
    )
    c = best_path_for_graph(g, p_store={}, params=p)
    assert c is not None
    assert [e.edge_key for e in c.walk] == c.all_edge_keys()
    if c.fans:
        assert any(c.fans.values())


def _chain_with_ev(cid: str, edge_key: str, evidence: str, score: float = 0.9) -> Chain:
    e = EdgeRecord(
        edge_key,
        f"id-{edge_key}",
        "INHIBITS",
        "a",
        "b",
        "A",
        "B",
        sim=score,
        evidence=evidence,
        chunk_id=f"chunk-{edge_key}",
    )
    return Chain(cid, [edge_key], score, edges=[e], source_graph="sq1")


def _fake_edge(key: str, sim: float, evidence: str = "") -> EdgeRecord:
    return EdgeRecord(
        edge_key=key,
        element_id=key,
        rel_type="REL",
        start_id="a",
        end_id="b",
        start_name="A",
        end_name="B",
        sim=sim,
        evidence=evidence or f"evidence for {key}",
    )


def test_rerank_disabled_keeps_top_by_sim():
    import asyncio

    from server.algorithm.stage2b_rerank import rerank_ann_by_sq

    sq = SubQuestion(id="sq1", text="Mixed lactic starters for prematuration.")
    hits = {
        "e1": _fake_edge("e1", 0.9),
        "e2": _fake_edge("e2", 0.5),
        "e3": _fake_edge("e3", 0.8),
    }
    params = Params(
        rerank_enabled=False,
        L_raw_max=300,
        L=2,
    )

    async def _run():
        return await rerank_ann_by_sq([sq], {"sq1": hits}, params)

    out, ann_keys, rerank_keys, ann_sims = asyncio.run(_run())
    assert ann_keys["sq1"] == ["e1", "e3", "e2"]
    assert rerank_keys["sq1"] == ["e1", "e3"]
    assert set(out["sq1"].keys()) == {"e1", "e3"}
    assert ann_sims["e1"] == 0.9
    assert ann_sims["e3"] == 0.8
    assert ann_sims["e2"] == 0.5


def test_rerank_ce_reverses_order_and_keeps():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from server.algorithm.stage2b_rerank import rerank_ann_by_sq

    sq = SubQuestion(id="sq1", text="Culture choices affect acidification.")
    hits = {
        "e1": _fake_edge("e1", 0.9, "high sim low ce"),
        "e2": _fake_edge("e2", 0.5, "mid"),
        "e3": _fake_edge("e3", 0.1, "low sim high ce"),
    }
    params = Params(
        rerank_enabled=True,
        L_raw_max=300,
        L=2,
        rerank_batch_size=64,
        rerank_url="http://127.0.0.1:7997",
    )

    # CE returns ascending scores by input index → reverse of sim order (e1,e2,e3)
    # so scores: e1=1, e2=2, e3=3 → keep e3, e2
    async def fake_post(url, json=None, timeout=None):
        texts = (json or {}).get("texts") or []
        n = len(texts)
        payload = [{"index": i, "score": float(i + 1)} for i in range(n)]
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=payload)
        return resp

    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=fake_post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    async def _run():
        with patch(
            "server.algorithm.stage2b_rerank.httpx.AsyncClient",
            return_value=mock_client,
        ):
            return await rerank_ann_by_sq([sq], {"sq1": hits}, params)

    out, ann_keys, rerank_keys, ann_sims = asyncio.run(_run())
    assert ann_keys["sq1"] == ["e1", "e2", "e3"]
    assert rerank_keys["sq1"] == ["e3", "e2"]
    assert list(out["sq1"].keys()) == ["e3", "e2"]
    assert out["sq1"]["e3"].rerank_score > out["sq1"]["e2"].rerank_score
    assert set(ann_sims) == {"e1", "e2", "e3"}


def test_n_gold_in_keys():
    def n_gold_in_keys(keys, gold_ev, key_to_ev):
        if not gold_ev:
            return 0
        found = {key_to_ev[k] for k in keys if k in key_to_ev and key_to_ev[k]}
        return len(found & gold_ev)

    gold = {"ev_a", "ev_b", "ev_c"}
    key_to_ev = {
        "k1": "ev_a",
        "k2": "ev_b",
        "k3": "noise",
        "k4": "ev_a",  # duplicate gold
    }
    assert n_gold_in_keys(["k1", "k2", "k3", "k4"], gold, key_to_ev) == 2
    assert n_gold_in_keys(["k3"], gold, key_to_ev) == 0
    assert n_gold_in_keys(["k1", "k2"], gold, key_to_ev) == 2
    assert n_gold_in_keys([], gold, key_to_ev) == 0


def test_graphs_for_sqs_drops_global_and_other_sqs():
    from server.algorithm.pipeline import _graphs_for_sqs

    g1 = CandidateGraph(source_graph="sq1", edges={"a": _fake_edge("a", 0.9)})
    g2 = CandidateGraph(source_graph="sq2", edges={"b": _fake_edge("b", 0.8)})
    gg = CandidateGraph(source_graph="global", edges={"c": _fake_edge("c", 0.7)})
    graphs = {"sq1": g1, "sq2": g2, "global": gg}
    open_sqs = [SubQuestion(id="sq1", text="open one")]
    sliced = _graphs_for_sqs(graphs, open_sqs)
    assert set(sliced.keys()) == {"sq1"}
    assert sliced["sq1"] is g1


def test_run_from_graph_cache_skips_s1_s3():
    import asyncio
    from unittest.mock import AsyncMock, patch

    from server.algorithm.graph_cache import build_s3_bundle
    from server.algorithm.pipeline import run

    e1 = _fake_edge("e1", 0.9, "fact one")
    e2 = _fake_edge("e2", 0.8, "fact two")
    e1.start_id, e1.end_id = "n1", "n2"
    e1.start_name, e1.end_name = "N1", "N2"
    e2.start_id, e2.end_id = "n2", "n3"
    e2.start_name, e2.end_name = "N2", "N3"
    e1.rerank_score = 0.95
    e2.rerank_score = 0.85
    g = _finalize_graph("sq1", {"e1": e1, "e2": e2}, branch_cap=20)
    sqs = [{"id": "sq1", "text": "Declarative statement about pathways."}]
    params = Params(
        run_id="test-run",
        effort="low",
        min_path_len=2,
        max_hops=3,
        max_paths_low=5,
        prize_top=25,
        s4_min_prize_edges=1,
    )
    bundle = build_s3_bundle(
        qid="q0",
        question="Q?",
        params=params,
        sqs=sqs,
        graphs={"sq1": g},
        ann_keys={"sq1": ["e1", "e2"]},
        rerank_keys={"sq1": ["e1", "e2"]},
        ann_edge_sims={"e1": 0.9, "e2": 0.8},
    )

    call_counts = {"ann": 0, "s3": 0, "embed": 0}

    async def boom_embed(*_a, **_k):
        call_counts["embed"] += 1
        raise AssertionError("embed should not run on cache hit")

    async def boom_ann(*_a, **_k):
        call_counts["ann"] += 1
        raise AssertionError("ann should not run on cache hit")

    async def boom_s3(*_a, **_k):
        call_counts["s3"] += 1
        raise AssertionError("s3 should not run on cache hit")

    async def fake_hydrate(driver, chains, *, run_id):
        assert run_id == "test-run"
        for c in chains:
            c.text = c.format_unit(c.chain_id)

    async def _run():
        with (
            patch(
                "server.algorithm.pipeline.embed_subquestions",
                side_effect=boom_embed,
            ),
            patch(
                "server.algorithm.pipeline.ann_for_subquestions",
                side_effect=boom_ann,
            ),
            patch(
                "server.algorithm.pipeline.build_all_graphs",
                side_effect=boom_s3,
            ),
            patch(
                "server.algorithm.pipeline.hydrate_chains",
                side_effect=fake_hydrate,
            ),
        ):
            return await run(
                AsyncMock(),
                subquestions=sqs,
                query="Q?",
                effort="low",
                params=params,
                s3_bundle=bundle,
            )

    result = asyncio.run(_run())
    assert call_counts["embed"] == 0
    assert call_counts["ann"] == 0
    assert call_counts["s3"] == 0
    assert result["from_graph_cache"] is True
    assert result["ann_keys_union"]
    assert len(result.get("accepted") or []) >= 1


def test_graph_cache_edge_roundtrip():
    from server.algorithm.graph_cache import (
        build_fingerprint,
        build_s3_bundle,
        bundle_matches,
        edge_from_cache_dict,
        edge_to_cache_dict,
        graph_from_cache_dict,
        graph_to_cache_dict,
        load_s3_bundle_graphs,
    )

    e = EdgeRecord(
        edge_key="k1",
        element_id="el1",
        rel_type="INHIBITS",
        start_id="n1",
        end_id="n2",
        start_name="A",
        end_name="B",
        sim=0.77,
        rerank_score=0.42,
        embedding=[0.1, 0.2, 0.3],
        chunk_id="c1",
        evidence="peptide X inhibits pathogen Y",
        source_file="paper.pdf",
        source="ann",
        confidence=0.9,
    )
    d = edge_to_cache_dict(e)
    assert "embedding" not in d
    e2 = edge_from_cache_dict(d)
    assert e2.edge_key == e.edge_key
    assert e2.rerank_score == e.rerank_score
    assert e2.evidence == e.evidence
    assert e2.embedding == []

    g = _finalize_graph("sq1", {"k1": e}, branch_cap=20)
    gd = graph_to_cache_dict(g)
    g2 = graph_from_cache_dict(gd, branch_cap=20)
    assert len(g2.edges) == 1
    assert "k1" in g2.transition_adj
    assert g2.node_to_edges

    sqs = [SubQuestion(id="sq1", text="claim about peptides")]
    p = Params(prize_top=99)  # S4-only change must not affect fingerprint
    p2 = Params(prize_top=5)
    assert build_fingerprint(p, sqs) == build_fingerprint(p2, sqs)
    p3 = Params(L=50)
    assert build_fingerprint(p, sqs) != build_fingerprint(p3, sqs)
    p4 = Params(run_id="corpus_a")
    assert build_fingerprint(p, sqs) != build_fingerprint(p4, sqs)

    bundle = build_s3_bundle(
        qid="q0",
        question="Q?",
        params=p,
        sqs=sqs,
        graphs={"sq1": g},
        ann_keys={"sq1": ["k1"]},
        rerank_keys={"sq1": ["k1"]},
        ann_edge_sims={"k1": 0.77},
    )
    assert bundle_matches(bundle, question="Q?", params=p2, sqs=sqs)
    assert not bundle_matches(bundle, question="Q?", params=p3, sqs=sqs)
    loaded = load_s3_bundle_graphs(bundle, branch_cap=20)
    assert set(loaded["sq1"].edges) == {"k1"}
    assert loaded["sq1"].edges["k1"].rerank_score == 0.42
    loaded_drop = load_s3_bundle_graphs(
        build_s3_bundle(
            qid="q0",
            question="Q?",
            params=p,
            sqs=sqs,
            graphs={"sq1": g, "global": g},
            ann_keys={"sq1": ["k1"]},
            rerank_keys={"sq1": ["k1"]},
            ann_edge_sims={"k1": 0.77},
        ),
        branch_cap=20,
    )
    assert "global" not in loaded_drop
    assert "sq1" in loaded_drop

    missing = EdgeRecord(
        edge_key="k2",
        element_id="el2",
        rel_type="R",
        start_id="n3",
        end_id="n4",
        start_name="C",
        end_name="D",
        sim=0.5,
        rerank_score=None,
        source="ann",
    )
    d_none = edge_to_cache_dict(missing)
    assert d_none["rerank_score"] is None
    assert edge_from_cache_dict(d_none).rerank_score is None


def test_run_embed_failure_sets_error():
    import asyncio
    from unittest.mock import patch

    from server.algorithm.embed_client import EmbeddingError
    from server.algorithm.pipeline import run

    async def boom(*_a, **_k):
        raise EmbeddingError("http 500")

    async def _run():
        with patch("server.algorithm.pipeline.embed_subquestions", boom):
            return await run(
                driver=None,  # type: ignore[arg-type]
                subquestions=[{"id": "sq1", "text": "q"}],
                effort="low",
                params=Params(run_id="test-run"),
            )

    result = asyncio.run(_run())
    assert result["error"] == "embed_failed"
    assert result["accepted"] == []
    assert "http 500" in str(result.get("error_detail") or "")


def test_parse_confidence_keeps_zero():
    from server.algorithm.models import parse_confidence

    assert parse_confidence(0.0) == 0.0
    assert parse_confidence(0) == 0.0
    assert parse_confidence(None) is None
    assert parse_confidence("") is None
    assert parse_confidence(0.42) == pytest.approx(0.42)


def test_rerank_malformed_json_missing_index():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    from server.algorithm.stage2b_rerank import RerankError, rerank_ann_by_sq

    sq = SubQuestion(id="sq1", text="Culture choices affect acidification.")
    hits = {"e1": _fake_edge("e1", 0.9, "evidence one")}
    params = Params(rerank_enabled=True, L_raw_max=300, L=2, rerank_url="http://127.0.0.1:7997")

    async def fake_post(url, json=None, timeout=None):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json = MagicMock(return_value=[{"score": 1.0}])
        return resp

    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=fake_post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    async def _run():
        with patch(
            "server.algorithm.stage2b_rerank.httpx.AsyncClient",
            return_value=mock_client,
        ):
            return await rerank_ann_by_sq([sq], {"sq1": hits}, params)

    with pytest.raises(RerankError, match="index"):
        asyncio.run(_run())


def test_rerank_http_error_raises():
    import asyncio
    from unittest.mock import AsyncMock, MagicMock, patch

    import httpx

    from server.algorithm.stage2b_rerank import RerankError, rerank_ann_by_sq

    sq = SubQuestion(id="sq1", text="Culture choices affect acidification.")
    hits = {"e1": _fake_edge("e1", 0.9, "evidence one")}
    params = Params(rerank_enabled=True, L_raw_max=300, L=2, rerank_url="http://127.0.0.1:7997")

    async def fake_post(url, json=None, timeout=None):
        raise httpx.HTTPStatusError(
            "502",
            request=httpx.Request("POST", url),
            response=httpx.Response(502),
        )

    mock_client = MagicMock()
    mock_client.post = AsyncMock(side_effect=fake_post)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)

    async def _run():
        with patch(
            "server.algorithm.stage2b_rerank.httpx.AsyncClient",
            return_value=mock_client,
        ):
            return await rerank_ann_by_sq([sq], {"sq1": hits}, params)

    with pytest.raises(RerankError):
        asyncio.run(_run())


def test_pipeline_rerank_failure_sets_error():
    import asyncio
    from unittest.mock import patch

    from server.algorithm.stage2b_rerank import RerankError
    from server.algorithm.pipeline import run

    async def fake_embed(*_a, **_k):
        return {"sq1": [0.1, 0.2]}

    async def fake_ann(*_a, **_k):
        return {"sq1": {"e1": _fake_edge("e1", 0.9)}}

    async def boom(*_a, **_k):
        raise RerankError("ce down")

    async def _run():
        with (
            patch("server.algorithm.pipeline.embed_subquestions", fake_embed),
            patch("server.algorithm.pipeline.ann_for_subquestions", fake_ann),
            patch("server.algorithm.pipeline.rerank_ann_by_sq", boom),
        ):
            return await run(
                driver=None,  # type: ignore[arg-type]
                subquestions=[{"id": "sq1", "text": "q"}],
                effort="low",
                params=Params(run_id="test-run"),
            )

    result = asyncio.run(_run())
    assert result["error"] == "rerank_failed"
    assert result["accepted"] == []
    assert "ce down" in str(result.get("error_detail") or "")


def test_scored_reports_exclude_infra_errors():
    from tests.evaluate_v6 import mean_metrics, scored_reports

    reports = [
        {
            "recall_accepted": 1.0,
            "precision_accepted": 1.0,
            "error": "rerank_failed",
        },
        {"recall_accepted": 0.5, "precision_accepted": 0.25},
    ]
    scored = scored_reports(reports)
    assert len(scored) == 1
    means = mean_metrics(scored)
    assert means["n"] == 1
    assert means["mean_recall_accepted"] == 0.5


@pytest.mark.asyncio
async def test_mock_decompose_raises_when_slm_fails(monkeypatch):
    from tests.algorithm import mock_decompose as md

    class BoomClient:
        def __init__(self, *a, **k):
            pass

        @property
        def chat(self):
            return self

        @property
        def completions(self):
            return self

        async def create(self, **kwargs):
            raise ConnectionError("slm down")

    monkeypatch.setattr(md, "AsyncOpenAI", BoomClient)
    monkeypatch.setattr(md, "load_config", lambda: object())
    monkeypatch.setattr(
        md,
        "_resolve_tool_llm_profile",
        lambda _cfg: type("P", (), {"api_key": "", "base_url": "http://x", "model": "m", "think": False})(),
    )
    monkeypatch.setattr(md, "_resolve_slm_base_url", lambda url: url)
    with pytest.raises(RuntimeError, match="mock_decompose failed"):
        await md.mock_decompose("What AMPs from casein?")

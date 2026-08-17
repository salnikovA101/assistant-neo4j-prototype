from server.algorithm.cypher.edges import FETCH_VIZ_BY_EDGE_IDS
from server.algorithm.models import EdgeRecord
from server.core.graph_runs import (
    chain_unit_index,
    current_graph_collector,
    new_graph_collector,
    record_accepted_chains,
    reset_graph_collector,
)
from server.tools.graph_viz import build_chain_views, merge_views


def _edge(element_id: str, start: str = "A", end: str = "B") -> dict:
    return {
        "edge_key": f"key-{element_id}",
        "element_id": element_id,
        "type": "PRODUCES",
        "start": start,
        "end": end,
        "start_label": "Microbe",
        "end_label": "Metabolite",
        "start_id": f"node-{start}",
        "end_id": f"node-{end}",
        "evidence": f"evidence {element_id}",
        "chunk_id": "chunk-1",
        "source_file": "paper.pdf",
        "sim": 0.5,
        "confidence": 0.87,
        "source": "prize",
    }


def test_to_dict_edge_has_viz_ids_and_no_embedding() -> None:
    edge = EdgeRecord(
        edge_key="ek",
        element_id="5:1",
        rel_type="PRODUCES",
        start_id="4:1",
        end_id="4:2",
        start_name="A",
        end_name="B",
        start_label="Microbe",
        end_label="Metabolite",
        embedding=[1.0, 2.0],
        confidence=0.87,
    )

    data = edge.to_dict_edge()

    assert data["element_id"] == "5:1"
    assert data["confidence"] == 0.87
    assert "embedding" not in data


def test_viz_query_selects_no_embeddings() -> None:
    assert "embedding" not in FETCH_VIZ_BY_EDGE_IDS


def test_build_chain_views_roles_and_merge_dedupe() -> None:
    spine = _edge("5:1")
    fan = _edge("5:2", start="A", end="C")
    chain = {
        "chain_id": "a1",
        "score": 0.9,
        "edges": [spine],
        "fans": {"node-A": [fan]},
    }

    views = build_chain_views([chain])

    assert len(views) == 1
    assert len(views[0]["nodes"]) == 3
    assert len(views[0]["edges"]) == 2
    roles = {edge["id"]: edge["role"] for edge in views[0]["edges"]}
    assert roles == {"5:1": "spine", "5:2": "fan"}
    assert all("embedding" not in edge["properties"] for edge in views[0]["edges"])

    views = build_chain_views(
        [
            chain,
            {"chain_id": "a2", "score": 0.8, "edges": [spine], "fans": {}},
        ]
    )
    merged = merge_views(views)

    assert len(merged["nodes"]) == 3
    assert len(merged["edges"]) == 2
    merged_spine = next(edge for edge in merged["edges"] if edge["id"] == "5:1")
    assert merged_spine["chain_ids"] == ["a1", "a2"]
    assert merged_spine["role"] == "spine"


def test_build_chain_views_uniquifies_duplicate_chain_ids() -> None:
    views = build_chain_views(
        [
            {"chain_id": "a1", "score": 0.9, "edges": [_edge("5:1")], "fans": {}},
            {"chain_id": "a1", "score": 0.4, "edges": [_edge("5:9", start="X", end="Y")], "fans": {}},
        ]
    )
    assert [v["id"] for v in views] == ["a1", "a2"]
    assert [v["label"] for v in views] == ["Цепь 1", "Цепь 2"]
    assert views[0]["edges"][0]["chain_ids"] == ["a1"]
    assert views[1]["edges"][0]["chain_ids"] == ["a2"]


def test_record_accepted_chains_renumbers_across_tool_calls() -> None:
    token = new_graph_collector()
    try:
        record_accepted_chains(
            [
                {"chain_id": "a1", "edges": []},
                {"chain_id": "a2", "edges": []},
            ]
        )
        record_accepted_chains(
            [
                {"chain_id": "a1", "edges": []},
                {"chain_id": "a2", "edges": []},
                {"chain_id": "a3", "edges": []},
            ]
        )
        ids = [c["chain_id"] for c in (current_graph_collector() or [])]
        assert ids == ["a1", "a2", "a3", "a4", "a5"]
        assert [chain_unit_index(c, 0) for c in (current_graph_collector() or [])] == [
            1,
            2,
            3,
            4,
            5,
        ]
    finally:
        reset_graph_collector(token)


def test_second_tool_batch_returns_continued_unit_ids() -> None:
    token = new_graph_collector()
    try:
        first = record_accepted_chains(
            [{"chain_id": "a1", "text": "UNIT a1\none"}, {"chain_id": "a2", "text": "two"}]
        )
        second = record_accepted_chains(
            [{"chain_id": "a1", "text": "UNIT a1\nthree"}]
        )
        assert [c["chain_id"] for c in first] == ["a1", "a2"]
        assert [c["chain_id"] for c in second] == ["a3"]
        assert chain_unit_index(second[0], 0) == 3
    finally:
        reset_graph_collector(token)


def test_node_caption_never_uses_element_id_slice() -> None:
    edge = _edge("4:abcdef12-deadbeef:184", start="", end="")
    edge["start"] = ""
    edge["end"] = ""
    edge["start_id"] = "4:abcdef12-deadbeef:1"
    edge["end_id"] = "4:abcdef12-deadbeef:2"
    views = build_chain_views([{"chain_id": "a1", "score": 0.1, "edges": [edge], "fans": {}}])
    captions = [node["caption"] for node in views[0]["nodes"]]
    assert captions == ["", ""]
    assert all(not cap.startswith("Node ") for cap in captions)


def test_later_node_name_replaces_empty_caption() -> None:
    nameless = _edge("5:1", start="A", end="B")
    nameless["start"] = ""
    named = _edge("5:2", start="C", end="A")
    named["end"] = "Lactobacillus"
    named["end_id"] = nameless["start_id"]
    views = build_chain_views(
        [{"chain_id": "a1", "score": 0.2, "edges": [nameless, named], "fans": {}}]
    )
    by_id = {node["id"]: node for node in views[0]["nodes"]}
    assert by_id[nameless["start_id"]]["caption"] == "Lactobacillus"


def test_edge_without_element_id_keeps_stable_id() -> None:
    edge = _edge("5:1")
    edge["element_id"] = ""
    views = build_chain_views([{"chain_id": "a1", "score": 0.3, "edges": [edge], "fans": {}}])
    assert views[0]["edges"][0]["id"] == "ek:key-5:1"
    assert views[0]["edges"][0]["from_name"] == "A"
    assert views[0]["edges"][0]["to_name"] == "B"
    assert views[0]["edges"][0]["from_group"] == "Microbe"
    assert views[0]["edges"][0]["to_group"] == "Metabolite"


def test_missing_confidence_stays_none() -> None:
    edge = _edge("5:1")
    edge["confidence"] = None
    views = build_chain_views([{"chain_id": "a1", "score": 0.4, "edges": [edge], "fans": {}}])
    assert views[0]["edges"][0]["properties"]["confidence"] is None
    assert "edge_key" not in views[0]["edges"][0]["properties"]
    assert "hub_id" not in views[0]["edges"][0]["properties"]
    assert "run_id" not in views[0]["edges"][0]["properties"]


def test_hub_name_from_fan_hub_names() -> None:
    spine = _edge("5:1")
    fan = _edge("5:2", start="A", end="C")
    chain = {
        "chain_id": "a1",
        "score": 0.5,
        "edges": [spine],
        "fans": {"node-A": [fan]},
        "fan_hub_names": {"node-A": "Microbe: A"},
    }
    views = build_chain_views([chain])
    fan_edge = next(edge for edge in views[0]["edges"] if edge["role"] == "fan")
    assert fan_edge["hub_name"] == "Microbe: A"


def test_merge_prefers_nonempty_evidence() -> None:
    empty = _edge("5:1")
    empty["evidence"] = ""
    filled = _edge("5:1")
    filled["evidence"] = "full quote from paper"
    views = build_chain_views(
        [
            {"chain_id": "a1", "score": 0.9, "edges": [empty], "fans": {}},
            {"chain_id": "a2", "score": 0.8, "edges": [filled], "fans": {}},
        ]
    )
    merged = merge_views(views)
    spine = next(edge for edge in merged["edges"] if edge["id"] == "5:1")
    assert spine["properties"]["evidence"] == "full quote from paper"
    assert spine["from_name"] == "A"

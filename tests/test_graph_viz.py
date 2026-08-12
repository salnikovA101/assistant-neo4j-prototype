from server.algorithm.cypher.edges import FETCH_VIZ_BY_EDGE_IDS
from server.algorithm.models import EdgeRecord
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

"""Tests for session source citations (registry, remap, render)."""

from __future__ import annotations

from server.tools.source_registry import (
    SourceRegistry,
    collect_source_files,
    extract_cited_source_files,
    filter_chains_by_source_files,
    remap_filenames_to_source_ids,
    render_citations,
)


def test_register_stable_and_clear():
    reg = SourceRegistry()
    assert reg.register("a.pdf") == 1
    assert reg.register("b.pdf") == 2
    assert reg.register("a.pdf") == 1
    assert reg.register("  ") is None
    assert reg.resolve(1) == "a.pdf"
    assert reg.resolve(2) == "b.pdf"
    reg.clear()
    assert reg.resolve(1) is None
    assert reg.register("a.pdf") == 1


def test_remap_filenames_longest_first():
    reg = SourceRegistry()
    short = "paper.pdf"
    long = "long_paper.pdf"
    reg.register(short)
    reg.register(long)
    text = f'A -[REL: "ev"]-> B  ({long}; conf=0.9)\nC -> D  ({short}; conf=0.5)'
    out = remap_filenames_to_source_ids(text, reg)
    assert long not in out
    assert f"(source:2; conf=0.9)" in out
    assert f"(source:1; conf=0.5)" in out


def test_remap_from_accepted_edges():
    reg = SourceRegistry()
    accepted = [
        {
            "text": (
                'UNIT c1  (score=1.0)\n'
                "SPINE:\n"
                '  Microbe: A -[PRODUCES: "x"]-> Metabolite: B  '
                "(Hashim et al. Anthocyanins.pdf; conf=0.88)"
            ),
            "edges": [
                {"source_file": "Hashim et al. Anthocyanins.pdf"},
            ],
            "fans": {},
        }
    ]
    for sf in collect_source_files(accepted):
        reg.register(sf)
    out = remap_filenames_to_source_ids(accepted[0]["text"], reg)
    assert "source:1" in out
    assert "Hashim et al. Anthocyanins.pdf" not in out
    assert reg.resolve(1) == "Hashim et al. Anthocyanins.pdf"


def test_render_citations_groups_and_bibliography():
    reg = SourceRegistry()
    reg.register("one.pdf")
    reg.register("two.pdf")
    raw = (
        "Факт A (source:1). Факт B (source:1; source:2).\n\n"
        "### Источники\n"
        "[1] wrong.pdf\n"
        "[9] invent.pdf\n"
    )
    display = render_citations(raw, reg)
    assert "Факт A [1]." in display
    assert "Факт B [1][2]." in display
    assert "wrong.pdf" not in display
    assert "invent.pdf" not in display
    assert "### Источники" in display
    assert "[1] one.pdf" in display
    assert "[2] two.pdf" in display


def test_render_dense_from_one_for_user():
    """Session ids may be 7,9; user-facing markers must be dense 1,2,…"""
    reg = SourceRegistry()
    for name in [f"f{i}.pdf" for i in range(1, 10)]:
        reg.register(name)
    raw = "A (source:7). B (source:9). C (source:7)."
    display = render_citations(raw, reg)
    assert "A [1]." in display
    assert "B [2]." in display
    assert "C [1]." in display
    assert "[7]" not in display
    assert "[9]" not in display
    assert "[1] f7.pdf" in display
    assert "[2] f9.pdf" in display
    biblio = display.split("### Источники")[-1]
    assert "[3]" not in biblio


def test_render_drops_unknown_ids():
    reg = SourceRegistry()
    reg.register("only.pdf")
    display = render_citations("Known (source:1). Fake (source:99).", reg)
    assert "[1]" in display
    assert "[99]" not in display
    assert "source:99" not in display
    assert "[1] only.pdf" in display
    biblio = display.split("### Источники")[-1]
    assert "99" not in biblio


def test_collect_source_files_from_fans():
    accepted = [
        {
            "edges": [{"source_file": "a.pdf"}],
            "fans": {"h1": [{"source_file": "b.pdf"}, {"source_file": "a.pdf"}]},
        }
    ]
    files = collect_source_files(accepted)
    assert files == ["a.pdf", "b.pdf"]


def test_extract_cited_source_files_order_and_unknown():
    reg = SourceRegistry()
    reg.register("one.pdf")
    reg.register("two.pdf")
    raw = "A (source:2). B (source:1; source:2). Fake (source:99)."
    assert extract_cited_source_files(raw, reg) == ["two.pdf", "one.pdf"]


def test_extract_cited_empty_when_no_citations():
    reg = SourceRegistry()
    reg.register("one.pdf")
    assert extract_cited_source_files("Нет цитат.", reg) == []


def test_filter_chains_by_source_files_edges_and_fans():
    chains = [
        {
            "chain_id": "a1",
            "edges": [{"source_file": "keep.pdf"}],
            "fans": {},
        },
        {
            "chain_id": "a2",
            "edges": [{"source_file": "other.pdf"}],
            "fans": {},
        },
        {
            "chain_id": "a3",
            "edges": [],
            "fans": {"h": [{"source_file": "keep.pdf"}]},
        },
    ]
    kept = filter_chains_by_source_files(chains, ["keep.pdf"])
    assert [c["chain_id"] for c in kept] == ["a1", "a3"]


def test_filter_chains_empty_cited_returns_empty():
    chains = [{"chain_id": "a1", "edges": [{"source_file": "a.pdf"}], "fans": {}}]
    assert filter_chains_by_source_files(chains, []) == []
    assert filter_chains_by_source_files(chains, ["  "]) == []

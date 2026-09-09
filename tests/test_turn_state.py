"""Per-turn invariants: search depth from the UI and the ask_subgraph budget."""

from __future__ import annotations

import pytest

from server.core.turn_state import (
    DEFAULT_SEARCH_DEPTH,
    bind_turn,
    current_turn,
    parse_search_depth,
    remember_subquestions,
    search_depth,
    searches_state,
    seen_subquestions,
    subquestion_key,
    take_search_slot,
)
from server.tools.subgraph_search import (
    MAX_SUBQUESTIONS,
    NO_RESULTS,
    TOOL_ERROR,
    SubgraphSearchAgent,
    normalize_subquestions,
)


def test_parse_search_depth_accepts_only_known_levels():
    assert parse_search_depth("low") == "low"
    assert parse_search_depth(" HIGH ") == "high"
    assert parse_search_depth("xhigh") is None
    assert parse_search_depth("") is None
    assert parse_search_depth(None) is None
    assert parse_search_depth(3) is None


def test_depth_defaults_outside_a_turn():
    assert current_turn() is None
    assert search_depth() == DEFAULT_SEARCH_DEPTH
    assert take_search_slot() is False


def test_bind_turn_exposes_depth_and_resets():
    with bind_turn("high", max_searches=2) as turn:
        assert turn.search_depth == "high"
        assert search_depth() == "high"
    assert current_turn() is None


def test_bind_turn_falls_back_on_bad_depth():
    with bind_turn("deepest", max_searches=2):
        assert search_depth() == DEFAULT_SEARCH_DEPTH


def test_bind_turn_can_start_with_spent_search_budget():
    with bind_turn("low", max_searches=1, context={"searches_used": 1}):
        assert take_search_slot() is False
    with bind_turn("low", max_searches=1, context={"searches_used": 0}):
        assert take_search_slot() is True
    with bind_turn("low", max_searches=2):
        assert take_search_slot() is True
        assert take_search_slot() is True
        assert take_search_slot() is False
        assert searches_state() == (2, 2)


def test_subquestion_memory_is_turn_scoped():
    with bind_turn("low", max_searches=2):
        remember_subquestions(["Kefiran inhibits Listeria."])
        assert subquestion_key("  kefiran   inhibits listeria?  ") in seen_subquestions()
    with bind_turn("low", max_searches=2):
        assert seen_subquestions() == set()


def test_normalize_drops_cyrillic_and_duplicates():
    clean, problems = normalize_subquestions(
        [
            "Kefiran inhibits Listeria in cheese.",
            "kefiran inhibits listeria in cheese?",
            "закваски для творога",
            "   ",
        ]
    )
    assert clean == ["Kefiran inhibits Listeria in cheese."]
    assert any("not English" in p for p in problems)
    assert any("duplicate" in p for p in problems)


def test_normalize_drops_repeats_of_earlier_call():
    seen = {subquestion_key("Kefiran inhibits Listeria.")}
    clean, problems = normalize_subquestions(
        ["Kefiran inhibits Listeria.", "Kefiran is an exopolysaccharide."],
        seen,
    )
    assert clean == ["Kefiran is an exopolysaccharide."]
    assert any("already searched" in p for p in problems)


def test_normalize_rejects_string_subquestions():
    clean, problems = normalize_subquestions("not-a-list")
    assert clean == []
    assert any("JSON array" in p for p in problems)


def test_normalize_caps_at_six():
    clean, problems = normalize_subquestions([f"Statement number {i}." for i in range(9)])
    assert len(clean) == MAX_SUBQUESTIONS
    assert any("only the first" in p for p in problems)


@pytest.mark.asyncio
async def test_query_rejects_unusable_input_without_spending_budget():
    agent = SubgraphSearchAgent()
    with bind_turn("medium", max_searches=2, context={"run_id": "corpus-test"}):
        out = await agent.query(["закваски для творога"])
        assert out.startswith(TOOL_ERROR)
        assert "not English" in out
        assert searches_state() == (0, 2)


@pytest.mark.asyncio
async def test_query_refuses_third_search_in_one_turn(monkeypatch):
    async def fake_run(driver, **kwargs):
        return {"accepted": []}

    monkeypatch.setattr("server.algorithm.pipeline.run", fake_run)
    monkeypatch.setattr("server.tools.subgraph_search.get_driver", lambda: object())

    agent = SubgraphSearchAgent()
    with bind_turn("medium", max_searches=2, context={"run_id": "corpus-test"}):
        first = await agent.query(["Lactic acid bacteria acidify milk."])
        second = await agent.query(["Anthocyanin films change colour."])
        third = await agent.query(["Chitosan films carry indicator dyes."])

    assert first.startswith(NO_RESULTS)
    assert second.startswith(NO_RESULTS)
    assert third.startswith(TOOL_ERROR)
    assert "budget" in third


@pytest.mark.asyncio
async def test_query_passes_ui_depth_to_the_pipeline(monkeypatch):
    seen: dict[str, object] = {}

    async def fake_run(driver, **kwargs):
        seen.update(kwargs)
        return {"accepted": []}

    monkeypatch.setattr("server.algorithm.pipeline.run", fake_run)
    monkeypatch.setattr("server.tools.subgraph_search.get_driver", lambda: object())

    agent = SubgraphSearchAgent()
    with bind_turn("high", max_searches=2, context={"run_id": "corpus-test"}):
        await agent.query(["Lactic acid bacteria acidify milk."])

    assert seen["effort"] == "high"
    assert [sq["text"] for sq in seen["subquestions"]] == [
        "Lactic acid bacteria acidify milk."
    ]
    assert seen["params"].run_id == "corpus-test"


def test_normalize_preserves_english_questions():
    question = "How does fermentation temperature affect syneresis in kefir?"
    clean, problems = normalize_subquestions([f"  {question}  ", question[:-1]])
    assert clean == [question]
    assert any("duplicate" in problem for problem in problems)

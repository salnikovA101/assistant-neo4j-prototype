from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from server.core.app_store import AppStore
from server.core.pipeline import _graph_done_payload
from server.core.sessions import SessionStore
from server.core.turn_state import bind_turn, take_search_slot
from server.llm.providers.openai_provider import OpenAIProvider
from server.llm.stream_events import StreamEvent
from server.utils.config import OpenAIProfile


@pytest.mark.asyncio
async def test_source_ids_are_stable_across_branches(tmp_path):
    store = AppStore(str(tmp_path / "sources.db"))
    await store.open()
    try:
        user = await store.create_user("source-user", "long source user password")
        conversation = await store.create_conversation(user.id)
        first = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "first",
        )
        await store.finish_turn(
            conversation["id"],
            first["assistantMessageId"],
            text="answer [1]",
            status="done",
            payload={},
            raw_text="answer (source:1)",
            sources=[(1, "paper.pdf")],
        )
        first_answer = (await store.get_conversation(user.id, conversation["id"]))["headCheckpointId"]
        fork = await store.create_fork(user.id, conversation["id"], first_answer)
        sibling = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            fork["id"],
            "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            "second",
            mode="auto",
        )
        snapshot = await store.merge_conversation_sources(
            conversation["id"], ["other.pdf"]
        )
        await store.finish_turn(
            conversation["id"],
            sibling["assistantMessageId"],
            text="other [2]",
            status="done",
            payload={},
            raw_text="other (source:2)",
            sources=snapshot,
        )
        kept = await store.conversation_source_snapshot(conversation["id"])
        assert kept[0] == (1, "paper.pdf")
        assert (2, "other.pdf") in kept
        main = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
            "third",
            base_checkpoint_id=first_answer,
            fork_if_needed=True,
        )
        await store.finish_turn(
            conversation["id"],
            main["assistantMessageId"],
            text="again [1]",
            status="done",
            payload={},
            raw_text="again (source:1)",
            sources=[(99, "paper.pdf")],
        )
        assert await store.conversation_source_snapshot(conversation["id"]) == kept
        later = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            "no new sources",
            base_checkpoint_id=(await store.get_conversation(user.id, conversation["id"]))["headCheckpointId"],
            fork_if_needed=True,
        )
        await store.finish_turn(
            conversation["id"],
            later["assistantMessageId"],
            text="plain",
            status="done",
            payload={},
            sources=[],
        )
        assert await store.conversation_source_snapshot(conversation["id"]) == kept
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_rollback_after_record_units_drops_turn_units(tmp_path):
    store = AppStore(str(tmp_path / "rollback.db"))
    await store.open()
    try:
        user = await store.create_user("roll-user", "long rollback user password")
        conversation = await store.create_conversation(user.id)
        first = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "11111111-1111-4111-8111-111111111111",
            "kept question",
        )
        kept_checkpoint = await store.finish_turn(
            conversation["id"],
            first["assistantMessageId"],
            text="kept answer",
            status="done",
            payload={},
            raw_text="kept answer",
        )
        failed = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "22222222-2222-4222-8222-222222222222",
            "failed question",
        )
        recorded = await store.record_units(
            conversation["id"],
            failed["userCheckpointId"],
            [{
                "edge_keys": ["edge-fail"],
                "spine_evidence_seq": ["quote-fail"],
                "text": "UNIT 1\nA —REL→ B",
            }],
        )
        assert recorded[0]["unit_no"] == 1
        rolled = await store.rollback_turn(
            conversation["id"],
            failed["assistantMessageId"],
            reason="error",
            message="boom",
        )
        assert rolled["text"] == "failed question"
        detail = await store.get_conversation(user.id, conversation["id"])
        assert [item["text"] for item in detail["messages"]] == ["kept question", "kept answer"]
        assert detail["headCheckpointId"] == kept_checkpoint
        assert await store.checkpoint_chains(user.id, kept_checkpoint) == []
        assert detail["turnFailures"][0]["reason"] == "error"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_branch_unit_numbers_continue_on_fork_and_reset_on_sibling(tmp_path):
    store = AppStore(str(tmp_path / "units.db"))
    await store.open()
    try:
        user = await store.create_user("unit-user", "long unit user password")
        conversation = await store.create_conversation(user.id)
        first = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "33333333-3333-4333-8333-333333333333",
            "q1",
        )
        recorded = await store.record_units(
            conversation["id"],
            first["userCheckpointId"],
            [{
                "edge_keys": ["e1"],
                "spine_evidence_seq": ["q1"],
                "text": "UNIT 1\nA —REL→ B",
            }],
        )
        first_answer = await store.finish_turn(
            conversation["id"],
            first["assistantMessageId"],
            text="a1",
            status="done",
            payload={},
            graph_chains=recorded,
        )
        second = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "44444444-4444-4444-8444-444444444444",
            "q2",
        )
        recorded_two = await store.record_units(
            conversation["id"],
            second["userCheckpointId"],
            [{
                "edge_keys": ["e2"],
                "spine_evidence_seq": ["q2"],
                "text": "UNIT 2\nC —REL→ D",
            }],
        )
        second_answer = await store.finish_turn(
            conversation["id"],
            second["assistantMessageId"],
            text="a2",
            status="done",
            payload={},
            graph_chains=recorded_two,
        )
        context = await store.checkpoint_chains(user.id, second_answer)
        assert [item["unit_no"] for item in context] == [1, 2]
        fork = await store.create_fork(user.id, conversation["id"], second_answer)
        continued = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            fork["id"],
            "55555555-5555-4555-8555-555555555555",
            "q3",
        )
        next_units = await store.record_units(
            conversation["id"],
            continued["userCheckpointId"],
            [{
                "edge_keys": ["e3"],
                "spine_evidence_seq": ["q3"],
                "text": "UNIT 3\nE —REL→ F",
            }],
        )
        assert next_units[0]["unit_no"] == 3

        empty = await store.create_conversation(user.id)
        blank = await store.begin_branch_turn(
            user.id,
            empty["id"],
            empty["activeBranchId"],
            "66666666-6666-4666-8666-666666666666",
            "blank",
        )
        blank_answer = await store.finish_turn(
            empty["id"],
            blank["assistantMessageId"],
            text="blank answer",
            status="done",
            payload={},
        )
        fork_a = await store.create_fork(user.id, empty["id"], blank_answer)
        fork_b = await store.create_fork(user.id, empty["id"], blank_answer)
        turn_a = await store.begin_branch_turn(
            user.id, empty["id"], fork_a["id"],
            "77777777-7777-4777-8777-777777777777", "a",
        )
        turn_b = await store.begin_branch_turn(
            user.id, empty["id"], fork_b["id"],
            "88888888-8888-4888-8888-888888888888", "b",
        )
        units_a = await store.record_units(
            empty["id"],
            turn_a["userCheckpointId"],
            [{"edge_keys": ["ea"], "spine_evidence_seq": ["qa"], "text": "UNIT 1\nA —REL→ B"}],
        )
        units_b = await store.record_units(
            empty["id"],
            turn_b["userCheckpointId"],
            [{"edge_keys": ["eb"], "spine_evidence_seq": ["qb"], "text": "UNIT 1\nC —REL→ D"}],
        )
        assert units_a[0]["unit_no"] == 1
        assert units_b[0]["unit_no"] == 1
        assert first_answer
    finally:
        await store.close()


def test_graph_keeps_units_without_citations():
    payload = _graph_done_payload(
        [{"chain_id": "u1", "text": "UNIT 1", "edges": []}],
        [],
    )
    assert payload["graph_chain_count"] == 1
    assert payload["graph_chains"]
    assert payload["open_graph"] is False
    cited = _graph_done_payload(payload["graph_chains"], ["paper.pdf"])
    assert cited["open_graph"] is True


def test_hydrate_refreshes_sources_on_every_call():
    store = SessionStore()
    first = store.hydrate("sid", 6, [], [(1, "a.pdf")])
    assert first.sources.snapshot() == [(1, "a.pdf")]
    again = store.hydrate("sid", 6, [], [(1, "a.pdf"), (2, "b.pdf")])
    assert again is first
    assert again.sources.snapshot() == [(1, "a.pdf"), (2, "b.pdf")]


def test_bind_turn_can_start_with_spent_search_budget():
    with bind_turn("low", max_searches=1, context={"searches_used": 1}):
        assert take_search_slot() is False
    with bind_turn("low", max_searches=1, context={"searches_used": 0}):
        assert take_search_slot() is True


@pytest.mark.asyncio
async def test_resume_messages_skip_extra_user_and_keep_tool_result():
    profile = OpenAIProfile(
        model="test-model",
        base_url="http://localhost",
        api_key="x",
        max_turns=1,
    )
    provider = OpenAIProvider(profile)
    captured: list[list] = []

    class _Delta:
        def __init__(self, content="ok"):
            self.content = content
            self.model_extra = {}

        def model_dump(self, exclude_none=False):
            return {"content": self.content}

    class _Choice:
        def __init__(self, delta):
            self.delta = delta

    class _Chunk:
        def __init__(self, delta=None):
            self.choices = [_Choice(delta)] if delta is not None else []
            self.usage = None

    async def _aiter(items):
        for item in items:
            yield item

    async def fake_create(**kwargs):
        captured.append(kwargs["messages"])
        return _aiter([_Chunk(_Delta("Final"))])

    provider.client.chat.completions.create = AsyncMock(side_effect=fake_create)
    events: list[StreamEvent] = []
    resume = [
        {"role": "user", "content": [{"type": "text", "text": "question"}]},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call-1",
                "type": "function",
                "function": {"name": "advance_research", "arguments": "{}"},
            }],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "UNIT [1]\nevidence"},
    ]
    async for event in provider.generate_response_stream(
        user_text="must not be appended",
        prompt="sys",
        history=[],
        resume_messages=resume,
    ):
        events.append(event)
    assert events[-1].type == "done"
    roles = [item["role"] for item in captured[0]]
    assert roles == ["system", "user", "assistant", "tool"]
    assert captured[0][1]["content"][0]["text"] == "question"
    assert captured[0][3]["content"] == "UNIT [1]\nevidence"
@pytest.mark.asyncio
async def test_generate_approved_resumes_as_tool_loop(monkeypatch):
    from server.llm.manager import LLMManager

    captured: dict = {}

    async def fake(self, user_text, **kwargs):
        captured["user_text"] = user_text
        captured["ctx"] = kwargs["turn_context"]
        yield StreamEvent("done", {"final_content": "ok", "cited_source_files": []})

    monkeypatch.setattr(LLMManager, "generate_response_stream", fake)
    manager = LLMManager.__new__(LLMManager)
    events = [
        event
        async for event in LLMManager.generate_approved_response_stream(
            manager,
            user_text="question",
            evidence="UNIT [1]\nevidence",
            tool_call={
                "id": "c1",
                "name": "advance_research",
                "arguments": {"new_subquestions": ["a"]},
            },
            turn_context={"model_user_text": "question"},
        )
    ]
    assert events[-1].type == "done"
    ctx = captured["ctx"]
    assert captured["user_text"] == "question"
    assert [item["role"] for item in ctx["resume_messages"]] == ["user", "assistant", "tool"]
    assert ctx["resume_messages"][2]["content"] == "UNIT [1]\nevidence"
    assert ctx["searches_used"] == 1
    assert "Ниже результат" not in captured["user_text"]

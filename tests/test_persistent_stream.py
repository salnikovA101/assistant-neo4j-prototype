from __future__ import annotations

from types import SimpleNamespace

import pytest

from server.core.app import _conversation_session_key, _persistent_stream, _revised_approval_stream
from server.core.app_store import AppStore
from server.core.http_api import TextProcessBody
from server.core.sessions import session_store
from server.llm.stream_events import StreamEvent
from server.utils.config import AppConfig


class FakePipeline:
    def __init__(self) -> None:
        self.config = AppConfig()

    async def process_text_stream(self, *_args, **_kwargs):
        yield StreamEvent("thinking", {"delta": "check"})
        yield StreamEvent("content", {"delta": "visible"})
        yield StreamEvent(
            "done",
            {
                "final_content": "visible [1]",
                "graph_run_id": "gr_test",
                "graph_chain_count": 1,
                "_raw_content": "visible (source:1)",
                "_history_tool_messages": [{"role": "tool", "content": "receipt"}],
                "_graph_chains": [{"chain_id": "a1", "edges": []}],
            },
        )


class FakeAbortedPipeline(FakePipeline):
    async def process_text_stream(self, *_args, **_kwargs):
        yield StreamEvent("content", {"delta": "partial"})


class FakeStagedPipeline(FakePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False

    async def process_text_stream(self, *_args, **_kwargs):
        try:
            yield StreamEvent("thinking", {"delta": "plan first"})
            yield StreamEvent(
                "tool_call",
                {
                    "id": "call-1",
                    "name": "ask_subgraph",
                    "arguments": {"subquestions": ["SQ one"]},
                    "_assistant_replay": {"role": "assistant", "tool_calls": []},
                },
            )
            yield StreamEvent("content", {"delta": "must not run"})
        finally:
            self.closed = True


@pytest.mark.asyncio
async def test_stream_persists_ui_model_context_and_graph_without_leaking_private_sse(tmp_path):
    store = AppStore(str(tmp_path / "stream.db"))
    await store.open()
    try:
        user = await store.create_user("stream-user", "long stream user password")
        conv = await store.create_conversation(user.id)
        session_key = _conversation_session_key(user.id, conv["id"])
        session = session_store.hydrate(session_key, 6, [], [])
        session.sources.register("paper.pdf")
        _, assistant_id = await store.begin_turn(
            user.id,
            conv["id"],
            "55555555-5555-4555-8555-555555555555",
            "question",
        )
        app_state = SimpleNamespace(pipeline=FakePipeline(), app_store=store)
        request = SimpleNamespace(
            app=SimpleNamespace(state=app_state),
            headers={},
        )
        chunks = []
        async for chunk in _persistent_stream(
            request,
            TextProcessBody(text="question"),
            user=user,
            conversation_id=conv["id"],
            session_key=session_key,
            assistant_message_id=assistant_id,
        ):
            chunks.append(chunk)

        joined = "".join(chunks)
        assert "_raw_content" not in joined
        detail = await store.get_conversation(user.id, conv["id"])
        assert detail is not None
        assistant = detail["messages"][-1]
        assert assistant["text"] == "visible [1]"
        assert assistant["thinking"] == "check"
        turns, sources = await store.load_model_context(user.id, conv["id"], 6)
        assert turns[0]["assistant"] == "visible (source:1)"
        assert sources == [(1, "paper.pdf")]
        assert await store.get_graph_run(user.id, "gr_test") == [{"chain_id": "a1", "edges": []}]
    finally:
        session_store.drop(session_key)
        await store.close()


@pytest.mark.asyncio
async def test_stream_without_terminal_event_is_saved_as_aborted(tmp_path):
    store = AppStore(str(tmp_path / "aborted.db"))
    await store.open()
    session_key = ""
    try:
        user = await store.create_user("abort-user", "long abort user password")
        conv = await store.create_conversation(user.id)
        session_key = _conversation_session_key(user.id, conv["id"])
        session_store.hydrate(session_key, 6, [], [])
        _, assistant_id = await store.begin_turn(
            user.id,
            conv["id"],
            "66666666-6666-4666-8666-666666666666",
            "question",
        )
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(pipeline=FakeAbortedPipeline(), app_store=store)),
            headers={},
        )
        async for _ in _persistent_stream(
            request,
            TextProcessBody(text="question"),
            user=user,
            conversation_id=conv["id"],
            session_key=session_key,
            assistant_message_id=assistant_id,
        ):
            pass
        detail = await store.get_conversation(user.id, conv["id"])
        assert detail is not None
        assert detail["messages"][-1]["status"] == "aborted"
        assert detail["messages"][-1]["text"] == "partial"
    finally:
        if session_key:
            session_store.drop(session_key)
        await store.close()


@pytest.mark.asyncio
async def test_staged_stream_closes_inner_generator_before_waiting_for_approval(tmp_path):
    store = AppStore(str(tmp_path / "staged.db"))
    await store.open()
    session_key = ""
    try:
        user = await store.create_user("staged-user", "long staged user password")
        conversation = await store.create_conversation(user.id)
        session_key = _conversation_session_key(
            user.id, conversation["id"], conversation["activeBranchId"]
        )
        session_store.hydrate(session_key, 6, [], [])
        started = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "88888888-8888-4888-8888-888888888888",
            "question",
            mode="staged",
        )
        pipeline = FakeStagedPipeline()
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(pipeline=pipeline, app_store=store)),
            headers={},
        )
        chunks = [
            chunk
            async for chunk in _persistent_stream(
                request,
                TextProcessBody(text="question", mode="staged"),
                user=user,
                conversation_id=conversation["id"],
                session_key=session_key,
                assistant_message_id=started["assistantMessageId"],
                user_message_id=started["userMessageId"],
                branch_id=conversation["activeBranchId"],
                user_checkpoint_id=started["userCheckpointId"],
                mode="staged",
            )
        ]
        assert pipeline.closed is True
        assert "approval_required" in "".join(chunks)
        detail = await store.get_conversation(user.id, conversation["id"])
        assert detail["messages"][-1]["thinking"] == "plan first"
        assert detail["messages"][-1]["status"] == "waiting_approval"
    finally:
        if session_key:
            session_store.drop(session_key)
        await store.close()


@pytest.mark.asyncio
async def test_revised_approval_persists_both_reasoning_phases(tmp_path):
    store = AppStore(str(tmp_path / "revised.db"))
    await store.open()
    session_key = ""
    try:
        user = await store.create_user("revised-user", "long revised user password")
        conversation = await store.create_conversation(user.id)
        branch_id = conversation["activeBranchId"]
        session_key = _conversation_session_key(user.id, conversation["id"], branch_id)
        session_store.hydrate(session_key, 6, [], [])
        started = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            branch_id,
            "99999999-9999-4999-8999-999999999999",
            "question",
            mode="staged",
        )
        pending = await store.create_pending_approval(
            user.id,
            conversation_id=conversation["id"],
            branch_id=branch_id,
            user_message_id=started["userMessageId"],
            assistant_message_id=started["assistantMessageId"],
            base_checkpoint_id=started["userCheckpointId"],
            tool_call={"id": "old", "name": "ask_subgraph", "arguments": {"subquestions": ["old SQ"]}},
            resume={"text": "question", "thinking": "original reasoning"},
            settings={"mode": "staged"},
        )
        await store.update_assistant_waiting(
            conversation["id"], started["assistantMessageId"], payload={"thinking": "original reasoning"}
        )
        approval = await store.claim_pending_approval(user.id, pending["id"], 1, "revise")
        pipeline = FakeStagedPipeline()
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(pipeline=pipeline, app_store=store)),
            state=SimpleNamespace(account_user=user),
            headers={},
        )
        chunks = [chunk async for chunk in _revised_approval_stream(request, approval, "make it precise")]
        assert pipeline.closed is True
        assert "approval_required" in "".join(chunks)
        detail = await store.get_conversation(user.id, conversation["id"])
        waiting = detail["messages"][-1]
        assert waiting["thinking"] == "original reasoning\n\nplan first"
        assert [step["kind"] for step in waiting["steps"]] == ["think", "think", "tool"]
        assert detail["pendingApproval"]["revision"] == 2
    finally:
        if session_key:
            session_store.drop(session_key)
        await store.close()

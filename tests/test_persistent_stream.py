from __future__ import annotations

from types import SimpleNamespace

import pytest

from server.core.app import _conversation_session_key, _persistent_stream
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

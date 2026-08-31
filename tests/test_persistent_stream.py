from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from server.core.app import (
    _conversation_session_key,
    _guarded_turn_stream,
    _persistent_stream,
    _revised_approval_stream,
    process_text_stream,
)
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


class FakeNamedPipeline(FakePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.config.llm.current_profile = "auto"
        self.config.llm.ui_profiles = ["auto"]

    async def process_text_stream(self, *_args, **_kwargs):
        yield StreamEvent("model", {"id": "qwen38_flash", "label": "Qwen 3.8 Flash"})
        yield StreamEvent("content", {"delta": "ok"})
        yield StreamEvent(
            "done",
            {
                "final_content": "ok",
                "modelId": "qwen38_flash",
                "modelLabel": "Qwen 3.8 Flash",
            },
        )


class FakeAbortedPipeline(FakePipeline):
    async def process_text_stream(self, *_args, **_kwargs):
        yield StreamEvent("content", {"delta": "partial"})


class FakeErrorPipeline(FakePipeline):
    async def process_text_stream(self, *_args, **_kwargs):
        yield StreamEvent("error", {"message": "boom"})


class FakeStagedPipeline(FakePipeline):
    def __init__(self) -> None:
        super().__init__()
        self.closed = False
        self.tool_arguments = {
            "open_sq_refs": [],
            "new_subquestions": ["SQ one"],
        }

    async def process_text_stream(self, *_args, **_kwargs):
        try:
            yield StreamEvent("thinking", {"delta": "plan first"})
            yield StreamEvent(
                "tool_call",
                {
                    "id": "call-1",
                    "name": "advance_research",
                    "arguments": self.tool_arguments,
                    "_assistant_replay": {"role": "assistant", "tool_calls": []},
                },
            )
            yield StreamEvent("content", {"delta": "must not run"})
        finally:
            self.closed = True


class FakeStagedPipelineOmittedRefs(FakeStagedPipeline):
    def __init__(self) -> None:
        super().__init__()
        self.tool_arguments = {"new_subquestions": ["SQ one"]}


class FakeStagedHelpPipeline(FakePipeline):
    async def process_text_stream(self, *_args, **_kwargs):
        yield StreamEvent(
            "tool_call",
            {"id": "help-1", "name": "get_service_guide", "arguments": {}},
        )
        yield StreamEvent(
            "tool_result",
            {
                "id": "help-1",
                "name": "get_service_guide",
                "ok": True,
                "result": "# Guide",
            },
        )
        yield StreamEvent("content", {"delta": "Вот как пользоваться сервисом."})
        yield StreamEvent(
            "done", {"final_content": "Вот как пользоваться сервисом."}
        )


@pytest.mark.asyncio
async def test_stream_endpoint_rejects_malformed_key_before_store_access():
    request = SimpleNamespace(
        app=SimpleNamespace(
            state=SimpleNamespace(
                pipeline=FakePipeline(),
                app_store=SimpleNamespace(),
            )
        ),
        headers={"X-LLM-Api-Key": "short"},
    )
    with pytest.raises(HTTPException) as raised:
        await process_text_stream(request, TextProcessBody(text="question"))
    assert raised.value.status_code == 400
    assert "Некорректный ключ LLM" in str(raised.value.detail)


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
        assert '"checkpoint_id"' in joined
        assert '"mode": "auto"' in joined
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
async def test_stream_persists_resolved_model_label(tmp_path):
    store = AppStore(str(tmp_path / "model.db"))
    await store.open()
    session_key = ""
    try:
        user = await store.create_user("model-user", "long model user password")
        conv = await store.create_conversation(user.id)
        session_key = _conversation_session_key(user.id, conv["id"])
        session_store.hydrate(session_key, 6, [], [])
        _, assistant_id = await store.begin_turn(
            user.id,
            conv["id"],
            "77777777-7777-4777-8777-777777777777",
            "question",
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(pipeline=FakeNamedPipeline(), app_store=store)
            ),
            headers={},
        )
        chunks = [
            chunk
            async for chunk in _persistent_stream(
                request,
                TextProcessBody(text="question", profile="auto"),
                user=user,
                conversation_id=conv["id"],
                session_key=session_key,
                assistant_message_id=assistant_id,
            )
        ]
        joined = "".join(chunks)
        assert "event: model" in joined
        assert "Qwen 3.8 Flash" in joined
        assert '"id": "auto"' not in joined
        detail = await store.get_conversation(user.id, conv["id"])
        assert detail is not None
        assistant = detail["messages"][-1]
        assert assistant["modelId"] == "qwen38_flash"
        assert assistant["modelLabel"] == "Qwen 3.8 Flash"
    finally:
        if session_key:
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
        chunks = [
            chunk
            async for chunk in _persistent_stream(
                request,
                TextProcessBody(text="question"),
                user=user,
                conversation_id=conv["id"],
                session_key=session_key,
                assistant_message_id=assistant_id,
            )
        ]
        detail = await store.get_conversation(user.id, conv["id"])
        assert detail is not None
        assert detail["messages"] == []
        assert detail["turnFailures"]
        assert detail["turnFailures"][0]["reason"] == "aborted"
        assert detail["turnFailures"][0]["text"] == "question"
        assert "turn_rolled_back" not in "".join(chunks)
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
        conversation = await store.create_conversation(user.id, mode="staged")
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
async def test_staged_approval_gate_treats_omitted_open_sq_refs_as_empty(tmp_path):
    store = AppStore(str(tmp_path / "staged-omitted-refs.db"))
    await store.open()
    session_key = ""
    try:
        user = await store.create_user("staged-omit-user", "long staged omit password")
        conversation = await store.create_conversation(user.id, mode="staged")
        session_key = _conversation_session_key(
            user.id, conversation["id"], conversation["activeBranchId"]
        )
        session_store.hydrate(session_key, 6, [], [])
        started = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "question",
            mode="staged",
        )
        pipeline = FakeStagedPipelineOmittedRefs()
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
        assert "approval_required" in "".join(chunks)
        detail = await store.get_conversation(user.id, conversation["id"])
        assert detail["messages"][-1]["status"] == "waiting_approval"
        pending = detail.get("pendingApproval") or {}
        args = (pending.get("toolCall") or {}).get("arguments") or {}
        assert args.get("open_sq_refs") == []
        assert args.get("new_subquestions") == ["SQ one"]
    finally:
        if session_key:
            session_store.drop(session_key)
        await store.close()


@pytest.mark.asyncio
async def test_staged_help_tool_does_not_open_search_approval(tmp_path):
    store = AppStore(str(tmp_path / "staged-help.db"))
    await store.open()
    session_key = ""
    try:
        user = await store.create_user("staged-help-user", "long staged help password")
        conversation = await store.create_conversation(user.id, mode="staged")
        branch_id = conversation["activeBranchId"]
        session_key = _conversation_session_key(user.id, conversation["id"], branch_id)
        session_store.hydrate(session_key, 6, [], [])
        started = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            branch_id,
            "bbbbbbbb-bbbb-4bbb-8bbb-bbbbbbbbbbbb",
            "Как пользоваться сервисом?",
            mode="staged",
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(
                    pipeline=FakeStagedHelpPipeline(), app_store=store
                )
            ),
            headers={},
        )
        chunks = [
            chunk
            async for chunk in _persistent_stream(
                request,
                TextProcessBody(text="Как пользоваться сервисом?", mode="staged"),
                user=user,
                conversation_id=conversation["id"],
                session_key=session_key,
                assistant_message_id=started["assistantMessageId"],
                user_message_id=started["userMessageId"],
                branch_id=branch_id,
                user_checkpoint_id=started["userCheckpointId"],
                mode="staged",
            )
        ]
        joined = "".join(chunks)
        assert "get_service_guide" in joined
        assert "approval_required" not in joined
        assert "open_sq_refs" not in joined
        detail = await store.get_conversation(user.id, conversation["id"])
        assert detail is not None
        assert detail["messages"][-1]["status"] == "done"
        assert detail["messages"][-1]["text"] == "Вот как пользоваться сервисом."
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
        conversation = await store.create_conversation(user.id, mode="staged")
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
            tool_call={
                "id": "old",
                "name": "advance_research",
                "arguments": {"open_sq_refs": [], "new_subquestions": ["old SQ"]},
            },
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
        joined = "".join(chunks)
        assert pipeline.closed is True
        assert "approval_required" in joined
        assert "make it precise" in joined
        assert "Пользователь отклонил" in joined
        detail = await store.get_conversation(user.id, conversation["id"])
        waiting = detail["messages"][-1]
        assert waiting["thinking"] == "original reasoning\n\nplan first"
        assert [step["kind"] for step in waiting["steps"]] == ["think", "tool"]
        assert detail["pendingApproval"]["revision"] == 1
    finally:
        if session_key:
            session_store.drop(session_key)
        await store.close()


@pytest.mark.asyncio
async def test_stream_error_rolls_back_and_returns_composer_text(tmp_path):
    store = AppStore(str(tmp_path / "error-rollback.db"))
    await store.open()
    session_key = ""
    try:
        user = await store.create_user("error-user", "long error user password")
        conv = await store.create_conversation(user.id)
        session_key = _conversation_session_key(user.id, conv["id"])
        session_store.hydrate(session_key, 6, [], [])
        _, assistant_id = await store.begin_turn(
            user.id,
            conv["id"],
            "77777777-7777-4777-8777-777777777777",
            "restore me",
        )
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(pipeline=FakeErrorPipeline(), app_store=store)),
            headers={},
        )
        chunks = [
            chunk
            async for chunk in _persistent_stream(
                request,
                TextProcessBody(text="restore me"),
                user=user,
                conversation_id=conv["id"],
                session_key=session_key,
                assistant_message_id=assistant_id,
            )
        ]
        joined = "".join(chunks)
        assert "turn_rolled_back" in joined
        assert "restore me" in joined
        detail = await store.get_conversation(user.id, conv["id"])
        assert detail is not None
        assert detail["messages"] == []
        assert detail["turnFailures"][0]["reason"] == "error"
        assert detail["turnFailures"][0]["text"] == "restore me"
    finally:
        if session_key:
            session_store.drop(session_key)
        await store.close()


@pytest.mark.asyncio
async def test_malformed_key_cannot_leave_branch_with_active_turn(tmp_path):
    store = AppStore(str(tmp_path / "malformed-key-rollback.db"))
    await store.open()
    session_key = ""
    try:
        user = await store.create_user("bad-key-user", "long bad key user password")
        conversation = await store.create_conversation(user.id)
        branch_id = conversation["activeBranchId"]
        session_key = _conversation_session_key(user.id, conversation["id"], branch_id)
        session_store.hydrate(session_key, 6, [], [])
        started = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            branch_id,
            "cccccccc-cccc-4ccc-8ccc-cccccccccccc",
            "верни меня в поле ввода",
            mode="auto",
        )
        request = SimpleNamespace(
            app=SimpleNamespace(
                state=SimpleNamespace(pipeline=FakePipeline(), app_store=store)
            ),
            headers={"X-LLM-Api-Key": "short"},
        )
        source = _persistent_stream(
            request,
            TextProcessBody(text="верни меня в поле ввода"),
            user=user,
            conversation_id=conversation["id"],
            session_key=session_key,
            assistant_message_id=started["assistantMessageId"],
            user_message_id=started["userMessageId"],
            branch_id=branch_id,
            user_checkpoint_id=started["userCheckpointId"],
        )
        chunks = [
            chunk
            async for chunk in _guarded_turn_stream(
                source,
                store=store,
                conversation_id=conversation["id"],
                assistant_message_id=started["assistantMessageId"],
            )
        ]

        joined = "".join(chunks)
        assert "turn_rolled_back" in joined
        assert "верни меня в поле ввода" in joined
        detail = await store.get_conversation(user.id, conversation["id"])
        assert detail is not None
        assert detail["messages"] == []
        assert detail["turnFailures"][0]["reason"] == "error"

        # The same branch must accept a fresh turn immediately after rollback.
        retry = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            branch_id,
            "dddddddd-dddd-4ddd-8ddd-dddddddddddd",
            "повтор",
            mode="auto",
        )
        assert retry["assistantMessageId"]
    finally:
        if session_key:
            session_store.drop(session_key)
        await store.close()

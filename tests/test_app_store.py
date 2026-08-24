from __future__ import annotations

import asyncio
import sqlite3

import pytest

from server.core.app_store import AppStore


@pytest.mark.asyncio
async def test_accounts_sessions_and_revocation(tmp_path):
    store = AppStore(str(tmp_path / "app.db"))
    await store.open()
    try:
        user = await store.create_user("Technologist", "correct horse battery")
        assert user.username == "technologist"
        assert await store.authenticate("TECHNOLOGIST", "correct horse battery") == user
        assert await store.authenticate("technologist", "wrong password") is None

        token = await store.create_session(user.id)
        assert token not in (tmp_path / "app.db").read_bytes().decode("utf-8", errors="ignore")
        assert await store.user_for_session(token) == user

        assert await store.reset_password("technologist", "a different long password")
        assert await store.user_for_session(token) is None
        assert await store.authenticate("technologist", "a different long password") == user
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_conversations_are_isolated_and_context_survives_reopen(tmp_path):
    path = str(tmp_path / "app.db")
    store = AppStore(path)
    await store.open()
    a = await store.create_user("user-a", "password for user a")
    b = await store.create_user("user-b", "password for user b")
    conv = await store.create_conversation(a.id)
    _, assistant_id = await store.begin_turn(a.id, conv["id"], "11111111-1111-4111-8111-111111111111", "Первый вопрос")
    await store.finish_turn(
        conv["id"],
        assistant_id,
        text="Ответ [1]",
        status="done",
        payload={"thinking": "", "steps": []},
        raw_text="Ответ (source:1)",
        tool_messages=[{"role": "tool", "content": "receipt"}],
        sources=[(1, "paper.pdf")],
        graph_run_id="gr_saved",
        graph_chains=[{"chain_id": "a1", "edges": []}],
    )

    assert await store.get_conversation(b.id, conv["id"]) is None
    assert await store.get_graph_run(b.id, "gr_saved") is None
    await store.close()

    reopened = AppStore(path)
    await reopened.open()
    try:
        detail = await reopened.get_conversation(a.id, conv["id"])
        assert detail is not None
        assert [m["text"] for m in detail["messages"]] == ["Первый вопрос", "Ответ [1]"]
        turns, sources = await reopened.load_model_context(a.id, conv["id"], 6)
        assert turns[0]["assistant"] == "Ответ (source:1)"
        assert turns[0]["tool_messages"][0]["content"] == "receipt"
        assert sources == [(1, "paper.pdf")]
        assert await reopened.get_graph_run(a.id, "gr_saved") == [{"chain_id": "a1", "edges": []}]
    finally:
        await reopened.close()


@pytest.mark.asyncio
async def test_duplicate_turn_and_backup(tmp_path):
    store = AppStore(str(tmp_path / "app.db"))
    await store.open()
    try:
        user = await store.create_user("backup-user", "long enough backup password")
        conv = await store.create_conversation(user.id)
        turn_id = "22222222-2222-4222-8222-222222222222"
        await store.begin_turn(user.id, conv["id"], turn_id, "question")
        with pytest.raises(ValueError):
            await store.begin_turn(user.id, conv["id"], turn_id, "question")
        destination = tmp_path / "backup.db"
        await store.backup(str(destination))
    finally:
        await store.close()

    with sqlite3.connect(destination) as db:
        assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 1


@pytest.mark.asyncio
async def test_concurrent_turns_keep_unique_order(tmp_path):
    store = AppStore(str(tmp_path / "app.db"))
    await store.open()
    try:
        user = await store.create_user("parallel-user", "long parallel user password")
        conv = await store.create_conversation(user.id)
        await asyncio.gather(
            store.begin_turn(user.id, conv["id"], "33333333-3333-4333-8333-333333333333", "one"),
            store.begin_turn(user.id, conv["id"], "44444444-4444-4444-8444-444444444444", "two"),
        )
        detail = await store.get_conversation(user.id, conv["id"])
        assert detail is not None
        user_texts = [item["text"] for item in detail["messages"] if item["role"] == "user"]
        assert len(user_texts) == 2
        assert set(user_texts) == {"one", "two"}
    finally:
        await store.close()

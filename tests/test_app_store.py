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
async def test_concurrent_turns_allow_only_one_active_turn_per_branch(tmp_path):
    store = AppStore(str(tmp_path / "app.db"))
    await store.open()
    try:
        user = await store.create_user("parallel-user", "long parallel user password")
        conv = await store.create_conversation(user.id)
        results = await asyncio.gather(
            store.begin_turn(user.id, conv["id"], "33333333-3333-4333-8333-333333333333", "one"),
            store.begin_turn(user.id, conv["id"], "44444444-4444-4444-8444-444444444444", "two"),
            return_exceptions=True,
        )
        assert sum(isinstance(result, RuntimeError) for result in results) == 1
        detail = await store.get_conversation(user.id, conv["id"])
        assert detail is not None
        user_texts = [item["text"] for item in detail["messages"] if item["role"] == "user"]
        assert len(user_texts) == 1
        assert user_texts[0] in {"one", "two"}
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_checkpoint_fork_agenda_and_cards_survive_conversation_delete(tmp_path):
    store = AppStore(str(tmp_path / "state.db"))
    await store.open()
    try:
        user = await store.create_user("state-user", "long state user password")
        conv = await store.create_conversation(user.id)
        branch_id = conv["activeBranchId"]
        started = await store.begin_branch_turn(
            user.id,
            conv["id"],
            branch_id,
            "55555555-5555-4555-8555-555555555555",
            "question",
            mode="staged",
        )
        agenda = await store.upsert_turn_subquestions(
            conv["id"], started["userCheckpointId"], ["Starter cultures acidify milk."], increment=True
        )
        sq_id = agenda[0]["id"]
        chain = {
            "chain_id": "c1",
            "source_graph": sq_id,
            "edge_keys": ["edge-1"],
            "spine_evidence_seq": ["quote-1"],
            "text": "UNIT c1\nCulture —ACIDIFIES→ Milk",
            "walk": [{
                "edge_key": "edge-1",
                "evidence": "Exact evidence quote.",
                "source_file": "paper.pdf",
                "start": "Culture",
                "end": "Milk",
                "type": "ACIDIFIES",
            }],
        }
        recorded = await store.record_units(conv["id"], started["userCheckpointId"], [chain])
        await store.finish_turn(
            conv["id"],
            started["assistantMessageId"],
            text="answer",
            status="done",
            payload={},
            raw_text="answer",
            graph_chains=recorded,
            retrieval_state={
                "algorithmVersion": "v6-checkpoint-1",
                "s3Bundle": {"graphs": {sq_id: {"source_graph": sq_id, "edges": []}}},
                "carousel": {"p_store": {"edge-1": 0.7}, "counts": {sq_id: 1}},
                "priorSignatures": ["quote-1"],
                "lastSubquestionIds": [sq_id],
                "lastTrace": {"accepted": 1},
            },
        )
        detail = await store.get_conversation(user.id, conv["id"], branch_id=branch_id)
        assert detail is not None
        answer_checkpoint = detail["headCheckpointId"]
        assert detail["agenda"][0]["graphSnapshotId"]
        assert (await store.load_retrieval_state(user.id, answer_checkpoint))["carousel"]["p_store"]["edge-1"] == 0.7

        fork = await store.create_fork(user.id, conv["id"], answer_checkpoint)
        changed = await store.apply_agenda_event(
            user.id,
            fork["id"],
            base_checkpoint_id=answer_checkpoint,
            action="close",
            sq_id=sq_id,
        )
        sibling = await store.get_conversation(user.id, conv["id"], branch_id=branch_id)
        assert changed["agenda"][0]["status"] == "closed"
        assert sibling["agenda"][0]["status"] == "open"

        template = (await store.list_card_templates(user.id))[0]
        draft = await store.create_card_draft(
            user.id,
            checkpoint_id=answer_checkpoint,
            template_version_id=template["latestVersion"]["id"],
            data={"title": "Trial", "objective": "Test", "product_or_matrix": "Milk", "gaps": []},
            provenance={
                "/objective": {
                    "unit_id": recorded[0]["unit_id"],
                    "edge_key": "edge-1",
                    "source_document": "paper.pdf",
                    "quote": "Exact evidence quote.",
                }
            },
        )
        card = await store.save_card_draft(user.id, draft["id"], title="Trial")
        attached = await store.attach_card_revision(
            user.id,
            fork["id"],
            base_checkpoint_id=changed["checkpointId"],
            card_revision_id=card["latestRevision"]["id"],
            attached=True,
        )
        context = await store.checkpoint_card_context(user.id, attached["checkpointId"])
        assert context and context[0]["title"] == "Trial"

        assert await store.delete_conversation(user.id, conv["id"])
        cards = await store.list_cards(user.id)
        assert cards[0]["title"] == "Trial"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_pending_approval_revision_is_persistent_and_single_use(tmp_path):
    store = AppStore(str(tmp_path / "approval.db"))
    await store.open()
    try:
        user = await store.create_user("approval-user", "long approval user password")
        conv = await store.create_conversation(user.id)
        started = await store.begin_branch_turn(
            user.id,
            conv["id"],
            conv["activeBranchId"],
            "66666666-6666-4666-8666-666666666666",
            "question",
            mode="staged",
        )
        approval = await store.create_pending_approval(
            user.id,
            conversation_id=conv["id"],
            branch_id=conv["activeBranchId"],
            user_message_id=started["userMessageId"],
            assistant_message_id=started["assistantMessageId"],
            base_checkpoint_id=started["userCheckpointId"],
            tool_call={"id": "call-1", "name": "ask_subgraph", "arguments": {"subquestions": ["A"]}},
            resume={"text": "question"},
            settings={"mode": "staged"},
        )
        await store.update_assistant_waiting(conv["id"], started["assistantMessageId"], payload={})
        claimed = await store.claim_pending_approval(user.id, approval["id"], 1, "approve")
        assert claimed and claimed["toolCall"]["id"] == "call-1"
        with pytest.raises(RuntimeError):
            await store.claim_pending_approval(user.id, approval["id"], 1, "approve")
    finally:
        await store.close()

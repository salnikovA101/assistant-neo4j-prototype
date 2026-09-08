from __future__ import annotations

import asyncio
import sqlite3

import pytest

from server.core.app_store import AppStore, ConversationRunMismatchError


@pytest.mark.asyncio
async def test_workspace_run_id_snapshots_conversations_and_keeps_old_chat_read_only(tmp_path):
    store = AppStore(str(tmp_path / "runs.db"), workspaces={"packaging": "run-new"})
    await store.open()
    try:
        user = await store.create_user("run-user", "long enough run password")
        assert user.workspace == "packaging"
        conversation = await store.create_conversation(user.id)
        assert conversation["runId"] == "run-new"
        assert conversation["readOnly"] is False

        await store._conn().execute(
            "UPDATE conversations SET run_id='run-old' WHERE id=?",
            (conversation["id"],),
        )
        await store._conn().commit()
        listed = (await store.list_conversations(user.id))["items"]
        assert listed[0]["runId"] == "run-old"
        assert listed[0]["workspaceRunId"] == "run-new"
        assert listed[0]["readOnly"] is True
        assert listed[0]["readOnlyReason"] == "run_id_changed"
        with pytest.raises(ConversationRunMismatchError) as exc_info:
            await store.begin_turn(
                user.id,
                conversation["id"],
                "10101010-1010-4010-8010-101010101010",
                "must stay read-only",
            )
        assert exc_info.value.conversation_run_id == "run-old"
        assert exc_info.value.workspace_run_id == "run-new"

        # Metadata remains manageable while content is frozen.
        assert await store.rename_conversation(user.id, conversation["id"], "Архив")
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_workspace_migration_moves_existing_accounts_and_drops_account_run_id(tmp_path):
    path = tmp_path / "legacy-runs.db"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute(
            """CREATE TABLE users (
                   id TEXT PRIMARY KEY, username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                   password_hash TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 1,
                   run_id TEXT NOT NULL DEFAULT '',
                   created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
               )"""
        )
        db.execute(
            """CREATE TABLE conversations (
                   id TEXT PRIMARY KEY,
                   user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                   title TEXT NOT NULL DEFAULT 'Новый чат',
                   created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
               )"""
        )
        db.execute(
            "INSERT INTO users VALUES('u1','legacy','hash',1,'legacy-account-run',1,1)"
        )
        db.execute(
            "INSERT INTO conversations VALUES('c1','u1','Legacy chat',1,1)"
        )

    store = AppStore(str(path), workspaces={"packaging": "bootstrap-run"})
    await store.open()
    try:
        user_row = await (await store._conn().execute(
            "SELECT workspace FROM users WHERE id='u1'"
        )).fetchone()
        conversation_row = await (await store._conn().execute(
            "SELECT title,run_id FROM conversations WHERE id='c1'"
        )).fetchone()
        columns = await store._table_columns("users")
        assert user_row["workspace"] == "packaging"
        assert "run_id" not in columns
        assert conversation_row is not None
        assert conversation_row["run_id"] == "bootstrap-run"
        assert conversation_row["title"] == "Legacy chat"
        users_sql = await (await store._conn().execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='users'"
        )).fetchone()
        compact = "".join(str(users_sql["sql"] or "").split()).lower().replace('"', "")
        assert "unique(username,workspace)" in compact
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_username_workspace_rebuild_does_not_cascade_delete_chats(tmp_path):
    path = tmp_path / "scoped-users.db"
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA foreign_keys=ON")
        db.execute(
            """CREATE TABLE users (
                   id TEXT PRIMARY KEY, username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                   password_hash TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 1,
                   workspace TEXT NOT NULL DEFAULT 'packaging',
                   created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
               )"""
        )
        db.execute(
            """CREATE TABLE conversations (
                   id TEXT PRIMARY KEY,
                   user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                   title TEXT NOT NULL DEFAULT 'Новый чат',
                   run_id TEXT NOT NULL DEFAULT '',
                   created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
               )"""
        )
        db.execute(
            "INSERT INTO users VALUES('u1','kefir','hash',1,'kefir',1,1)"
        )
        db.execute(
            "INSERT INTO conversations VALUES('c1','u1','Keep me','new_mega_run',1,1)"
        )

    store = AppStore(
        str(path), workspaces={"packaging": "p-run", "kefir": "new_mega_run"}
    )
    await store.open()
    try:
        user_row = await (await store._conn().execute(
            "SELECT username,workspace FROM users WHERE id='u1'"
        )).fetchone()
        conversation_row = await (await store._conn().execute(
            "SELECT title,run_id FROM conversations WHERE id='c1'"
        )).fetchone()
        users_sql = await (await store._conn().execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='users'"
        )).fetchone()
        compact = "".join(str(users_sql["sql"] or "").split()).lower().replace('"', "")
        assert user_row["username"] == "kefir"
        assert user_row["workspace"] == "kefir"
        assert conversation_row["title"] == "Keep me"
        assert conversation_row["run_id"] == "new_mega_run"
        assert "unique(username,workspace)" in compact
        fk = await (await store._conn().execute("PRAGMA foreign_keys")).fetchone()
        assert int(fk[0]) == 1
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_same_login_is_unique_per_workspace(tmp_path):
    store = AppStore(
        str(tmp_path / "shared-login.db"),
        workspaces={"packaging": "p-run", "kefir": "k-run"},
    )
    await store.open()
    try:
        packaging = await store.create_user(
            "worker", "a sufficiently long password", workspace="packaging"
        )
        kefir = await store.create_user(
            "worker", "another sufficiently long password", workspace="kefir"
        )
        assert packaging.id != kefir.id
        assert await store.authenticate(
            "worker", "a sufficiently long password", workspace="packaging"
        ) == packaging
        assert await store.authenticate(
            "worker", "another sufficiently long password", workspace="kefir"
        ) == kefir
        assert await store.authenticate(
            "worker", "a sufficiently long password", workspace="kefir"
        ) is None
        with pytest.raises(ValueError, match="уже существует"):
            await store.create_user(
                "worker", "third sufficiently long password", workspace="packaging"
            )
        assert await store.reset_password(
            "worker", "packaging reset pw", workspace="packaging"
        )
        assert await store.authenticate(
            "worker", "packaging reset pw", workspace="packaging"
        ) == packaging
        assert await store.authenticate(
            "worker", "another sufficiently long password", workspace="kefir"
        ) == kefir
    finally:
        await store.close()


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

        assert await store.reset_password(
            "technologist", "a different long password", workspace="packaging"
        )
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
    assert await store.graph_run_corpus_id(b.id, "gr_saved") is None
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
        assert await reopened.graph_run_corpus_id(a.id, "gr_saved") == conv["runId"]
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
async def test_research_map_historical_view_and_atomic_auto_fork(tmp_path):
    store = AppStore(str(tmp_path / "research-map.db"))
    await store.open()
    try:
        user = await store.create_user("map-user", "long research map password")
        conv = await store.create_conversation(user.id)
        branch_id = conv["activeBranchId"]

        first = await store.begin_branch_turn(
            user.id,
            conv["id"],
            branch_id,
            "77777777-7777-4777-8777-777777777777",
            "Какая культура подходит для кефира?",
            mode="auto",
        )
        recorded = await store.record_units(
            conv["id"],
            first["userCheckpointId"],
            [{
                "chain_id": "c1",
                "edge_keys": ["edge-map-1"],
                "text": "UNIT c1",
                "walk": [{
                    "edge_key": "edge-map-1",
                    "start": "Culture",
                    "end": "Kefir",
                    "type": "USED_IN",
                    "evidence": "Culture is used in kefir.",
                }],
            }],
        )
        first_answer_checkpoint = await store.finish_turn(
            conv["id"],
            first["assistantMessageId"],
            text="Используйте молочнокислую культуру.",
            status="done",
            payload={"graphChainCount": 1},
            graph_chains=recorded,
        )

        second = await store.begin_branch_turn(
            user.id,
            conv["id"],
            branch_id,
            "88888888-8888-4888-8888-888888888888",
            "Какая нужна температура?",
            base_checkpoint_id=first_answer_checkpoint,
            mode="auto",
        )
        await store.finish_turn(
            conv["id"],
            second["assistantMessageId"],
            text="Около 30 градусов.",
            status="done",
            payload={},
        )

        research = await store.research_map(user.id, conv["id"], branch_id)
        assert research is not None
        assert [step["displayNo"] for step in research["steps"]] == [1, 2]
        assert research["steps"][0]["unitNos"] == [1]
        assert research["steps"][1]["parentStepId"] == research["steps"][0]["id"]

        chains = await store.checkpoint_chains(
            user.id, first_answer_checkpoint, scope="new_in_answer"
        )
        assert chains and chains[0]["origin"]["question"] == "Какая культура подходит для кефира?"
        assert chains[0]["origin"]["step_no"] == 1

        historical = await store.get_conversation(
            user.id,
            conv["id"],
            branch_id=branch_id,
            checkpoint_id=first_answer_checkpoint,
        )
        assert historical is not None
        assert historical["atBranchHead"] is False
        assert [message["text"] for message in historical["messages"]] == [
            "Какая культура подходит для кефира?",
            "Используйте молочнокислую культуру.",
        ]

        forked = await store.begin_branch_turn(
            user.id,
            conv["id"],
            branch_id,
            "99999999-9999-4999-8999-999999999999",
            "А если использовать дрожжи?",
            base_checkpoint_id=first_answer_checkpoint,
            fork_if_needed=True,
            mode="auto",
        )
        assert forked["branchCreated"]["name"] == "Версия 2"
        assert forked["branchId"] != branch_id
        fork_detail = await store.get_conversation(
            user.id, conv["id"], branch_id=forked["branchId"]
        )
        assert [message["text"] for message in fork_detail["messages"]] == [
            "Какая культура подходит для кефира?",
            "Используйте молочнокислую культуру.",
            "А если использовать дрожжи?",
        ]
        assert [message["branchId"] for message in fork_detail["messages"]] == [
            branch_id,
            branch_id,
            forked["branchId"],
        ]
        assert len(await store.list_branches(user.id, conv["id"])) == 2

        with pytest.raises(RuntimeError, match="invalid_fork_checkpoint"):
            await store.begin_branch_turn(
                user.id,
                conv["id"],
                branch_id,
                "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
                "Неверная точка",
                base_checkpoint_id=first["userCheckpointId"],
                fork_if_needed=True,
                mode="auto",
            )
        assert len(await store.list_branches(user.id, conv["id"])) == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_manual_fork_starts_at_selected_assistant_step(tmp_path):
    store = AppStore(str(tmp_path / "manual-fork-map.db"))
    await store.open()
    try:
        user = await store.create_user("manual-fork-user", "long manual fork password")
        conv = await store.create_conversation(user.id, mode="staged")
        branch_id = conv["activeBranchId"]
        first = await store.begin_branch_turn(
            user.id,
            conv["id"],
            branch_id,
            "12121212-1212-4212-8212-121212121212",
            "Первый вопрос",
            mode="staged",
        )
        first_answer_checkpoint = await store.finish_turn(
            conv["id"],
            first["assistantMessageId"],
            text="Первый ответ",
            status="done",
            payload={},
        )
        second = await store.begin_branch_turn(
            user.id,
            conv["id"],
            branch_id,
            "34343434-3434-4434-8434-343434343434",
            "Второй вопрос",
            base_checkpoint_id=first_answer_checkpoint,
            mode="staged",
        )
        await store.finish_turn(
            conv["id"],
            second["assistantMessageId"],
            text="Второй ответ",
            status="done",
            payload={},
        )

        fork = await store.create_fork(
            user.id,
            conv["id"],
            first_answer_checkpoint,
            source_branch_id=branch_id,
        )
        detail = await store.get_conversation(user.id, conv["id"], branch_id=fork["id"])
        assert detail is not None
        assert [message["text"] for message in detail["messages"]] == [
            "Первый вопрос",
            "Первый ответ",
        ]

        research = await store.research_map(user.id, conv["id"], fork["id"])
        assert research is not None
        first_step = research["steps"][0]
        fork_branch = next(branch for branch in research["branches"] if branch["id"] == fork["id"])
        assert fork_branch["originStepId"] == first_step["id"]
        assert fork_branch["headStepId"] == first_step["id"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_checkpoint_fork_agenda_and_cards_survive_conversation_delete(tmp_path):
    store = AppStore(str(tmp_path / "state.db"))
    await store.open()
    try:
        user = await store.create_user("state-user", "long state user password")
        conv = await store.create_conversation(user.id, mode="staged")
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
            conv["id"],
            started["userCheckpointId"],
            ["Starter cultures acidify milk."],
            increment=True,
            agenda_visible=True,
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
                "algorithmVersion": "retrieval-carousel-v1",
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
        renamed = await store.rename_branch(user.id, fork["id"], "  Клубничная версия  ")
        assert renamed["name"] == "Клубничная версия"
        changed = await store.apply_agenda_event(
            user.id,
            fork["id"],
            base_checkpoint_id=answer_checkpoint,
            action="close",
            sq_ref=agenda[0]["ref"],
        )
        sibling = await store.get_conversation(user.id, conv["id"], branch_id=branch_id)
        assert changed["agenda"][0]["status"] == "closed"
        assert sibling["agenda"][0]["status"] == "not_closed"

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
        conv = await store.create_conversation(user.id, mode="staged")
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
            tool_call={
                "id": "call-1",
                "name": "advance_research",
                "arguments": {"open_sq_refs": [], "new_subquestions": ["A"]},
            },
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


@pytest.mark.asyncio
async def test_research_map_preview_keeps_markdown_headings(tmp_path):
    store = AppStore(str(tmp_path / "md-preview.db"))
    await store.open()
    try:
        user = await store.create_user("md-preview-user", "long markdown preview password")
        conv = await store.create_conversation(user.id)
        branch_id = conv["activeBranchId"]
        turn = await store.begin_branch_turn(
            user.id,
            conv["id"],
            branch_id,
            "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa",
            "Можешь собрать клубничную закваску для кефира?",
            mode="auto",
        )
        await store.finish_turn(
            conv["id"],
            turn["assistantMessageId"],
            text="## Что подтверждено в этой версии\n\nПрямо «клубничной закваски» в базе нет: ни одного узла по клубнике. " + ("Дополнительные данные по кисломолочным продуктам и стартовым культурам. " * 8),
            status="done",
            payload={},
        )
        research = await store.research_map(user.id, conv["id"], branch_id)
        preview = research["steps"][0]["answer"]["preview"]
        assert preview.startswith("## Что подтверждено в этой версии")
        assert "\n\n" in preview
        assert "## Что подтверждено в этой версии Прямо" not in preview
        assert len(preview) > 180
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_corrupt_state_json_raises(tmp_path):
    from server.core.app_store import AppStore, CorruptStoreError

    store = AppStore(str(tmp_path / "corrupt.db"))
    await store.open()
    try:
        user = await store.create_user("corrupt-user", "long enough corrupt password")
        conv = await store.create_conversation(user.id)
        started = await store.begin_branch_turn(
            user.id,
            conv["id"],
            conv["activeBranchId"],
            "33333333-3333-4333-8333-333333333333",
            "question",
        )
        await store._conn().execute(
            "UPDATE checkpoints SET state_json=? WHERE id=?",
            ("{not-json", started["userCheckpointId"]),
        )
        await store._conn().commit()
        with pytest.raises(CorruptStoreError):
            await store.checkpoint_state(user.id, started["userCheckpointId"])
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_failed_user_rebuild_rolls_back_and_can_retry(tmp_path, monkeypatch):
    """A failure after DROP must preserve the old parent and its child rows."""
    path = tmp_path / "interrupted-migration.db"
    with sqlite3.connect(path) as db:
        db.executescript("""
            CREATE TABLE users (
                id TEXT PRIMARY KEY, username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL, is_active INTEGER NOT NULL DEFAULT 1,
                workspace TEXT NOT NULL DEFAULT 'packaging',
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
            );
            CREATE TABLE conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                title TEXT NOT NULL DEFAULT 'Новый чат', run_id TEXT NOT NULL DEFAULT '',
                created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL
            );
            INSERT INTO users VALUES('u1','worker','hash',1,'packaging',1,1);
            INSERT INTO conversations VALUES('c1','u1','Keep me','p-run',1,1);
        """)
    import aiosqlite
    original = aiosqlite.Connection.execute

    def fail_rename(self, sql, parameters=None):
        if sql == "ALTER TABLE users_workspace_unique RENAME TO users":
            raise sqlite3.OperationalError("simulated migration failure")
        return original(self, sql, parameters or [])

    store = AppStore(str(path), workspaces={"packaging": "p-run"})
    with monkeypatch.context() as patch:
        patch.setattr(aiosqlite.Connection, "execute", fail_rename)
        try:
            with pytest.raises(sqlite3.OperationalError, match="simulated"):
                await store.open()
            fk = await (await store._conn().execute("PRAGMA foreign_keys")).fetchone()
            assert fk[0] == 1
        finally:
            await store.close()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
        assert db.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 1
        assert not db.execute("PRAGMA foreign_key_check").fetchall()
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='users_workspace_unique'").fetchall()
    for _ in range(2):
        retry = AppStore(str(path), workspaces={"packaging": "p-run"})
        await retry.open()
        try:
            assert (await (await retry._conn().execute("SELECT COUNT(*) FROM users")).fetchone())[0] == 1
            assert (await (await retry._conn().execute("SELECT COUNT(*) FROM conversations")).fetchone())[0] == 1
        finally:
            await retry.close()

from __future__ import annotations

from pathlib import Path

import pytest
import aiosqlite

from server.core.app import _checkpoint_prompt_context
from server.core.app_store import AppStore
from server.tools.registry import Tools
from server.tools.subgraph_search import MAX_SUBQUESTIONS, _format_accepted_chains, normalize_subquestions
from server.tools.graph_viz import build_chain_views
from server.utils.config import AppConfig


@pytest.mark.asyncio
async def test_staged_context_keeps_agenda_and_units_while_auto_hides_them(tmp_path):
    store = AppStore(str(tmp_path / "modes.db"))
    await store.open()
    try:
        user = await store.create_user("mode-user", "long enough mode password")
        conversation = await store.create_conversation(user.id, mode="staged")
        started = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "10101010-1010-4010-8010-101010101010",
            "question",
            mode="staged",
        )
        agenda = await store.upsert_turn_subquestions(
            conversation["id"],
            started["userCheckpointId"],
            ["Starter cultures acidify milk."],
            increment=True,
            agenda_visible=True,
        )
        sq_id = agenda[0]["id"]
        sq_ref = agenda[0]["ref"]
        chains = [
            {
                "source_graph": sq_id,
                "edge_keys": [f"edge-{index}"],
                "spine_evidence_seq": [f"quote-{index}"],
                "text": f"UNIT {index}\nCulture —ACIDIFIES→ Milk {index}",
            }
            for index in (1, 2)
        ]
        await store.record_units(
            conversation["id"], started["userCheckpointId"], chains
        )

        checkpoint = await store.checkpoint_state(user.id, started["userCheckpointId"])
        assert checkpoint is not None
        assert checkpoint["agenda"][0]["unitCount"] == 2
        assert checkpoint["agenda"][0]["reviewRecommended"] is True

        staged = await _checkpoint_prompt_context(
            store, user.id, started["userCheckpointId"], mode="staged"
        )
        auto = await _checkpoint_prompt_context(
            store, user.id, started["userCheckpointId"], mode="auto"
        )
        card = await _checkpoint_prompt_context(
            store,
            user.id,
            started["userCheckpointId"],
            mode="auto",
            purpose="card",
        )
        assert "CURRENT RESEARCH QUESTIONS" in staged
        assert "assess only these refs" in staged
        assert sq_ref in staged
        assert sq_id not in staged
        assert "UNIT" not in staged
        assert staged.count("Chain [1]") == 1
        assert "Culture —acidifies→ Milk 1" in staged
        assert "UNIT" not in card
        assert "paths=2, review recommended" in staged
        assert "EVIDENCE Chains" in staged
        assert "CURRENT RESEARCH QUESTIONS" not in auto
        assert "EVIDENCE Chains" not in auto
        assert "CURRENT RESEARCH QUESTIONS" not in card
        assert "EVIDENCE Chains" in card

        closed = await store.finish_turn(
            conversation["id"],
            started["assistantMessageId"],
            text="answer",
            status="done",
            payload={},
            sq_assessments=[{
                "ref": sq_ref,
                "status": "closed",
                "reason": "Подтверждено",
                "source_refs": ["source:1"],
            }],
        )
        after_close = await _checkpoint_prompt_context(store, user.id, closed, mode="staged")
        assert "CLOSED RESEARCH QUESTIONS (do not assess" in after_close
        assert sq_ref in after_close
        assert "[closed]" in after_close
        assert "- none" in after_close

        auto_fork = await store.create_fork(
            user.id,
            conversation["id"],
            started["userCheckpointId"],
            mode="auto",
        )
        assert auto_fork["mode"] == "auto"
        with pytest.raises(RuntimeError, match="agenda_unavailable"):
            await store.apply_agenda_event(
                user.id,
                auto_fork["id"],
                base_checkpoint_id=started["userCheckpointId"],
                action="add",
                text="A new search direction.",
            )
        with pytest.raises(ValueError):
            await store.create_fork(
                user.id,
                conversation["id"],
                auto_fork["headCheckpointId"],
                mode="staged",
                source_branch_id=auto_fork["id"],
            )
    finally:
        await store.close()


def test_mode_tools_have_distinct_contracts_and_five_sq_limit():
    tools = Tools(AppConfig())
    auto = tools.get_openai_tools("auto")[0]["function"]
    staged = tools.get_openai_tools("staged")[0]["function"]
    assert auto["name"] == "ask_subgraph"
    assert auto["parameters"]["properties"]["subquestions"]["maxItems"] == 5
    assert staged["name"] == "advance_research"
    assert "required" not in staged["parameters"]
    assert "Empty research-question list" in staged["description"]
    refs = staged["parameters"]["properties"]["open_sq_refs"]
    assert refs["maxItems"] == 5
    assert refs["items"]["pattern"] == "^subquestion:[1-9][0-9]*$"
    clean, problems = normalize_subquestions(
        [f"Distinct evidence direction {index}." for index in range(7)]
    )
    assert len(clean) == MAX_SUBQUESTIONS == 5
    assert problems


def test_model_contract_uses_research_questions_and_evidence_language():
    prompt_root = Path(__file__).resolve().parents[1] / "prompts"
    prompt_text = "\n".join(
        (prompt_root / name).read_text(encoding="utf-8")
        for name in ("assistant_logic.md", "assistant_staged.md", "card_generation.md")
    )
    staged_tool = Tools(AppConfig()).get_openai_tools("staged")[0]["function"]
    formatted = _format_accepted_chains([{"text": "UNIT 1\nThing —REL→ Other\n  Evidence (paper.pdf; conf=1)"}])

    assert "CURRENT RESEARCH QUESTIONS" in prompt_text
    assert "CURRENT SQ AGENDA" not in prompt_text
    assert "дословная цитата" not in prompt_text
    assert "точным текстом evidence" in prompt_text
    assert prompt_text.count("Ты отвечаешь по данным пользователя и базы знаний Neo4j") == 2
    assert prompt_text.count("считай источником предоставленной пользователем информации") == 2
    assert prompt_text.count("если пользователь не просит их проверить") == 2
    assert "research-question list" in staged_tool["description"]
    assert "agenda" not in staged_tool["description"].lower()
    assert "evidence text" in formatted
    assert "verbatim quote" not in formatted


@pytest.mark.asyncio
async def test_checkpoint_graph_scope_separates_auto_and_staged_units(tmp_path):
    store = AppStore(str(tmp_path / "checkpoint-graph.db"))
    await store.open()
    try:
        user = await store.create_user("graph-mode-user", "long graph mode password")

        async def add_answer(conversation, branch_id, turn_id, mode, edge_key):
            started = await store.begin_branch_turn(
                user.id, conversation["id"], branch_id, turn_id, edge_key, mode=mode
            )
            chain = {
                "source_graph": "",
                "edge_keys": [edge_key],
                "spine_evidence_seq": [edge_key],
                "text": f"UNIT {edge_key}",
                "edges": [],
            }
            recorded = await store.record_units(
                conversation["id"], started["userCheckpointId"], [chain]
            )
            checkpoint_id = await store.finish_turn(
                conversation["id"],
                started["assistantMessageId"],
                text="answer",
                status="done",
                payload={},
                graph_chains=recorded,
            )
            return checkpoint_id

        staged = await store.create_conversation(user.id, mode="staged")
        staged_one = await add_answer(
            staged, staged["activeBranchId"], "graph-staged-one", "staged", "staged-1"
        )
        staged_two = await add_answer(
            staged, staged["activeBranchId"], "graph-staged-two", "staged", "staged-2"
        )
        staged_context = await store.checkpoint_chains(user.id, staged_two, scope="context")
        assert [(item["unit_no"], item["is_new"]) for item in staged_context or []] == [
            (1, False), (2, True)
        ]
        first_context = await store.checkpoint_chains(user.id, staged_one, scope="context")
        assert [(item["unit_no"], item["is_new"]) for item in first_context or []] == [(1, True)]
        assert [view["label"] for view in build_chain_views(staged_context or [])] == ["UNIT 1", "UNIT 2"]

        auto = await store.create_conversation(user.id, mode="auto")
        await add_answer(auto, auto["activeBranchId"], "graph-auto-one", "auto", "auto-1")
        auto_two = await add_answer(auto, auto["activeBranchId"], "graph-auto-two", "auto", "auto-2")
        auto_last = await store.checkpoint_chains(user.id, auto_two, scope="new_in_answer")
        assert [item["unit_no"] for item in auto_last or []] == [2]
        assert [item["unit_no"] for item in (await store.checkpoint_chains(user.id, auto_two, scope="context")) or []] == [1, 2]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_subquestion_refs_are_stable_across_forks_and_hide_uuid(tmp_path):
    store = AppStore(str(tmp_path / "refs.db"))
    await store.open()
    try:
        user = await store.create_user("refs-user", "long enough refs password")
        conversation = await store.create_conversation(user.id, mode="staged")
        started = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "20202020-2020-4020-8020-202020202020",
            "question",
            mode="staged",
        )
        internal = await store.upsert_turn_subquestions(
            conversation["id"],
            started["userCheckpointId"],
            ["Starter cultures acidify milk."],
            agenda_visible=True,
        )
        sq_id = internal[0]["id"]
        assert internal[0]["ref"] == "subquestion:1"
        checkpoint = await store.checkpoint_state(user.id, started["userCheckpointId"])
        assert checkpoint is not None
        assert checkpoint["agenda"] == [{
            "ref": "subquestion:1",
            "text": "Starter cultures acidify milk.",
            "status": "not_closed",
            "statusOrigin": "legacy",
            "statusReason": "",
            "statusSourceRefs": [],
            "statusMessageId": None,
            "position": 0,
            "questionCount": 1,
            "unitCount": 0,
            "graphSnapshotId": None,
            "reviewRecommended": False,
        }]
        assert sq_id not in str(checkpoint)
        assert (await store.open_agenda_subquestions(
            started["userCheckpointId"], ["subquestion:1"]
        ))[0]["id"] == sq_id
        assert await store.open_agenda_subquestions(
            started["userCheckpointId"], [sq_id]
        ) == []

        fork = await store.create_fork(
            user.id, conversation["id"], started["userCheckpointId"]
        )
        changed = await store.apply_agenda_event(
            user.id,
            fork["id"],
            base_checkpoint_id=started["userCheckpointId"],
            action="add",
            text="Yeasts produce aroma during kefir fermentation.",
        )
        assert [item["ref"] for item in changed["agenda"]] == [
            "subquestion:1", "subquestion:2"
        ]
        sibling = await store.get_conversation(
            user.id, conversation["id"], branch_id=conversation["activeBranchId"]
        )
        assert sibling is not None
        assert [item["ref"] for item in sibling["agenda"]] == ["subquestion:1"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_migration_backfills_display_numbers_deterministically(tmp_path):
    path = tmp_path / "legacy-refs.db"
    conn = await aiosqlite.connect(path)
    try:
        await conn.executescript(
            """CREATE TABLE subquestions (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                text TEXT NOT NULL,
                canonical_text TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                UNIQUE(conversation_id, canonical_text)
            );"""
        )
        await conn.executemany(
            "INSERT INTO subquestions(id,conversation_id,text,canonical_text,created_at) VALUES(?,?,?,?,?)",
            [
                ("legacy-b", "conversation-a", "B", "b", 10),
                ("legacy-a", "conversation-a", "A", "a", 10),
                ("legacy-c", "conversation-b", "C", "c", 1),
            ],
        )
        await conn.commit()
    finally:
        await conn.close()
    store = AppStore(str(path))
    await store.open()
    try:
        rows = await (await store._conn().execute(
            "SELECT id,conversation_id,display_no FROM subquestions ORDER BY conversation_id,display_no"
        )).fetchall()
        assert [(row["id"], row["display_no"]) for row in rows] == [
            ("legacy-a", 1), ("legacy-b", 2), ("legacy-c", 1)
        ]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_legacy_pending_approval_is_presented_with_refs(tmp_path):
    store = AppStore(str(tmp_path / "legacy-approval.db"))
    await store.open()
    try:
        user = await store.create_user("legacy-approval", "long enough legacy password")
        conversation = await store.create_conversation(user.id, mode="staged")
        started = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "30303030-3030-4030-8030-303030303030",
            "question",
            mode="staged",
        )
        agenda = await store.upsert_turn_subquestions(
            conversation["id"],
            started["userCheckpointId"],
            ["Starter cultures acidify milk."],
            agenda_visible=True,
        )
        internal_id = agenda[0]["id"]
        approval = await store.create_pending_approval(
            user.id,
            conversation_id=conversation["id"],
            branch_id=conversation["activeBranchId"],
            user_message_id=started["userMessageId"],
            assistant_message_id=started["assistantMessageId"],
            base_checkpoint_id=started["userCheckpointId"],
            tool_call={
                "id": "call-legacy",
                "name": "advance_research",
                "arguments": {"open_sq_ids": [internal_id], "new_subquestions": []},
            },
            resume={},
            settings={"mode": "staged"},
        )
        args = approval["toolCall"]["arguments"]
        assert args == {"open_sq_refs": ["subquestion:1"], "new_subquestions": []}
        assert internal_id not in str(approval)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_model_history_sanitizes_legacy_sq_uuid_in_text_and_tool_call(tmp_path):
    store = AppStore(str(tmp_path / "history-refs.db"))
    await store.open()
    try:
        user = await store.create_user("history-refs", "long enough history password")
        conversation = await store.create_conversation(user.id, mode="staged")
        started = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "40404040-4040-4040-8040-404040404040",
            "question",
            mode="staged",
        )
        internal = await store.upsert_turn_subquestions(
            conversation["id"],
            started["userCheckpointId"],
            ["Starter cultures acidify milk."],
            agenda_visible=True,
        )
        sq_id = internal[0]["id"]
        await store.finish_turn(
            conversation["id"],
            started["assistantMessageId"],
            text=f"Shown {sq_id}",
            raw_text=f"Raw {sq_id}",
            status="done",
            payload={"steps": [{"kind": "tool", "args": {"open_sq_ids": [sq_id]}}]},
            tool_messages=[{
                "role": "assistant",
                "tool_calls": [{"function": {
                    "name": "advance_research",
                    "arguments": f'{{"open_sq_ids":["{sq_id}"]}}',
                }}],
            }],
        )
        detail = await store.get_conversation(user.id, conversation["id"])
        assert detail is not None
        assert sq_id not in str(detail)
        assert "subquestion:1" in str(detail)
        history = await store.checkpoint_model_messages(
            user.id, str(detail["headCheckpointId"])
        )
        assert history is not None
        assert sq_id not in str(history)
        assert "open_sq_refs" in str(history)
        assert "subquestion:1" in str(history)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_deferred_question_is_hidden_protected_and_can_be_reopened(tmp_path):
    from server.core.sq_status import resolve_active_sq_refs

    store = AppStore(str(tmp_path / "deferred.db"))
    await store.open()
    try:
        user = await store.create_user("deferred-user", "long enough deferred password")
        conv = await store.create_conversation(user.id, mode="staged")
        turn = await store.begin_branch_turn(
            user.id, conv["id"], conv["activeBranchId"],
            "85858585-8585-4585-8585-858585858585", "Investigate", mode="staged",
        )
        agenda = await store.upsert_turn_subquestions(
            conv["id"], turn["userCheckpointId"], ["How does milk acidify?"],
            increment=True, agenda_visible=True,
        )
        ref = agenda[0]["ref"]
        finished = await store.finish_turn(
            conv["id"], turn["assistantMessageId"],
            text="Insufficient evidence.", status="done", payload={},
        )
        deferred = await store.apply_agenda_event(
            user.id, conv["activeBranchId"], base_checkpoint_id=finished,
            action="set_status", sq_ref=ref, text="deferred",
        )
        cp = deferred["checkpointId"]
        assert deferred["agenda"][0]["status"] == "deferred"
        assert await store.open_agenda_subquestions(cp, [ref]) == []
        assert await store.open_agenda_subquestions(cp, [ref], open_only=False)
        context = await _checkpoint_prompt_context(store, user.id, cp, mode="staged")
        assert ref not in context
        assert "How does milk acidify?" not in context
        assert await resolve_active_sq_refs({
            "store": store, "user_id": user.id, "checkpoint_id": cp,
            "active_sq_refs": [ref],
        }) == []

        next_turn = await store.begin_branch_turn(
            user.id, conv["id"], conv["activeBranchId"],
            "86868686-8686-4686-8686-868686868686", "Continue", mode="staged",
        )
        next_cp = await store.finish_turn(
            conv["id"], next_turn["assistantMessageId"],
            text="Answer.", status="done", payload={},
            sq_assessments=[{"ref": ref, "status": "closed", "reason": "Attempted update"}],
        )
        state = await store.checkpoint_state(user.id, next_cp)
        assert state["agenda"][0]["status"] == "deferred"
        reopened = await store.apply_agenda_event(
            user.id, conv["activeBranchId"], base_checkpoint_id=next_cp,
            action="set_status", sq_ref=ref, text="not_closed",
        )
        assert await store.open_agenda_subquestions(reopened["checkpointId"], [ref])
        assert ref in await _checkpoint_prompt_context(
            store, user.id, reopened["checkpointId"], mode="staged",
        )
    finally:
        await store.close()

from __future__ import annotations

import pytest

from server.core.app import _normalize_generated_card, _parse_card_arguments
from server.core.app_store import AppStore
from server.core.card_schema import validate_card_data


def test_card_arguments_accept_json_fence_and_surrounding_text() -> None:
    parsed = _parse_card_arguments(
        'Result:\n```json\n{"data":{"title":"Trial"},"provenance":{},"gaps":[]}\n```'
    )
    assert parsed == {"data": {"title": "Trial"}, "provenance": {}, "gaps": []}


def test_generated_card_normalizes_schema_and_provenance_per_field() -> None:
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": ["string", "null"]},
            "objective": {"type": ["string", "null"]},
            "controls": {"type": ["string", "null"]},
            "gaps": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "objective", "gaps"],
        "additionalProperties": False,
    }
    unit = {
        "unit_id": "unit-1",
        "walk": [{
            "edge_key": "edge-1",
            "source_file": "paper.pdf",
            "evidence": "Exact quote.",
        }],
    }
    data, provenance, gaps = _normalize_generated_card(
        {"title": "Trial", "objective": "Measure acidification", "controls": {"bad": True}},
        {
            "/objective": {
                "unit_id": "unit-1",
                "edge_key": "edge-1",
                "source_document": "hallucinated.pdf",
                "quote": "not exact",
            }
        },
        [],
        schema,
        [unit],
    )

    assert data["title"] is None
    assert data["objective"] == "Measure acidification"
    assert data["controls"] is None
    assert provenance["/objective"][0] == {
        "unit_id": "unit-1",
        "edge_key": "edge-1",
        "source_document": "paper.pdf",
        "quote": "Exact quote.",
    }
    assert any("/title" in item for item in gaps)
    assert any("controls" in item for item in gaps)
    assert validate_card_data(data, schema) == []


@pytest.mark.asyncio
async def test_card_draft_is_a_checkpointed_chat_message_and_save_updates_it(tmp_path) -> None:
    store = AppStore(str(tmp_path / "cards.db"))
    await store.open()
    try:
        user = await store.create_user("cards-user", "long cards user password")
        conversation = await store.create_conversation(user.id)
        started = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "77777777-7777-4777-8777-777777777777",
            "Plan an experiment",
        )
        await store.finish_turn(
            conversation["id"],
            started["assistantMessageId"],
            text="Use the evidence.",
            status="done",
            payload={},
            raw_text="Use the evidence.",
        )
        detail = await store.get_conversation(user.id, conversation["id"])
        checkpoint_id = detail["headCheckpointId"]
        template = (await store.list_card_templates(user.id))[0]
        draft = await store.create_card_draft(
            user.id,
            checkpoint_id=checkpoint_id,
            template_version_id=template["latestVersion"]["id"],
            data={"title": "Trial", "objective": None, "product_or_matrix": None, "gaps": []},
        )
        message = await store.append_card_draft_message(
            user.id, checkpoint_id, draft, template_name=template["name"]
        )
        assert message["checkpointId"] != checkpoint_id

        detail = await store.get_conversation(user.id, conversation["id"])
        assert detail["messages"][-1]["cardDraft"]["id"] == draft["id"]
        context = await store.checkpoint_text_context(user.id, detail["headCheckpointId"])
        assert context == [
            {"role": "user", "text": "Plan an experiment"},
            {"role": "assistant", "text": "Use the evidence."},
        ]

        await store.save_card_draft(user.id, draft["id"], title="Trial")
        detail = await store.get_conversation(user.id, conversation["id"])
        assert detail["messages"][-1]["cardDraft"]["status"] == "saved"
    finally:
        await store.close()

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from server.core.app import (
    _alias_sources_for_model,
    _card_generation_events,
    _discover_card_source_files,
    _normalize_generated_card,
    _parse_card_arguments,
)
from server.core.app_store import AppStore
from server.core.card_schema import validate_card_data, validate_template_schema
from server.core.http_api import TextProcessBody
from server.llm.stream_events import StreamEvent
from server.tools.source_registry import present_source_aliases_in_value


def test_card_arguments_accept_json_fence_and_surrounding_text() -> None:
    parsed = _parse_card_arguments(
        'Result:\n```json\n{"data":{"title":"Trial"},"provenance":{}}\n```'
    )
    assert parsed == {"data": {"title": "Trial"}, "provenance": {}}


def test_card_schema_validates_choice_and_iso_date() -> None:
    schema = {
        "type": "object",
        "properties": {
            "state": {"type": ["string", "null"], "enum": ["planned", "done", None]},
            "started": {"type": ["string", "null"], "format": "date"},
        },
        "additionalProperties": False,
    }
    assert validate_template_schema(schema) == []
    assert validate_card_data({"state": "planned", "started": "2026-08-31"}, schema) == []
    assert "expected one of" in validate_card_data({"state": "unknown", "started": None}, schema)[0]
    assert "expected ISO date" in validate_card_data({"state": None, "started": "31.08.2026"}, schema)[0]


def test_generated_card_normalizes_schema_and_provenance_per_field() -> None:
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": ["string", "null"]},
            "objective": {"type": ["string", "null"]},
            "controls": {"type": ["string", "null"]},
        },
        "required": ["title", "objective"],
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

    assert data["title"] == "Trial"
    assert data["objective"] == "Measure acidification"
    assert data["controls"] is None
    assert provenance["/objective"][0] == {
        "unit_id": "unit-1",
        "edge_key": "edge-1",
        "source_document": "paper.pdf",
        "quote": "Exact quote.",
        "verification": "evidence",
    }
    assert provenance["/title"] == [{"verification": "assistant-generated/unverified"}]
    assert gaps == []
    assert validate_card_data(data, schema) == []


def test_generated_card_tracks_dialogue_and_unmatched_values() -> None:
    schema = {
        "type": "object",
        "properties": {
            "title": {"type": ["string", "null"]},
            "objective": {"type": ["string", "null"]},
        },
        "additionalProperties": False,
    }
    data, provenance, _ = _normalize_generated_card(
        {"title": "Проба 7", "objective": "Сводный вывод модели"},
        [{
            "pointer": "/title",
            "origin": "dialogue",
            "message_alias": "M1",
            "quote": "Проба 7",
        }],
        [],
        schema,
        [],
        {
            "M1": {
                "message_id": "message-1",
                "role": "user",
                "content": "Назовём опыт Проба 7",
            }
        },
    )
    assert provenance["/title"] == [{
        "verification": "user-provided/unverified",
        "message_id": "message-1",
        "quote": "Проба 7",
    }]
    assert provenance["/objective"] == [
        {"verification": "assistant-generated/unverified"}
    ]


def test_model_context_replaces_source_files_with_session_aliases() -> None:
    context = [
        {"role": "assistant", "content": "Facts from /papers/paper-a.pdf and paper-b.pdf"},
        {"role": "tool", "content": {"source_document": "paper-a.pdf"}},
    ]
    sanitized = _alias_sources_for_model(
        context,
        [(1, "/papers/paper-a.pdf"), (2, "paper-b.pdf")],
    )

    assert sanitized == [
        {"role": "assistant", "content": "Facts from source:1 and source:2"},
        {"role": "tool", "content": {"source_document": "source:1"}},
    ]


def test_card_sources_have_separate_model_and_user_presentations() -> None:
    stored = {
        "source_document": "source:1",
        "claim": "Confirmed in (source:1).",
    }
    presented = present_source_aliases_in_value(stored, [(1, "PMC12345_paper.pdf")])
    assert presented == {
        "source_document": "PMC12345_paper.pdf",
        "claim": "Confirmed in (PMC12345_paper.pdf).",
    }
    assert _discover_card_source_files({
        "source_document": "new paper.pdf",
        "claim": "Comparison with (second-paper.pdf).",
        "note": "A sentence merely ending in paper.pdf",
    }) == ["new paper.pdf", "second-paper.pdf"]


@pytest.mark.asyncio
async def test_system_template_can_be_hidden_per_user(tmp_path) -> None:
    store = AppStore(str(tmp_path / "cards.db"))
    await store.open()
    try:
        first = await store.create_user("template-owner", "long template owner password")
        second = await store.create_user("template-other", "long template other password")
        system_template = next(item for item in await store.list_card_templates(first.id) if item["system"])

        assert await store.archive_card_template(first.id, system_template["id"]) is True
        assert all(item["id"] != system_template["id"] for item in await store.list_card_templates(first.id))
        assert any(item["id"] == system_template["id"] for item in await store.list_card_templates(second.id))
    finally:
        await store.close()


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
        snapshot = await store.register_checkpoint_sources(
            user.id, checkpoint_id, ["PMC12345_paper.pdf", "PMC12345_paper.pdf"]
        )
        assert snapshot == [(1, "PMC12345_paper.pdf")]
        template = (await store.list_card_templates(user.id))[0]
        draft = await store.create_card_draft(
            user.id,
            checkpoint_id=checkpoint_id,
            template_version_id=template["latestVersion"]["id"],
            data={
                "title": "Trial",
                "objective": "Confirmed in source:1",
                "product_or_matrix": None,
            },
            provenance={
                "/objective": [{"verification": "assistant-generated/unverified"}],
            },
        )
        message = await store.append_card_draft_message(
            user.id, checkpoint_id, draft, template_name=template["name"]
        )
        assert message["checkpointId"] != checkpoint_id

        detail = await store.get_conversation(user.id, conversation["id"])
        assert detail["messages"][-1]["cardDraft"]["id"] == draft["id"]
        assert detail["messages"][-1]["cardDraft"]["data"]["objective"] == (
            "Confirmed in PMC12345_paper.pdf"
        )
        context = await store.checkpoint_text_context(user.id, detail["headCheckpointId"])
        assert context == [
            {"role": "user", "text": "Plan an experiment"},
            {"role": "assistant", "text": "Use the evidence."},
        ]

        model_context = await store.checkpoint_model_messages(
            user.id, detail["headCheckpointId"]
        )
        assert model_context is not None
        assert "CARD DRAFT DATA" in model_context[-1]["content"]
        assert '"title":"Trial"' in model_context[-1]["content"]
        assert "source:1" in model_context[-1]["content"]
        assert "PMC12345_paper.pdf" not in model_context[-1]["content"]
        assert '"provenance"' not in model_context[-1]["content"]
        assert "assistant-generated/unverified" not in model_context[-1]["content"]

        edited_data = {**draft["data"], "objective": "Technologist correction"}
        edited_provenance = {"/objective": [{"verification": "user-edited"}]}
        assert await store.update_card_draft(
            user.id,
            draft["id"],
            data=edited_data,
            provenance=edited_provenance,
            gaps=[],
        )
        detail = await store.get_conversation(user.id, conversation["id"])
        assert detail["messages"][-1]["cardDraft"]["data"]["objective"] == (
            "Technologist correction"
        )

        saved = await store.save_card_draft(user.id, draft["id"], title="Trial")
        cards = await store.list_cards(user.id)
        assert cards[0]["latestRevision"]["data"]["objective"] == (
            "Technologist correction"
        )
        assert cards[0]["latestRevision"]["provenance"] == edited_provenance

        editable = await store.card_for_edit(user.id, saved["id"])
        assert editable is not None
        revised_data = {**editable["data"], "title": "Trial revised", "objective": "Second correction"}
        revised = await store.add_card_revision(
            user.id,
            saved["id"],
            title="Trial revised",
            data=revised_data,
            provenance={"/objective": [{"verification": "user-edited"}]},
            gaps=editable["gaps"],
            origin_snapshot=editable["originSnapshot"],
        )
        assert revised["latestRevision"]["revision"] == 2
        cards = await store.list_cards(user.id)
        assert cards[0]["title"] == "Trial revised"
        assert cards[0]["latestRevision"]["data"]["objective"] == "Second correction"
        detail = await store.get_conversation(user.id, conversation["id"])
        assert detail["messages"][-1]["cardDraft"]["status"] == "saved"

        inserted = await store.append_card_reference_message(
            user.id,
            conversation["activeBranchId"],
            base_checkpoint_id=detail["headCheckpointId"],
            card_revision_id=saved["latestRevision"]["id"],
        )
        model_context = await store.checkpoint_model_messages(
            user.id, inserted["checkpointId"]
        )
        assert model_context is not None
        assert "INSERTED CARD DATA" in model_context[-1]["content"]
        assert model_context[-1]["role"] == "user"
        assert '"provenance"' not in model_context[-1]["content"]
        assert "user-edited" not in model_context[-1]["content"]
    finally:
        await store.close()


class _CardFallbackProvider:
    def __init__(self, unit_id: str) -> None:
        self.calls: list[dict] = []
        self.unit_id = unit_id

    async def generate_response_stream(self, *_args, **kwargs):
        self.calls.append(kwargs)
        if kwargs.get("tools"):
            # Qwen can emit a syntactically valid but empty tool invocation
            # after it has spent its response on reasoning.
            yield StreamEvent(
                "tool_call",
                {"name": "submit_card", "arguments": {}},
            )
            return
        payload = {
            "data": {"title": "Trial"},
            "provenance": {
                "/title": [{
                    "unit_id": self.unit_id,
                    "edge_key": "edge-1",
                    "source_document": "paper.pdf",
                    "quote": "Exact quote.",
                }]
            },
        }
        yield StreamEvent("content", {"delta": json.dumps(payload, ensure_ascii=False)})
        yield StreamEvent("done", {"final_content": ""})


@pytest.mark.asyncio
async def test_card_generation_falls_back_to_validated_raw_json_when_function_call_fails(tmp_path) -> None:
    store = AppStore(str(tmp_path / "fallback-cards.db"))
    await store.open()
    try:
        user = await store.create_user("fallback-user", "long fallback user password")
        conversation = await store.create_conversation(user.id)
        started = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "11111111-1111-4111-8111-111111111111",
            "Create a card",
        )
        template = (await store.list_card_templates(user.id))[0]
        recorded = await store.record_units(
            conversation["id"],
            started["userCheckpointId"],
            [{
                "chain_id": "c1",
                "edge_keys": ["edge-1"],
                "spine_evidence_seq": ["Exact quote."],
                "text": "UNIT",
                "walk": [{
                    "edge_key": "edge-1",
                    "evidence": "Exact quote.",
                    "source_file": "paper.pdf",
                }],
            }],
        )
        provider = _CardFallbackProvider(recorded[0]["unit_id"])
        pipeline = SimpleNamespace(
            llm=SimpleNamespace(
                provider_for=lambda _profile: provider,
                prompt_manager=SimpleNamespace(get_system_prompt=lambda _mode: "card prompt"),
                _context_fits=lambda **_kwargs: True,
            )
        )
        request = SimpleNamespace(
            app=SimpleNamespace(state=SimpleNamespace(app_store=store, pipeline=pipeline)),
            headers={},
        )

        events = [
            event
            async for event in _card_generation_events(
                request,
                TextProcessBody(
                    text="Create a card",
                    intent="generate_card",
                    template_version_id=template["latestVersion"]["id"],
                ),
                user=user,
                checkpoint_id=started["userCheckpointId"],
                model_history=[],
                inherited_context="",
                profile_name="qwen_cloud",
                think_effort=None,
            )
        ]

        draft_event = next(event for event in events if event.type == "card_draft")
        assert draft_event.data["draft"]["data"]["title"] == "Trial"
        assert len(provider.calls) == 2
        assert provider.calls[0]["tool_choice"] == "required"
        assert provider.calls[1]["tools"] is None
        assert provider.calls[0]["think_effort"] is None
        assert provider.calls[1]["think_effort"] == "off"
    finally:
        await store.close()

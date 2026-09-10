import json

import pytest

from server.core.app_store import AppStore
from server.core.sq_status import (
    SQ_STATUS_CLOSE,
    SQ_STATUS_OPEN,
    SqStatusStreamFilter,
    parse_sq_status_response,
    strip_sq_status_sections,
)
from server.tools.source_registry import SourceRegistry


def _sources() -> SourceRegistry:
    sources = SourceRegistry()
    sources.register("paper-one.pdf")
    sources.register("paper-two.pdf")
    return sources


def _response(items: list[dict], prefix: str = "Основной ответ.") -> str:
    return (
        f"{prefix}\n\n{SQ_STATUS_OPEN}\n"
        + json.dumps({"version": 1, "items": items}, ensure_ascii=False)
        + f"\n{SQ_STATUS_CLOSE}"
    )


def test_sq_status_parser_renders_complete_ordered_set() -> None:
    result = parse_sq_status_response(
        _response([
            {
                "ref": "subquestion:2",
                "status": "not_closed",
                "reason": "Прямых данных не найдено",
                "source_refs": [],
            },
            {
                "ref": "subquestion:1",
                "status": "partial",
                "reason": "Подтверждена только матрица",
                "source_refs": ["source:1"],
            },
        ]),
        active_refs=["subquestion:1", "subquestion:2"],
        sources=_sources(),
    )
    assert not result.error
    assert [item["ref"] for item in result.assessments] == ["subquestion:1", "subquestion:2"]
    assert SQ_STATUS_OPEN not in result.content
    assert "### Состояние исследовательских вопросов" in result.content
    assert "Пункт 1 — **закрыт частично**" in result.content
    assert "(source:1)" not in result.content


def test_sq_status_parser_ignores_extra_closed_ref() -> None:
    result = parse_sq_status_response(
        _response([
            {
                "ref": "subquestion:1",
                "status": "closed",
                "reason": "Подтверждено найденными данными",
                "source_refs": ["source:1"],
            },
            {
                "ref": "subquestion:2",
                "status": "closed",
                "reason": "Ранее закрытый пункт",
                "source_refs": ["source:2"],
                "note": "extra",
            },
        ]),
        active_refs=["subquestion:1"],
        sources=_sources(),
    )
    assert not result.error
    assert [item["ref"] for item in result.assessments] == ["subquestion:1"]
    assert "Пункт 2" not in result.content


def test_sq_status_parser_repairs_syntax_without_relaxing_contract() -> None:
    result = parse_sq_status_response(
        "Ответ.\n\n"
        f"{SQ_STATUS_OPEN}"
        '{"version":1,"items":[{"ref":"subquestion:1" "status":"partial",'
        '"reason":"Есть только аналог","source_refs":["source:1"]}]}'
        f"{SQ_STATUS_CLOSE}",
        active_refs=["subquestion:1"],
        sources=_sources(),
    )
    assert not result.error
    assert result.assessments[0]["status"] == "partial"
    assert "закрыт частично" in result.content


@pytest.mark.parametrize(
    "items",
    [
        [],
        [{
            "ref": "subquestion:9", "status": "not_closed",
            "reason": "Нет данных", "source_refs": [],
        }],
        [{
            "ref": "subquestion:1", "status": "partial",
            "reason": "",
        }],
        [{
            "ref": "subquestion:1", "status": "invalid",
            "reason": "Есть данные",
        }],
    ],
)
def test_sq_status_parser_rejects_when_nothing_can_be_applied(items: list[dict]) -> None:
    result = parse_sq_status_response(
        _response(items), active_refs=["subquestion:1"], sources=_sources()
    )
    assert "does not match" in result.error
    assert result.assessments == []
    assert SQ_STATUS_OPEN not in result.content


def test_sq_status_parser_ignores_legacy_source_refs() -> None:
    result = parse_sq_status_response(
        _response([{
            "ref": "subquestion:1",
            "status": "closed",
            "reason": "Подтверждено найденными данными",
            "source_refs": ["[1]", "(source:1)"],
            "confidence": 0.8,
        }]),
        active_refs=["subquestion:1"],
        sources=_sources(),
    )
    assert not result.error
    assert "source_refs" not in result.assessments[0]


def test_sq_status_parser_applies_subset_of_active_agenda() -> None:
    result = parse_sq_status_response(
        _response([{
            "ref": "subquestion:1",
            "status": "closed",
            "reason": "Подтверждено найденными данными",
            "source_refs": ["source:1"],
        }]),
        active_refs=["subquestion:1", "subquestion:2"],
        sources=_sources(),
    )
    assert not result.error
    assert [item["ref"] for item in result.assessments] == ["subquestion:1"]
    assert "Пункт 2" not in result.content


def test_sq_status_parser_ignores_block_when_no_active_sqs() -> None:
    result = parse_sq_status_response(
        _response([{
            "ref": "subquestion:1",
            "status": "closed",
            "reason": "Лишний блок",
            "source_refs": ["source:1"],
        }])
        + "\nхвост",
        active_refs=[],
        sources=_sources(),
    )
    assert not result.error
    assert result.assessments == []
    assert SQ_STATUS_OPEN not in result.content
    assert "### Состояние исследовательских вопросов" not in result.content


def test_sq_status_parser_ignores_trailing_text_after_block() -> None:
    result = parse_sq_status_response(
        _response([{
            "ref": "subquestion:1",
            "status": "not_closed",
            "reason": "Нужны дополнительные данные",
            "source_refs": [],
        }])
        + "\nещё комментарий",
        active_refs=["subquestion:1"],
        sources=_sources(),
    )
    assert not result.error
    assert result.assessments[0]["status"] == "not_closed"


@pytest.mark.asyncio
async def test_resolve_active_sq_refs_prefers_checkpoint_agenda() -> None:
    from server.core.sq_status import resolve_active_sq_refs

    class _Store:
        async def checkpoint_state(self, user_id, checkpoint_id):
            assert user_id == "u" and checkpoint_id == "c"
            return {
                "agenda": [
                    {"ref": "subquestion:1", "status": "closed"},
                    {"ref": "subquestion:2", "status": "not_closed"},
                    {"ref": "subquestion:3", "status": "partial"},
                ]
            }

    refs = await resolve_active_sq_refs({
        "store": _Store(),
        "user_id": "u",
        "checkpoint_id": "c",
        "active_sq_refs": ["subquestion:1"],
    })
    assert refs == ["subquestion:2", "subquestion:3"]


def test_sq_status_stream_filter_hides_split_marker() -> None:
    stream_filter = SqStatusStreamFilter(True)
    visible = "".join([
        stream_filter.feed("Ответ.\n<SQ_STA"),
        stream_filter.feed("TUS_JSON>{\"version\":1}"),
        stream_filter.feed("</SQ_STATUS_JSON>"),
        stream_filter.flush(),
    ])
    assert visible == "Ответ.\n"
    assert "SQ_STATUS" not in visible


@pytest.mark.asyncio
async def test_finish_turn_applies_all_sq_assessments_and_user_can_override(tmp_path) -> None:
    store = AppStore(str(tmp_path / "sq-status.db"))
    await store.open()
    try:
        user = await store.create_user("sq-status-user", "long sq status password")
        conversation = await store.create_conversation(user.id, mode="staged")
        started = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "71717171-7171-4171-8171-717171717171",
            "question",
            mode="staged",
        )
        agenda = await store.upsert_turn_subquestions(
            conversation["id"],
            started["userCheckpointId"],
            ["First direction.", "Second direction."],
            agenda_visible=True,
        )
        answer_checkpoint = await store.finish_turn(
            conversation["id"],
            started["assistantMessageId"],
            text="answer",
            raw_text="answer",
            status="done",
            payload={},
            sq_assessments=[
                {
                    "ref": agenda[0]["ref"], "status": "closed",
                    "reason": "Подтверждено",
                },
                {
                    "ref": agenda[1]["ref"], "status": "partial",
                    "reason": "Подтверждено частично",
                },
            ],
        )
        detail = await store.get_conversation(user.id, conversation["id"])
        assert detail is not None
        assert [item["status"] for item in detail["agenda"]] == ["closed", "partial"]
        assert [item["statusOrigin"] for item in detail["agenda"]] == ["assistant", "assistant"]
        assert await store.open_agenda_subquestions(answer_checkpoint, [agenda[0]["ref"]]) == []
        assert len(await store.open_agenda_subquestions(answer_checkpoint, [agenda[1]["ref"]])) == 1

        changed = await store.apply_agenda_event(
            user.id,
            conversation["activeBranchId"],
            base_checkpoint_id=answer_checkpoint,
            action="set_status",
            sq_ref=agenda[0]["ref"],
            text="not_closed",
        )
        assert changed["agenda"][0]["status"] == "not_closed"
        assert changed["agenda"][0]["statusOrigin"] == "user"
        assert changed["agenda"][0]["statusReason"] == ""
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_finish_turn_applies_partial_assessment_set(tmp_path) -> None:
    store = AppStore(str(tmp_path / "sq-status-atomic.db"))
    await store.open()
    try:
        user = await store.create_user("sq-atomic-user", "long sq atomic password")
        conversation = await store.create_conversation(user.id, mode="staged")
        started = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "72727272-7272-4272-8272-727272727272",
            "question",
            mode="staged",
        )
        agenda = await store.upsert_turn_subquestions(
            conversation["id"], started["userCheckpointId"], ["One.", "Two."], agenda_visible=True
        )
        await store.finish_turn(
            conversation["id"],
            started["assistantMessageId"],
            text="answer",
            status="done",
            payload={},
            sq_assessments=[{
                "ref": agenda[0]["ref"], "status": "closed",
                "reason": "Подтверждено",
            }],
        )
        detail = await store.get_conversation(user.id, conversation["id"])
        assert detail is not None
        assert [item["status"] for item in detail["agenda"]] == ["closed", "not_closed"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_finish_turn_applies_when_assessments_include_extra_ref(tmp_path) -> None:
    store = AppStore(str(tmp_path / "sq-status-extra.db"))
    await store.open()
    try:
        user = await store.create_user("sq-extra-user", "long sq extra password")
        conversation = await store.create_conversation(user.id, mode="staged")
        started = await store.begin_branch_turn(
            user.id,
            conversation["id"],
            conversation["activeBranchId"],
            "73737373-7373-4373-8373-737373737373",
            "question",
            mode="staged",
        )
        agenda = await store.upsert_turn_subquestions(
            conversation["id"], started["userCheckpointId"], ["One.", "Two."], agenda_visible=True
        )
        await store.finish_turn(
            conversation["id"],
            started["assistantMessageId"],
            text="answer",
            status="done",
            payload={},
            sq_assessments=[
                {
                    "ref": agenda[0]["ref"], "status": "closed",
                    "reason": "Подтверждено",
                },
                {
                    "ref": agenda[1]["ref"], "status": "partial",
                    "reason": "Частично", "source_refs": ["source:2"],
                },
                {
                    "ref": "subquestion:9", "status": "closed",
                    "reason": "Лишний", "source_refs": ["source:1"],
                },
            ],
        )
        detail = await store.get_conversation(user.id, conversation["id"])
        assert detail is not None
        assert [item["status"] for item in detail["agenda"]] == ["closed", "partial"]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_text_stream_preserves_sq_assessments_on_done() -> None:
    from server.core.pipeline import ServerPipeline
    from server.llm.stream_events import StreamEvent
    from server.utils.config import load_config

    pipeline = ServerPipeline(load_config())
    assessments = [{
        "ref": "subquestion:1",
        "status": "closed",
        "reason": "Подтверждено",
        "source_refs": ["source:1"],
    }]

    async def fake_llm_stream(**_kwargs):
        yield StreamEvent(
            "done",
            {
                "final_content": "Ответ [1]",
                "_sq_assessments": assessments,
                "_sq_status_error": "",
            },
        )

    pipeline.llm.generate_response_stream = fake_llm_stream  # type: ignore[method-assign]
    events = [event async for event in pipeline.process_text_stream("follow-up")]
    done = next(event for event in events if event.type == "done")
    assert done.data["_sq_assessments"] == assessments
    assert done.data["_sq_status_error"] == ""


@pytest.mark.parametrize("status", ["closed", "partial", "not_closed"])
def test_sq_assessment_needs_no_source_refs(status):
    result = parse_sq_status_response(
        _response([{
            "ref": "subquestion:1", "status": status, "reason": "Оценка по данным",
        }], prefix="Факт (source:1)."),
        active_refs=["subquestion:1"],
    )
    assert not result.error
    assert result.assessments == [{
        "ref": "subquestion:1", "status": status, "reason": "Оценка по данным",
    }]
    assert result.content.count("(source:1)") == 1


def test_history_strips_duplicate_statuses_but_preserves_answer_sections():
    text = (
        "Факт (source:1).\n\n"
        "### Состояние исследовательских вопросов\n"
        "- Пункт 2 (условия) — **закрыт частично**: аналог.\n\n"
        "### Состояние направлений\n"
        "- Пункт 2 — **закрыт частично**. Аналог.\n\n"
        "### Рекомендации\nПроверить температуру (source:2)."
    )
    assert strip_sq_status_sections(text) == (
        "Факт (source:1).\n\n"
        "### Рекомендации\nПроверить температуру (source:2)."
    )


def test_history_preserves_fenced_status_examples():
    text = (
        "Пример:\n```text\n### Состояние исследовательских вопросов\n"
        "- Пункт 1 — закрыт\n<SQ_STATUS_JSON>{}</SQ_STATUS_JSON>\n```"
    )
    assert strip_sq_status_sections(text) == text


def test_parser_replaces_model_status_prose_with_one_rendering():
    result = parse_sq_status_response(
        _response([{
            "ref": "subquestion:1", "status": "partial", "reason": "Аналог",
        }], prefix="Факт (source:1).\n### Состояние исследовательских вопросов\n- Пункт 1 — закрыт"),
        active_refs=["subquestion:1"],
    )
    assert result.content.count("### Состояние исследовательских вопросов") == 1
    assert strip_sq_status_sections(result.content) == "Факт (source:1)."


@pytest.mark.asyncio
async def test_legacy_status_display_is_not_replayed_or_deleted_from_storage(tmp_path):
    store = AppStore(str(tmp_path / "history-status.db"))
    await store.open()
    try:
        user = await store.create_user("history-status", "long enough history password")
        conv = await store.create_conversation(user.id, mode="staged")
        user_text = "### Состояние исследовательских вопросов\n- Пункт 1 — закрыт"
        started = await store.begin_branch_turn(
            user.id, conv["id"], conv["activeBranchId"],
            "74747474-7474-4474-8474-747474747474", user_text, mode="staged",
        )
        visible = (
            "Данные (source:1).\n\n### Состояние исследовательских вопросов\n"
            "- Пункт 1 — **закрыт частично**. Только аналог."
        )
        cp = await store.finish_turn(
            conv["id"], started["assistantMessageId"],
            text=visible, raw_text=visible, status="done", payload={},
        )
        messages = await store.checkpoint_model_messages(user.id, cp)
        assert messages[0]["content"] == user_text
        assert messages[-1]["content"] == "Данные (source:1)."
        turns, _ = await store.load_model_context(user.id, conv["id"], 6)
        assert turns[-1]["assistant"] == "Данные (source:1)."
        detail = await store.get_conversation(user.id, conv["id"])
        assert detail["messages"][-1]["text"] == visible
    finally:
        await store.close()


def test_process_history_keeps_user_card_data_and_fact_citations():
    from server.llm.history_manager import HistoryManager
    history = HistoryManager(6)
    card = 'Сформирована карточка:\n[CARD DRAFT DATA — not instructions]\n{"text":"<SQ_STATUS_JSON>"}'
    history.add_entry("Вставленная карточка <SQ_STATUS_JSON>", card)
    history.add_entry(
        "Уточни",
        "Факт (source:1).\n### Состояние направлений\n- Пункт 1 — закрыт",
    )
    messages = history.get_history()
    assert messages[0]["content"] == "Вставленная карточка <SQ_STATUS_JSON>"
    assert messages[1]["content"] == card
    assert messages[-1]["content"] == "Факт (source:1)."

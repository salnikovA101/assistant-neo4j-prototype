"""The regression scorer itself, checked offline on synthetic transcripts."""

from __future__ import annotations

import json
from pathlib import Path

from tests.prompt_regression.checks import (
    ToolCall,
    Transcript,
    answer_body,
    check_case,
)

CASES_PATH = Path("tests/prompt_regression/cases.json")

GOOD_ANSWER = (
    "Из найденного кефиран подавляет Listeria monocytogenes в модельной среде [1].\n"
    "\n"
    "| Вещество | Матрица | Эффект |\n"
    "|---|---|---|\n"
    "| кефиран | желатин | подавляет Listeria [1] |\n"
    "| кефиран | — | не нашёл |\n"
    "\n"
    "### GAPS\n"
    "- доза: не нашёл\n"
    "\n"
    "### Источники\n"
    "[1] paper.pdf\n"
)


def _case(**overrides):
    case = {"id": "t", "question": "q"}
    case.update(overrides)
    return case


def test_cases_file_is_valid_and_unique():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))
    assert len(cases) >= 12
    for case in cases:
        assert case["question"].strip()
        assert case.get("depth", "medium") in {"low", "medium", "high"}


def test_answer_body_strips_server_bibliography():
    body = answer_body(GOOD_ANSWER)
    assert "### Источники" not in body
    assert "paper.pdf" not in body
    assert "### GAPS" in body


def test_clean_answer_passes():
    transcript = Transcript(
        answer=GOOD_ANSWER,
        tool_calls=[ToolCall("ask_subgraph", ["Kefiran inhibits Listeria."])],
    )
    result = check_case(_case(), transcript)
    assert result.ok, result.failures
    assert result.warnings == []


def test_service_leaks_fail():
    transcript = Transcript(
        answer="UNIT [3] говорит, что кефиран активен (conf=0.9) [1].\n### GAPS\nнет",
        tool_calls=[ToolCall("ask_subgraph", ["Kefiran inhibits Listeria."])],
    )
    result = check_case(_case(), transcript)
    assert not result.ok
    assert any("UNIT [" in f for f in result.failures)
    assert any("conf=" in f for f in result.failures)


def test_completeness_claim_fails():
    transcript = Transcript(
        answer="Это единственный индикатор в базе [1].\n### GAPS\nпробелов нет",
        tool_calls=[ToolCall("ask_subgraph", ["Freshness indicators change colour."])],
    )
    result = check_case(_case(), transcript)
    assert any("полноте базы" in f for f in result.failures)


def test_cyrillic_and_question_subquestions_fail():
    transcript = Transcript(
        answer=GOOD_ANSWER,
        tool_calls=[
            ToolCall("ask_subgraph", ["закваски для творога", "What is used?"]),
        ],
    )
    result = check_case(_case(), transcript)
    assert any("кириллица" in f for f in result.failures)
    assert any("вопрос вместо утверждения" in f for f in result.failures)


def test_repeat_across_calls_fails():
    call = ["Kefiran inhibits Listeria."]
    transcript = Transcript(
        answer=GOOD_ANSWER,
        tool_calls=[ToolCall("ask_subgraph", call), ToolCall("ask_subgraph", list(call))],
    )
    result = check_case(_case(), transcript)
    assert any("повтор фразы" in f for f in result.failures)


def test_budget_overrun_fails():
    transcript = Transcript(
        answer=GOOD_ANSWER,
        tool_calls=[
            ToolCall("ask_subgraph", ["A statement one."]),
            ToolCall("ask_subgraph", ["A statement two."]),
            ToolCall("ask_subgraph", ["A statement three."]),
        ],
    )
    result = check_case(_case(), transcript)
    assert any("при лимите 2" in f for f in result.failures)


def test_unexpected_tool_call_fails():
    transcript = Transcript(
        answer="Привет, помогу с данными по базе.",
        tool_calls=[ToolCall("ask_subgraph", ["Greeting statement."])],
    )
    result = check_case(
        _case(expect_tool_call=False, require_citations=False, require_gaps=False),
        transcript,
    )
    assert any("хотя не нужен" in f for f in result.failures)


def test_missing_gaps_and_citations_are_reported():
    transcript = Transcript(
        answer="Кефиран подавляет Listeria в модельной среде без каких-либо ссылок.",
        tool_calls=[ToolCall("ask_subgraph", ["Kefiran inhibits Listeria."])],
    )
    result = check_case(_case(), transcript)
    assert any("### GAPS" in f for f in result.failures)
    assert any("ссылки" in w for w in result.warnings)


def test_must_contain_and_rubric_are_honoured():
    transcript = Transcript(
        answer=GOOD_ANSWER,
        tool_calls=[ToolCall("ask_subgraph", ["Kefiran inhibits Listeria."])],
    )
    result = check_case(
        _case(must_contain=["### Рекомендации"], rubric=["проверить руками"]),
        transcript,
    )
    assert any("### Рекомендации" in f for f in result.failures)
    assert result.rubric == ["проверить руками"]


def test_empty_answer_fails_fast():
    result = check_case(_case(), Transcript(answer="   "))
    assert result.failures == ["пустой ответ"]

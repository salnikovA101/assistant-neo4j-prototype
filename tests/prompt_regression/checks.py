"""Automatic scoring for prompt regression cases.

Pure functions: a case plus a captured turn transcript in, failures out. The
live SSE client lives in `run.py`; keeping the rules here makes them testable
without a server.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")
BIBLIO_RE = re.compile(r"(?ms)^[ \t]*###[ \t]*Источники[ \t]*\n.*\Z")
CITATION_RE = re.compile(r"\[\d+\]")
WS_RE = re.compile(r"\s+")

# Service artefacts that must never reach the user.
LEAKED_ARTEFACTS: tuple[str, ...] = (
    "conf=",
    "UNIT [",
    "[?]",
    ".pdf",
    "ask_subgraph",
    "source:",
    "TOOL_ERROR",
    "NO_RESULTS",
)

# Claims about corpus completeness the assistant cannot make.
COMPLETENESS_CLAIMS: tuple[str, ...] = (
    "единственный",
    "единственная",
    "в базе нет",
    "не существует",
    "из общих знаний",
)

MAX_SUBQUESTIONS = 6


@dataclass
class ToolCall:
    name: str
    subquestions: list[str] = field(default_factory=list)


@dataclass
class Transcript:
    """What one user turn produced: the rendered answer and the tool calls."""

    answer: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    def searches(self) -> list[ToolCall]:
        return [tc for tc in self.tool_calls if tc.name == "ask_subgraph"]


@dataclass
class CaseResult:
    case_id: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    rubric: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def answer_body(answer: str) -> str:
    """Answer without the server-appended bibliography."""
    return BIBLIO_RE.sub("", answer or "").strip()


def subquestion_key(text: str) -> str:
    return WS_RE.sub(" ", str(text).strip().lower()).strip(" .?!")


def _check_tool_usage(case: dict[str, Any], transcript: Transcript) -> list[str]:
    failures: list[str] = []
    searches = transcript.searches()
    expects_tool = bool(case.get("expect_tool_call", True))

    if expects_tool and not searches:
        failures.append("инструмент не вызван, хотя вопрос требует поиска")
    if not expects_tool and searches:
        failures.append(f"инструмент вызван {len(searches)} раз(а), хотя не нужен")

    limit = int(case.get("max_tool_calls", 2))
    if len(searches) > limit:
        failures.append(f"{len(searches)} вызовов поиска при лимите {limit}")

    seen: dict[str, int] = {}
    for i, call in enumerate(searches, 1):
        if not call.subquestions:
            failures.append(f"вызов {i}: пустой список subquestions")
        if len(call.subquestions) > MAX_SUBQUESTIONS:
            failures.append(
                f"вызов {i}: {len(call.subquestions)} фраз при лимите {MAX_SUBQUESTIONS}"
            )
        for sq in call.subquestions:
            if CYRILLIC_RE.search(sq):
                failures.append(f"вызов {i}: кириллица в sq — {sq[:60]}")
            if "?" in sq:
                failures.append(f"вызов {i}: вопрос вместо утверждения — {sq[:60]}")
            key = subquestion_key(sq)
            if key in seen:
                failures.append(
                    f"вызов {i}: повтор фразы из вызова {seen[key]} — {sq[:60]}"
                )
            else:
                seen[key] = i
    return failures


def _check_answer(case: dict[str, Any], body: str) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    low = body.lower()

    for artefact in LEAKED_ARTEFACTS:
        if artefact.lower() in low:
            failures.append(f"служебная утечка в ответе: {artefact}")
    for claim in COMPLETENESS_CLAIMS:
        if claim in low:
            failures.append(f"заявление о полноте базы: «{claim}»")

    for needle in case.get("must_contain", []):
        if needle.lower() not in low:
            failures.append(f"нет обязательного фрагмента: «{needle}»")
    for needle in case.get("must_not_contain", []):
        if needle.lower() in low:
            failures.append(f"есть запрещённый фрагмент: «{needle}»")

    if case.get("require_citations", True) and not CITATION_RE.search(body):
        warnings.append("в ответе нет ни одной ссылки [n]")
    if case.get("require_gaps", True) and "### gaps" not in low:
        failures.append("нет раздела ### GAPS")
    return failures, warnings


def check_case(case: dict[str, Any], transcript: Transcript) -> CaseResult:
    """Score one case. Rubric items are for human review, never auto-failed."""
    body = answer_body(transcript.answer)
    result = CaseResult(case_id=str(case.get("id") or "?"))

    if not body:
        result.failures.append("пустой ответ")
        return result

    result.failures.extend(_check_tool_usage(case, transcript))
    answer_failures, warnings = _check_answer(case, body)
    result.failures.extend(answer_failures)
    result.warnings.extend(warnings)
    result.rubric = list(case.get("rubric", []))
    return result

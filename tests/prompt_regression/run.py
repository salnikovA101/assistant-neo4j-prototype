"""Prompt regression harness: run cases against a live server and score them.

Needs a running backend (Neo4j + local Ollama embeddings + LLM):

    .venv/bin/python -m tests.prompt_regression.run
    .venv/bin/python -m tests.prompt_regression.run --case catalog_freshness_indicators

Automatic checks are in `checks.py`; rubric items are printed for human review.
Report → tests/reports/prompt_regression/report.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

from tests.prompt_regression.checks import (
    CaseResult,
    ToolCall,
    Transcript,
    answer_body,
    check_case,
)

CASES_PATH = Path(__file__).with_name("cases.json")
REPORT_DIR = Path("tests/reports/prompt_regression")


def load_cases(path: Path, only: str = "") -> list[dict[str, Any]]:
    cases = json.loads(path.read_text(encoding="utf-8"))
    if only:
        cases = [c for c in cases if str(c.get("id")) == only]
    return cases


def _parse_sse(chunk_lines: list[str]) -> tuple[str, dict[str, Any]] | None:
    event = ""
    data = ""
    for line in chunk_lines:
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data = line[5:].strip()
    if not event:
        return None
    try:
        payload = json.loads(data) if data else {}
    except json.JSONDecodeError:
        payload = {}
    return event, payload


def _tool_call_from_event(payload: dict[str, Any]) -> ToolCall:
    args = payload.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {}
    if not isinstance(args, dict):
        args = {}
    raw = args.get("subquestions") or []
    sqs = [str(s) for s in raw] if isinstance(raw, list) else []
    return ToolCall(name=str(payload.get("name") or ""), subquestions=sqs)


def run_turn(
    client: httpx.Client,
    base_url: str,
    session_id: str,
    question: str,
    depth: str,
) -> Transcript:
    """Send one user turn, collect tool calls and the final rendered answer."""
    transcript = Transcript(answer="")
    body = {"text": question, "search_depth": depth}
    with client.stream(
        "POST",
        f"{base_url}/process_text_stream",
        json=body,
        headers={"X-Session-Id": session_id},
    ) as response:
        response.raise_for_status()
        buffer: list[str] = []
        for line in response.iter_lines():
            if line:
                buffer.append(line)
                continue
            parsed = _parse_sse(buffer)
            buffer = []
            if parsed is None:
                continue
            event, payload = parsed
            if event == "tool_call":
                transcript.tool_calls.append(_tool_call_from_event(payload))
            elif event == "done":
                transcript.answer = str(payload.get("final_content") or "")
            elif event == "error":
                transcript.answer = f"[stream error] {payload.get('message')}"
    return transcript


def run_case(
    client: httpx.Client,
    base_url: str,
    case: dict[str, Any],
) -> tuple[CaseResult, Transcript]:
    created = client.post(f"{base_url}/api/conversations")
    created.raise_for_status()
    session_id = str(created.json()["id"])
    depth = str(case.get("depth") or "medium")
    transcript = run_turn(
        client, base_url, session_id, str(case["question"]), depth
    )
    follow_up = case.get("follow_up")
    if follow_up:
        transcript = run_turn(client, base_url, session_id, str(follow_up), depth)
    return check_case(case, transcript), transcript


def render_report(
    results: list[tuple[CaseResult, Transcript]],
    base_url: str,
) -> str:
    passed = sum(1 for r, _ in results if r.ok)
    lines = [
        "# Prompt regression",
        "",
        f"Сервер: `{base_url}`",
        f"Пройдено автоматических проверок: **{passed}/{len(results)}**",
        "",
    ]
    for result, transcript in results:
        status = "PASS" if result.ok else "FAIL"
        lines.append(f"## {result.case_id} — {status}")
        lines.append("")
        searches = transcript.searches()
        lines.append(f"Вызовов поиска: {len(searches)}")
        for i, call in enumerate(searches, 1):
            for sq in call.subquestions:
                lines.append(f"- вызов {i}: `{sq}`")
        lines.append("")
        if result.failures:
            lines.append("**Ошибки:**")
            lines.extend(f"- {item}" for item in result.failures)
            lines.append("")
        if result.warnings:
            lines.append("**Предупреждения:**")
            lines.extend(f"- {item}" for item in result.warnings)
            lines.append("")
        if result.rubric:
            lines.append("**Проверить глазами:**")
            lines.extend(f"- [ ] {item}" for item in result.rubric)
            lines.append("")
        lines.append("<details><summary>Ответ</summary>")
        lines.append("")
        lines.append(answer_body(transcript.answer) or "(пусто)")
        lines.append("")
        lines.append("</details>")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--cases", default=str(CASES_PATH))
    parser.add_argument("--case", default="", help="run a single case id")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--username", default=os.getenv("ASSISTANT_USER", ""))
    parser.add_argument("--password", default=os.getenv("ASSISTANT_PASSWORD", ""))
    parser.add_argument("--list", action="store_true", help="print case ids and exit")
    args = parser.parse_args(argv)

    cases = load_cases(Path(args.cases), args.case)
    if not cases:
        print("Нет кейсов для запуска", file=sys.stderr)
        return 2
    if args.list:
        for case in cases:
            print(case["id"])
        return 0
    if not args.username or not args.password:
        print("Задайте ASSISTANT_USER и ASSISTANT_PASSWORD", file=sys.stderr)
        return 2

    results: list[tuple[CaseResult, Transcript]] = []
    with httpx.Client(timeout=args.timeout, auth=(args.username, args.password)) as client:
        for case in cases:
            print(f"→ {case['id']}", flush=True)
            result, transcript = run_case(client, args.base_url, case)
            status = "PASS" if result.ok else "FAIL"
            print(f"  {status} {'; '.join(result.failures)}".rstrip(), flush=True)
            results.append((result, transcript))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "report.md"
    report_path.write_text(render_report(results, args.base_url), encoding="utf-8")
    print(f"\nОтчёт: {report_path}")

    failed = [r.case_id for r, _ in results if not r.ok]
    if failed:
        print(f"Провалено: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

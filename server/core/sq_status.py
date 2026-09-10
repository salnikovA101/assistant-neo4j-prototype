"""Parse and render staged SQ coverage assessments emitted by the model."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from server.tools.source_registry import SourceRegistry


SQ_STATUS_OPEN = "<SQ_STATUS_JSON>"
SQ_STATUS_CLOSE = "</SQ_STATUS_JSON>"
SQ_COVERAGE_STATUSES = frozenset({"closed", "partial", "not_closed"})
SQ_STATUS_USER_NOTICE = (
    "Не удалось автоматически обновить статусы в плане. "
    "Текст ответа сохранён, статусы можно поправить вручную."
)
_SQ_REF_RE = re.compile(r"subquestion:(\d+)")
_REQUIRED_ITEM_KEYS = frozenset({"ref", "status", "reason"})
_STATUS_LABELS = {
    "closed": "закрыт",
    "partial": "закрыт частично",
    "not_closed": "не закрыт",
}


def strip_sq_status_sections(content: str) -> str:
    """Remove service status output from assistant prose, including old history.

    Only recognize dedicated status headings with status bullets. Preserve other
    sections, fenced examples, user messages and structured card payloads (callers
    must apply this helper only to ordinary assistant text).
    """
    lines = (content or "").splitlines(keepends=True)
    headings: list[tuple[int, int, str]] = []
    service_start: int | None = None
    fence = ""
    for index, line in enumerate(lines):
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            token = marker.group(1)
            if not fence:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = ""
            continue
        if fence:
            continue
        if SQ_STATUS_OPEN in line:
            service_start = index
            # Keep any answer text before a marker on the same line.
            lines[index] = line.split(SQ_STATUS_OPEN, 1)[0]
            break
        heading = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*#*\s*$", line)
        if heading:
            headings.append((index, len(heading.group(1)), heading.group(2).strip()))
    end = service_start + 1 if service_start is not None else len(lines)
    removed: set[int] = set()
    names = {"состояние исследовательских вопросов", "состояние направлений"}
    for position, (start, level, title) in enumerate(headings):
        if title.lower() not in names:
            continue
        stop = next(
            (i for i, depth, _ in headings[position + 1:] if depth <= level),
            end,
        )
        body = "".join(lines[start + 1:stop])
        if not re.search(
            r"(?m)^\s*[-*]\s+(?:Пункт\s+\d+|(?:SQ|subquestion:)\s*\d+).*"
            r"(?:закрыт|не закрыт|closed|partial|not_closed)",
            body,
            re.IGNORECASE,
        ):
            continue
        removed.update(range(start, stop))
    return "".join(line for i, line in enumerate(lines[:end]) if i not in removed).rstrip()


async def resolve_active_sq_refs(turn_context: dict[str, Any] | None) -> list[str]:
    """Prefer the checkpoint agenda at done-time over the turn-start snapshot.

    Search can add SQs after the stream starts; parsing against the stale
    snapshot would drop a valid block or warn when nothing was open at start.
    """
    ctx = turn_context or {}
    snapshot = [
        str(ref).strip()
        for ref in list(ctx.get("active_sq_refs") or [])
        if str(ref).strip()
    ]
    store = ctx.get("store")
    user_id = ctx.get("user_id")
    checkpoint_id = ctx.get("checkpoint_id")
    getter = getattr(store, "checkpoint_state", None)
    if getter is None or not user_id or not checkpoint_id:
        return snapshot
    try:
        state = await getter(user_id, checkpoint_id)
    except Exception:
        return snapshot
    if not state:
        return snapshot
    return [
        str(item.get("ref") or "").strip()
        for item in list(state.get("agenda") or [])
        if item.get("status") in {"not_closed", "partial"} and str(item.get("ref") or "").strip()
    ]


@dataclass(frozen=True)
class SqStatusParseResult:
    content: str
    assessments: list[dict[str, Any]]
    error: str = ""


def _strip_service_block(content: str) -> tuple[str, str | None, str]:
    """Return visible prefix, raw JSON body, and a structural error."""
    raw = content or ""
    start = raw.find(SQ_STATUS_OPEN)
    if start < 0:
        return raw.rstrip(), None, ""
    prefix = raw[:start].rstrip()
    end = raw.find(SQ_STATUS_CLOSE, start + len(SQ_STATUS_OPEN))
    if end < 0:
        return prefix, None, "unterminated SQ_STATUS_JSON block"
    if raw.find(SQ_STATUS_OPEN, start + len(SQ_STATUS_OPEN)) >= 0:
        return prefix, None, "multiple SQ_STATUS_JSON blocks"
    body = raw[start + len(SQ_STATUS_OPEN):end].strip()
    return prefix, body, ""


def parse_sq_status_response(
    content: str,
    *,
    active_refs: Iterable[str],
    sources: SourceRegistry | None = None,
) -> SqStatusParseResult:
    """Parse a trailing SQ status block and render a visible summary.

    Extra keys, extra refs, and a subset of the active agenda are accepted so a
    slightly off-spec model turn still updates the points it did assess. A
    warning is only returned when the block cannot be applied at all.
    """
    visible, body, structural_error = _strip_service_block(content)
    expected = list(dict.fromkeys(str(ref).strip() for ref in active_refs if str(ref).strip()))
    expected_set = set(expected)
    if not expected:
        return SqStatusParseResult(visible, [], "")
    if structural_error:
        return SqStatusParseResult(visible, [], structural_error)
    if body is None:
        return SqStatusParseResult(visible, [], "")
    try:
        payload = json.loads(body)
    except (TypeError, ValueError) as exc:
        # Qwen occasionally emits a syntactically damaged trailing object
        # (usually a missing comma) even when all fields are present. Repair
        # only the JSON syntax, then run the same schema/ref
        # validation below; unusable payloads still fail closed.
        try:
            from json_repair import repair_json

            payload = json.loads(repair_json(body))
        except (ImportError, TypeError, ValueError) as repair_exc:
            return SqStatusParseResult(
                visible,
                [],
                f"invalid SQ_STATUS_JSON: {exc}; repair failed: {repair_exc}",
            )
    if not isinstance(payload, dict) or payload.get("version") != 1:
        return SqStatusParseResult(visible, [], "unsupported SQ status payload")
    items = payload.get("items")
    if not isinstance(items, list):
        return SqStatusParseResult(visible, [], "SQ status items must be a list")

    assessments: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        ref = str(item.get("ref") or "").strip()
        if _SQ_REF_RE.fullmatch(ref) is None or ref in assessments:
            continue
        if ref not in expected_set:
            continue
        if not _REQUIRED_ITEM_KEYS.issubset(item):
            continue
        status = str(item.get("status") or "").strip()
        reason = " ".join(str(item.get("reason") or "").split())
        if status not in SQ_COVERAGE_STATUSES:
            continue
        if not reason or len(reason) > 500:
            continue
        assessments[ref] = {
            "ref": ref,
            "status": status,
            "reason": reason,
        }

    ordered = [assessments[ref] for ref in expected if ref in assessments]
    if not ordered:
        return SqStatusParseResult(visible, [], "SQ status set does not match active agenda")

    lines = ["### Состояние исследовательских вопросов", ""]
    for item in ordered:
        number = _SQ_REF_RE.fullmatch(item["ref"]).group(1)  # type: ignore[union-attr]
        reason = item["reason"].rstrip(". ") + "."
        lines.append(f"- Пункт {number} — **{_STATUS_LABELS[item['status']]}**. {reason}")
    # If the model also wrote visible statuses, replace them with one canonical
    # rendering only after valid assessments have been recovered.
    visible = strip_sq_status_sections(visible)
    rendered = "\n\n".join(part for part in (visible, "\n".join(lines)) if part.strip())
    return SqStatusParseResult(rendered, ordered)


class SqStatusStreamFilter:
    """Hide a trailing service block while preserving normal streamed content."""

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.pending = ""
        self.suppressing = False

    def feed(self, delta: str) -> str:
        if not self.enabled or self.suppressing:
            return "" if self.suppressing else delta
        self.pending += delta
        marker_at = self.pending.find(SQ_STATUS_OPEN)
        if marker_at >= 0:
            visible = self.pending[:marker_at]
            self.pending = ""
            self.suppressing = True
            return visible
        keep = len(SQ_STATUS_OPEN) - 1
        if len(self.pending) <= keep:
            return ""
        visible, self.pending = self.pending[:-keep], self.pending[-keep:]
        return visible

    def flush(self) -> str:
        if not self.enabled or self.suppressing:
            self.pending = ""
            return ""
        visible = self.pending
        self.pending = ""
        return visible

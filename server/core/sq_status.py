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
_SOURCE_REF_LOOSE_RE = re.compile(r"^\(?\s*source\s*:?\s*(\d+)\s*\)?$", re.IGNORECASE)
_DISPLAY_OR_BARE_REF_RE = re.compile(r"^\[(\d+)\]$|^(\d+)$")
_REQUIRED_ITEM_KEYS = frozenset({"ref", "status", "reason", "source_refs"})
_STATUS_LABELS = {
    "closed": "закрыт",
    "partial": "закрыт частично",
    "not_closed": "не закрыт",
}


def _coerce_source_ref(raw: Any, sources: SourceRegistry) -> str | None:
    """Accept source:N, (source:N), [N], or N when that session id exists."""
    text = str(raw or "").strip()
    if not text:
        return None
    match = _SOURCE_REF_LOOSE_RE.fullmatch(text)
    if match is None:
        display = _DISPLAY_OR_BARE_REF_RE.fullmatch(text)
        if display is None:
            return None
        sid = int(display.group(1) or display.group(2))
    else:
        sid = int(match.group(1))
    if sources.resolve(sid) is None:
        return None
    return f"source:{sid}"


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
        if item.get("status") != "closed" and str(item.get("ref") or "").strip()
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
    sources: SourceRegistry,
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
        # only the JSON syntax, then run the same schema/ref/source
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
        raw_source_refs = item.get("source_refs")
        if status not in SQ_COVERAGE_STATUSES:
            continue
        if not reason or len(reason) > 500:
            continue
        if not isinstance(raw_source_refs, list):
            continue
        source_refs: list[str] = []
        for raw_ref in raw_source_refs:
            source_ref = _coerce_source_ref(raw_ref, sources)
            if source_ref is None or source_ref in source_refs:
                continue
            source_refs.append(source_ref)
        if status in {"closed", "partial"} and not source_refs:
            continue
        assessments[ref] = {
            "ref": ref,
            "status": status,
            "reason": reason,
            "source_refs": source_refs,
        }

    ordered = [assessments[ref] for ref in expected if ref in assessments]
    if not ordered:
        return SqStatusParseResult(visible, [], "SQ status set does not match active agenda")

    lines = ["### Состояние направлений", ""]
    for item in ordered:
        number = _SQ_REF_RE.fullmatch(item["ref"]).group(1)  # type: ignore[union-attr]
        reason = item["reason"].rstrip(". ") + "."
        citations = ""
        if item["source_refs"]:
            citations = " (" + "; ".join(item["source_refs"]) + ")"
        lines.append(f"- Пункт {number} — **{_STATUS_LABELS[item['status']]}**. {reason}{citations}")
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

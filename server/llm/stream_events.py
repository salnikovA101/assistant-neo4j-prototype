"""SSE stream events + multi-provider chat.completion delta normalization."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from server.utils.config import OpenAIProfile

StreamEventType = Literal[
    "thinking",
    "tool_call",
    "tool_result",
    "content",
    "graph_highlight",
    "done",
    "error",
]

# Inline think tags that may appear in content deltas (Gemma / Ollama fallbacks).
_OPEN_THINK_RE = re.compile(
    r"(?s)^(?:<think>|<\|think\|>|<\|channel\>thought)"
)
_CLOSE_THINK_RE = re.compile(
    r"(?s)(?:</think>|<channel\|>)"
)


@dataclass
class StreamEvent:
    type: StreamEventType
    data: Dict[str, Any] = field(default_factory=dict)

    def to_sse(self) -> str:
        import json

        return f"event: {self.type}\ndata: {json.dumps(self.data, ensure_ascii=False)}\n\n"


@dataclass
class NormalizedDelta:
    thinking: str = ""
    content: str = ""
    # Raw OpenAI-style tool_call delta objects (with index).
    tool_call_deltas: List[Dict[str, Any]] = field(default_factory=list)


def _delta_as_dict(delta: Any) -> Dict[str, Any]:
    if delta is None:
        return {}
    if isinstance(delta, dict):
        return delta
    dump = getattr(delta, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=False) or {}
        except TypeError:
            return dump() or {}
    extra = getattr(delta, "model_extra", None) or {}
    out: Dict[str, Any] = {}
    for key in (
        "content",
        "role",
        "tool_calls",
        "reasoning_content",
        "reasoning",
        "reasoning_details",
        "thinking",
    ):
        val = getattr(delta, key, None)
        if val is None and key in extra:
            val = extra[key]
        if val is not None:
            out[key] = val
    if extra:
        for k, v in extra.items():
            if k not in out and v is not None:
                out[k] = v
    return out


def _extract_reasoning_text(d: Dict[str, Any]) -> str:
    """
    Pull reasoning string from OpenRouter / DeepSeek / Ollama / LM Studio fields.

    Providers often mirror the same delta in several keys (e.g. OpenRouter
    `reasoning` + `reasoning_details`). Take the first non-empty source only —
    concatenating them doubles every chunk in the UI.
    """

    def _from_val(val: Any) -> str:
        if isinstance(val, str) and val:
            return val
        if isinstance(val, dict):
            text = val.get("text") or val.get("content") or ""
            return str(text) if text else ""
        return ""

    for key in ("reasoning_content", "reasoning", "thinking"):
        text = _from_val(d.get(key))
        if text:
            return text

    details = d.get("reasoning_details")
    if isinstance(details, list):
        parts: List[str] = []
        for item in details:
            if isinstance(item, str) and item:
                parts.append(item)
            elif isinstance(item, dict):
                text = (
                    item.get("text")
                    or item.get("content")
                    or item.get("summary")
                    or ""
                )
                if text:
                    parts.append(str(text))
        return "".join(parts)
    if isinstance(details, str) and details:
        return details

    return ""


def _tool_calls_as_dicts(raw: Any) -> List[Dict[str, Any]]:
    if not raw:
        return []
    out: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(item)
            continue
        dump = getattr(item, "model_dump", None)
        if callable(dump):
            try:
                out.append(dump(exclude_none=False) or {})
            except TypeError:
                out.append(dump() or {})
        else:
            fn = getattr(item, "function", None)
            out.append(
                {
                    "index": getattr(item, "index", None),
                    "id": getattr(item, "id", None),
                    "type": getattr(item, "type", "function"),
                    "function": {
                        "name": getattr(fn, "name", None) if fn else None,
                        "arguments": getattr(fn, "arguments", None) if fn else None,
                    },
                }
            )
    return out


class ContentThinkSplitter:
    """
    Stateful splitter: when providers embed thinking inside content
    (Gemma tags / bare think_token), route those pieces to thinking.
    """

    def __init__(self, profile: OpenAIProfile) -> None:
        self._in_think = False
        self._buf = ""
        token = (profile.think_token or "").strip() if profile.think else ""
        self._bare_token = token

    def feed(self, text: str) -> tuple[str, str]:
        if not text:
            return "", ""
        self._buf += text
        thinking_parts: List[str] = []
        content_parts: List[str] = []

        while self._buf:
            if self._in_think:
                m = _CLOSE_THINK_RE.search(self._buf)
                if m:
                    thinking_parts.append(self._buf[: m.start()])
                    self._buf = self._buf[m.end() :]
                    self._in_think = False
                    continue
                # Keep a short tail in case close tag arrives split across chunks.
                if len(self._buf) > 32:
                    thinking_parts.append(self._buf[:-32])
                    self._buf = self._buf[-32:]
                break

            # Bare think_token is a reopen marker (Gemma); reasoning usually
            # arrives via reasoning_content — strip the token, do not swallow content.
            if self._bare_token and self._buf.startswith(self._bare_token):
                self._buf = self._buf[len(self._bare_token) :]
                continue

            m = _OPEN_THINK_RE.search(self._buf)
            if m and m.start() == 0:
                # Consume open tag; enter think mode.
                # For <|channel>thought … no fixed open length — find end of open.
                if self._buf.startswith("<think>"):
                    self._buf = self._buf[len("<think>") :]
                elif self._buf.startswith("<|think|>"):
                    self._buf = self._buf[len("<|think|>") :]
                else:
                    # <|channel>thought — leave until we see more; treat rest as think
                    # after the matched prefix length.
                    self._buf = self._buf[m.end() :]
                self._in_think = True
                continue

            if m:
                content_parts.append(self._buf[: m.start()])
                self._buf = self._buf[m.start() :]
                continue

            # No open tag in buffer. Hold back a prefix that might be a partial tag/token.
            hold = self._partial_hold_len(self._buf)
            if hold and hold == len(self._buf):
                break
            emit = self._buf if hold == 0 else self._buf[:-hold]
            self._buf = "" if hold == 0 else self._buf[-hold:]
            if emit:
                content_parts.append(emit)
            break

        return "".join(thinking_parts), "".join(content_parts)

    def flush(self) -> tuple[str, str]:
        rest = self._buf
        self._buf = ""
        if not rest:
            return "", ""
        if self._in_think:
            self._in_think = False
            return rest, ""
        return "", rest

    def _partial_hold_len(self, buf: str) -> int:
        candidates = ["<think>", "</think>", "<|think|>", "<|channel>", "<channel|>"]
        if self._bare_token:
            candidates.append(self._bare_token)
        hold = 0
        for tag in candidates:
            for i in range(1, len(tag)):
                if buf.endswith(tag[:i]):
                    hold = max(hold, i)
        return hold


def normalize_chunk(delta: Any, splitter: ContentThinkSplitter) -> NormalizedDelta:
    """
    Map a single chat.completion stream delta into thinking / content / tool deltas.
    """
    d = _delta_as_dict(delta)
    reasoning = _extract_reasoning_text(d)
    raw_content = d.get("content") or ""
    if not isinstance(raw_content, str):
        raw_content = str(raw_content) if raw_content else ""

    tag_think, clean_content = splitter.feed(raw_content)
    thinking = reasoning + tag_think

    return NormalizedDelta(
        thinking=thinking,
        content=clean_content,
        tool_call_deltas=_tool_calls_as_dicts(d.get("tool_calls")),
    )


@dataclass
class AssembledToolCall:
    id: str
    name: str
    arguments: str
    index: int


class ToolCallAssembler:
    """Accumulate streamed tool_call deltas by index until the turn ends."""

    def __init__(self) -> None:
        self._by_index: Dict[int, Dict[str, Any]] = {}

    def reset(self) -> None:
        self._by_index.clear()

    def push(self, deltas: List[Dict[str, Any]]) -> None:
        for d in deltas:
            idx = d.get("index")
            if idx is None:
                idx = len(self._by_index)
            idx = int(idx)
            slot = self._by_index.setdefault(
                idx,
                {"id": None, "type": "function", "function": {"name": None, "arguments": ""}},
            )
            if d.get("id"):
                slot["id"] = d["id"]
            if d.get("type"):
                slot["type"] = d["type"]
            fn = d.get("function") or {}
            if not isinstance(fn, dict):
                name = getattr(fn, "name", None)
                arguments = getattr(fn, "arguments", None)
                fn = {"name": name, "arguments": arguments}
            slot_fn = slot["function"]
            if fn.get("name"):
                slot_fn["name"] = fn["name"]
            args = fn.get("arguments")
            if args:
                slot_fn["arguments"] = (slot_fn.get("arguments") or "") + str(args)

    def finish(self) -> List[AssembledToolCall]:
        out: List[AssembledToolCall] = []
        for idx in sorted(self._by_index.keys()):
            slot = self._by_index[idx]
            fn = slot.get("function") or {}
            name = fn.get("name") or ""
            args = fn.get("arguments") or ""
            tc_id = slot.get("id") or f"call_{idx}"
            if not name and not args:
                continue
            out.append(
                AssembledToolCall(
                    id=str(tc_id),
                    name=str(name),
                    arguments=str(args),
                    index=idx,
                )
            )
        self.reset()
        return out


def assembled_to_openai_tool_calls(
    calls: List[AssembledToolCall],
) -> List[Dict[str, Any]]:
    return [
        {
            "id": c.id,
            "type": "function",
            "function": {"name": c.name, "arguments": c.arguments},
        }
        for c in calls
    ]


def preview_tool_result(text: str, limit: int = 500) -> str:
    s = str(text)
    if len(s) <= limit:
        return s
    return s[: limit - 1] + "…"


def build_assistant_replay(
    *,
    content: Optional[str],
    tool_calls: List[AssembledToolCall],
    reasoning_parts: List[str],
    profile: OpenAIProfile,
) -> Dict[str, Any]:
    """
    Build an assistant message dict for the next request after a streamed turn.
    Mirrors _assistant_message_dict behaviour for reasoning + think_token.
    """
    msg: Dict[str, Any] = {"role": "assistant"}
    text = content if content is not None else ""
    reasoning = "".join(reasoning_parts)
    if reasoning:
        msg["reasoning_content"] = reasoning
        msg["reasoning"] = reasoning

    if tool_calls:
        msg["tool_calls"] = assembled_to_openai_tool_calls(tool_calls)
        # Content may be empty on tool-only turns.
        if text:
            msg["content"] = text
        else:
            token = (profile.think_token or "").strip()
            if profile.think and token:
                msg["content"] = token
            else:
                msg["content"] = None
    else:
        msg["content"] = text

    return msg

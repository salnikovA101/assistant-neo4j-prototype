"""Per-turn state: UI search depth and per-tool budgets of one answer.

`max_turns` in the LLM profile bounds tool-calling *turns*; a single turn may
still emit several parallel tool calls. This module makes the search budget of
one user turn explicit and enforceable, and carries the UI search depth so the
model never has to choose it.
"""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

logger = logging.getLogger(__name__)

SEARCH_DEPTHS: tuple[str, ...] = ("low", "medium", "high")
DEFAULT_SEARCH_DEPTH = "medium"
DEFAULT_MAX_ASK = 2
DEFAULT_MAX_SEARCHES = DEFAULT_MAX_ASK
DEFAULT_MAX_QUERY = 8

ASK_SUBGRAPH_TOOL = "ask_subgraph"
QUERY_GRAPH_TOOL = "query_graph"
QUOTA_TOOLS: frozenset[str] = frozenset({ASK_SUBGRAPH_TOOL, QUERY_GRAPH_TOOL})
BOTH_EXHAUSTED_PHRASE = "Tools are exhausted, answer now."

_WS_RE = re.compile(r"\s+")


def parse_search_depth(value: object) -> str | None:
    """Accept UI values low | medium | high; anything else → None."""
    if not isinstance(value, str):
        return None
    depth = value.strip().lower()
    return depth if depth in SEARCH_DEPTHS else None


def subquestion_key(text: str) -> str:
    """Dedupe key: case- and whitespace-insensitive, trailing punctuation off."""
    return _WS_RE.sub(" ", str(text).strip().lower()).strip(" .?!")


@dataclass
class TurnState:
    search_depth: str = DEFAULT_SEARCH_DEPTH
    max_searches: int = DEFAULT_MAX_SEARCHES
    searches_used: int = 0
    max_query: int = DEFAULT_MAX_QUERY
    query_used: int = 0
    seen_subquestions: set[str] = field(default_factory=set)
    user_id: str = ""
    run_id: str = ""
    conversation_id: str = ""
    branch_id: str = ""
    checkpoint_id: str = ""
    mode: str = "auto"
    store: Any = None
    retrieval_state: dict[str, Any] = field(default_factory=dict)
    approved_subquestions: list[str] = field(default_factory=list)

    def searches_left(self) -> int:
        return max(0, self.max_searches - self.searches_used)

    def query_left(self) -> int:
        return max(0, self.max_query - self.query_used)


_current_turn: ContextVar[TurnState | None] = ContextVar(
    "current_turn_state",
    default=None,
)


def current_turn() -> TurnState | None:
    return _current_turn.get()


def search_depth() -> str:
    turn = _current_turn.get()
    return turn.search_depth if turn else DEFAULT_SEARCH_DEPTH


def searches_state() -> tuple[int, int]:
    """(used, max) ask_subgraph slots for this turn; (0, default) outside a bound turn."""
    return tool_quota_state(ASK_SUBGRAPH_TOOL)


def tool_quota_state(name: str) -> tuple[int, int]:
    """(used, max) for a quota tool; defaults outside a bound turn."""
    turn = _current_turn.get()
    if turn is None:
        if name == QUERY_GRAPH_TOOL:
            return 0, DEFAULT_MAX_QUERY
        return 0, DEFAULT_MAX_SEARCHES
    pair = _quota_pair(turn, name)
    if pair is None:
        return 0, 0
    return pair


def _quota_pair(turn: TurnState, name: str) -> tuple[int, int] | None:
    if name == ASK_SUBGRAPH_TOOL:
        return turn.searches_used, turn.max_searches
    if name == QUERY_GRAPH_TOOL:
        return turn.query_used, turn.max_query
    return None


def has_tool_slot(name: str) -> bool:
    """True if this quota tool can still run. Unbound turns are not quota-limited."""
    turn = _current_turn.get()
    if turn is None:
        return True
    pair = _quota_pair(turn, name)
    if pair is None:
        return True
    used, limit = pair
    return used < limit


def take_tool_slot(name: str) -> bool:
    """Spend one successful-call slot. False → exhausted or unbound, do not count."""
    turn = _current_turn.get()
    if turn is None:
        return False
    if name == ASK_SUBGRAPH_TOOL:
        if turn.searches_used >= turn.max_searches:
            return False
        turn.searches_used += 1
        return True
    if name == QUERY_GRAPH_TOOL:
        if turn.query_used >= turn.max_query:
            return False
        turn.query_used += 1
        return True
    return False


def take_search_slot() -> bool:
    """Spend one ask_subgraph slot. False → budget exhausted, do not search."""
    return take_tool_slot(ASK_SUBGRAPH_TOOL)


def quotas_exhausted() -> bool:
    """True when both ask_subgraph and query_graph slots are spent."""
    turn = _current_turn.get()
    if turn is None:
        return False
    return (
        turn.searches_used >= turn.max_searches
        and turn.query_used >= turn.max_query
    )


def is_quota_error_result(text: str) -> bool:
    stripped = (text or "").lstrip()
    return stripped.startswith("TOOL_ERROR") or stripped.startswith("QUERY_ERROR")


def tool_quota_footer(name: str) -> str:
    """n/X notice for the called tool plus the other search tool's remaining."""
    ask_used, ask_max = tool_quota_state(ASK_SUBGRAPH_TOOL)
    query_used, query_max = tool_quota_state(QUERY_GRAPH_TOOL)
    both_done = ask_used >= ask_max and query_used >= query_max
    parts = [
        _quota_part(ASK_SUBGRAPH_TOOL, ask_used, ask_max, name, both_done),
        _quota_part(QUERY_GRAPH_TOOL, query_used, query_max, name, both_done),
    ]
    text = " ".join(parts)
    if both_done:
        text = f"{text} {BOTH_EXHAUSTED_PHRASE}"
    return f"\n\n[{text}]"


def _quota_part(
    tool: str,
    used: int,
    limit: int,
    called: str,
    both_done: bool,
) -> str:
    if used >= limit:
        if both_done:
            return f"{tool} {used}/{limit} exhausted."
        if tool == called:
            return f"{tool} {used}/{limit} exhausted for this tool."
        return f"{tool} {used}/{limit} exhausted."
    if tool == called:
        return f"{tool} {used}/{limit} used. You may call it again."
    return f"{tool} {used}/{limit} remaining."


def with_tool_quota_footer(text: str, name: str) -> str:
    footer = tool_quota_footer(name)
    body = (text or "").rstrip()
    if footer.strip() in body:
        return body
    return f"{body}{footer}"


def finalize_tool_result(name: str, result: str, *, success: bool) -> str:
    """Spend a slot only after success, then append n/X (and both-exhausted if needed)."""
    if success:
        take_tool_slot(name)
    return with_tool_quota_footer(result, name)


def tool_limit_message(name: str) -> str:
    """Refusal when this tool's successful-call quota is already spent."""
    used, limit = tool_quota_state(name)
    other = (
        QUERY_GRAPH_TOOL if name == ASK_SUBGRAPH_TOOL else ASK_SUBGRAPH_TOOL
    )
    other_used, other_limit = tool_quota_state(other)
    head = f"TOOL_ERROR: {name} limit for this answer is spent ({used}/{limit})."
    if quotas_exhausted():
        body = f"{head} {BOTH_EXHAUSTED_PHRASE}"
    else:
        body = f"{head} Use {other} ({other_used}/{other_limit} remaining)."
    return with_tool_quota_footer(body, name)


def remember_subquestions(texts: Iterable[str]) -> None:
    turn = _current_turn.get()
    if turn is None:
        return
    for text in texts:
        key = subquestion_key(text)
        if key:
            turn.seen_subquestions.add(key)


def seen_subquestions() -> set[str]:
    turn = _current_turn.get()
    return set(turn.seen_subquestions) if turn else set()


@contextmanager
def bind_turn(
    depth: str | None = None,
    max_searches: int = DEFAULT_MAX_SEARCHES,
    max_query: int = DEFAULT_MAX_QUERY,
    *,
    context: dict[str, Any] | None = None,
) -> Iterator[TurnState]:
    """Bind a fresh TurnState for one user turn."""
    ctx = context or {}
    state = TurnState(
        search_depth=parse_search_depth(depth) or DEFAULT_SEARCH_DEPTH,
        max_searches=max(1, int(max_searches or DEFAULT_MAX_SEARCHES)),
        searches_used=max(0, int(ctx.get("searches_used") or 0)),
        max_query=max(1, int(max_query or DEFAULT_MAX_QUERY)),
        query_used=max(0, int(ctx.get("query_used") or 0)),
        user_id=str(ctx.get("user_id") or ""),
        run_id=str(ctx.get("run_id") or "").strip(),
        conversation_id=str(ctx.get("conversation_id") or ""),
        branch_id=str(ctx.get("branch_id") or ""),
        checkpoint_id=str(ctx.get("checkpoint_id") or ""),
        mode="staged" if str(ctx.get("mode") or "auto") == "staged" else "auto",
        store=ctx.get("store"),
        retrieval_state=dict(ctx.get("retrieval_state") or {}),
        approved_subquestions=[str(v) for v in (ctx.get("approved_subquestions") or [])],
    )
    token = _current_turn.set(state)
    try:
        yield state
    finally:
        # A streamed turn can be closed from another context (client disconnect),
        # where the token no longer applies.
        try:
            _current_turn.reset(token)
        except ValueError:
            _current_turn.set(None)

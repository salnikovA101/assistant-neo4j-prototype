"""Per-turn state: UI search depth and the ask_subgraph budget of one answer.

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
from typing import Iterable, Iterator

logger = logging.getLogger(__name__)

SEARCH_DEPTHS: tuple[str, ...] = ("low", "medium", "high")
DEFAULT_SEARCH_DEPTH = "medium"
DEFAULT_MAX_SEARCHES = 2

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
    seen_subquestions: set[str] = field(default_factory=set)

    def searches_left(self) -> int:
        return max(0, self.max_searches - self.searches_used)


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
    """(used, max) for this turn; (0, default) outside a bound turn."""
    turn = _current_turn.get()
    if turn is None:
        return 0, DEFAULT_MAX_SEARCHES
    return turn.searches_used, turn.max_searches


def take_search_slot() -> bool:
    """Spend one ask_subgraph slot. False → budget exhausted, do not search."""
    turn = _current_turn.get()
    if turn is None:
        return True
    if turn.searches_used >= turn.max_searches:
        return False
    turn.searches_used += 1
    return True


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
) -> Iterator[TurnState]:
    """Bind a fresh TurnState for one user turn."""
    state = TurnState(
        search_depth=parse_search_depth(depth) or DEFAULT_SEARCH_DEPTH,
        max_searches=max(1, int(max_searches or DEFAULT_MAX_SEARCHES)),
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

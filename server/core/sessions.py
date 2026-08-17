"""Per-tab conversation state: history + source ids, isolated by X-Session-Id."""

from __future__ import annotations

import asyncio
import re
import time
import uuid
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import AsyncIterator

from server.llm.history_manager import HistoryManager
from server.tools.source_registry import SourceRegistry

SESSION_HEADER = "X-Session-Id"
# UUID and similar client ids (sessionStorage). Hyphen is allowed.
SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


@dataclass
class ConversationSession:
    session_id: str
    history: HistoryManager
    sources: SourceRegistry
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    last_used: float = field(default_factory=time.monotonic)


_current_session: ContextVar[ConversationSession | None] = ContextVar(
    "current_conversation_session",
    default=None,
)


def current_session() -> ConversationSession | None:
    return _current_session.get()


def current_sources() -> SourceRegistry | None:
    sess = _current_session.get()
    return sess.sources if sess else None


def resolve_session_id(raw: str | None) -> str:
    """Return a usable session id. Missing → ephemeral UUID; invalid → ValueError."""
    s = (raw or "").strip()
    if not s:
        return str(uuid.uuid4())
    if not SESSION_ID_RE.fullmatch(s):
        raise ValueError("Invalid X-Session-Id")
    return s


class SessionStore:
    """In-memory map session_id → {history, sources}. Same TTL/cap idea as GraphRunStore."""

    def __init__(self, ttl_seconds: int = 1800, max_entries: int = 100) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._sessions: dict[str, ConversationSession] = {}

    def _drop_expired(self) -> None:
        now = time.monotonic()
        expired = [
            sid
            for sid, sess in self._sessions.items()
            if now - sess.last_used > self.ttl_seconds
        ]
        for sid in expired:
            self._sessions.pop(sid, None)

    def get_or_create(self, session_id: str, history_len: int) -> ConversationSession:
        self._drop_expired()
        sess = self._sessions.get(session_id)
        if sess is None:
            while len(self._sessions) >= self.max_entries:
                oldest = min(
                    self._sessions.items(), key=lambda item: item[1].last_used
                )[0]
                self._sessions.pop(oldest, None)
            sess = ConversationSession(
                session_id=session_id,
                history=HistoryManager(history_len),
                sources=SourceRegistry(),
            )
            self._sessions[session_id] = sess
        sess.last_used = time.monotonic()
        return sess

    def clear(self, session_id: str) -> bool:
        """Empty history + source ids for this tab; keep the slot if it exists."""
        sess = self._sessions.get(session_id)
        if sess is None:
            return False
        sess.history.clear_history()
        sess.sources.clear()
        sess.last_used = time.monotonic()
        return True


session_store = SessionStore()


def set_current_session(sess: ConversationSession) -> Token:
    return _current_session.set(sess)


def reset_current_session(token: Token) -> None:
    _current_session.reset(token)


@asynccontextmanager
async def bind_conversation(
    session_id: str | None,
    history_len: int,
) -> AsyncIterator[ConversationSession]:
    """Bind this task to a session and serialize turns on that session."""
    sid = resolve_session_id(session_id)
    sess = session_store.get_or_create(sid, history_len)
    async with sess.lock:
        token = set_current_session(sess)
        try:
            yield sess
        finally:
            reset_current_session(token)

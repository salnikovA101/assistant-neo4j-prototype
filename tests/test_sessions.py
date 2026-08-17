"""Per-tab conversation sessions (history + source registry)."""

from __future__ import annotations

import time
import uuid

import pytest

from server.core.sessions import (
    SessionStore,
    bind_conversation,
    current_sources,
    resolve_session_id,
    session_store,
)
from server.tools.source_registry import SourceRegistry


def test_resolve_session_id_missing_is_ephemeral():
    a = resolve_session_id(None)
    b = resolve_session_id("  ")
    uuid.UUID(a)
    uuid.UUID(b)
    assert a != b


def test_resolve_session_id_accepts_uuid():
    sid = str(uuid.uuid4())
    assert resolve_session_id(sid) == sid


def test_resolve_session_id_rejects_garbage():
    with pytest.raises(ValueError):
        resolve_session_id("no")
    with pytest.raises(ValueError):
        resolve_session_id("spaces not ok!!")


def test_store_isolates_history_and_sources():
    store = SessionStore(ttl_seconds=3600, max_entries=10)
    a = store.get_or_create("isol-aaa", 6)
    b = store.get_or_create("isol-bbb", 6)
    a.history.add_entry("q1", "a1")
    a.sources.register("a.pdf")
    b.sources.register("b.pdf")
    assert a.history.get_history()[0]["content"] == "q1"
    assert b.history.get_history() == []
    assert a.sources.resolve(1) == "a.pdf"
    assert b.sources.resolve(1) == "b.pdf"


def test_store_clear_empties_but_keeps_slot():
    store = SessionStore(ttl_seconds=3600, max_entries=10)
    sess = store.get_or_create("clr-aaaa", 6)
    sess.history.add_entry("q", "a")
    sess.sources.register("x.pdf")
    assert store.clear("clr-aaaa") is True
    assert sess.history.get_history() == []
    assert sess.sources.resolve(1) is None
    assert store.get_or_create("clr-aaaa", 6) is sess


def test_store_evicts_oldest_at_cap():
    store = SessionStore(ttl_seconds=3600, max_entries=2)
    store.get_or_create("cap-aaaa", 6)
    store.get_or_create("cap-bbbb", 6)
    store.get_or_create("cap-cccc", 6)
    assert "cap-aaaa" not in store._sessions
    assert "cap-bbbb" in store._sessions
    assert "cap-cccc" in store._sessions


def test_store_ttl_expires():
    store = SessionStore(ttl_seconds=0.01, max_entries=10)
    store.get_or_create("ttl-aaaa", 6)
    time.sleep(0.03)
    store.get_or_create("ttl-bbbb", 6)
    assert "ttl-aaaa" not in store._sessions


@pytest.mark.asyncio
async def test_bind_conversation_keeps_tabs_apart():
    async with bind_conversation("bind-aaa", 6) as a:
        a.history.add_entry("from-a", "ans-a")
        a.sources.register("a.pdf")
    async with bind_conversation("bind-bbb", 6) as b:
        assert b.history.get_history() == []
        b.history.add_entry("from-b", "ans-b")
    async with bind_conversation("bind-aaa", 6) as a:
        hist = a.history.get_history()
        assert hist[0]["content"] == "from-a"
        assert all(m.get("content") != "from-b" for m in hist)
        assert a.sources.resolve(1) == "a.pdf"


@pytest.mark.asyncio
async def test_current_sources_follows_bound_session():
    async with bind_conversation("src-aaaa", 6) as sess:
        assert current_sources() is sess.sources
        sess.sources.register("paper.pdf")
        assert current_sources() is not None
        assert current_sources().resolve(1) == "paper.pdf"
    assert current_sources() is None

    leftover = session_store.get_or_create("src-aaaa", 6)
    assert leftover.sources.resolve(1) == "paper.pdf"
    assert isinstance(leftover.sources, SourceRegistry)

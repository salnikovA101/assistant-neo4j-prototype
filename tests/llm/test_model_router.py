from __future__ import annotations

import pytest

from server.llm.model_router import (
    AUTH,
    KEY_DEAD,
    MemoryBanStore,
    QUOTA_EXHAUSTED,
    RATE_LIMIT,
    candidate_profiles,
    classify_llm_error,
    llm_key_fingerprint,
)
from server.llm.stream_events import StreamEvent
from server.utils.config import load_config
from server.tools.source_registry import SourceRegistry


class _Err(Exception):
    def __init__(self, status: int, text: str, code: str | None = None) -> None:
        super().__init__(text)
        self.status_code = status
        self.body = text
        self.code = code


def test_fingerprint_is_stable_and_secret_free():
    a = llm_key_fingerprint("sk-user-one")
    b = llm_key_fingerprint("sk-user-two")
    assert a != b
    assert a == llm_key_fingerprint("sk-user-one")
    assert "sk-user" not in a
    assert len(a) == 64
    assert llm_key_fingerprint("") == "empty"
    assert llm_key_fingerprint(None) == "empty"


def test_classify_free_tier_quota_not_rate_limit():
    quota = _Err(403, "AllocationQuota.FreeTierOnly Free allocated quota exceeded")
    assert classify_llm_error(quota) == QUOTA_EXHAUSTED
    rate = _Err(429, "AllocationQuota.RateQuota exceeded")
    assert classify_llm_error(rate) == RATE_LIMIT
    auth = _Err(401, "InvalidApiKey")
    assert classify_llm_error(auth) == AUTH
    overdue = _Err(403, "PrepaidBillOverdue")
    assert classify_llm_error(overdue) == KEY_DEAD


@pytest.mark.asyncio
async def test_auto_skips_banned_models_for_key_fingerprint():
    llm = load_config().llm
    store = MemoryBanStore()
    fp = llm_key_fingerprint("sk-byok")
    first, second = llm.auto_order[0], llm.auto_order[1]
    await store.ban_llm_model(fp, first, QUOTA_EXHAUSTED)
    live = await candidate_profiles(llm, "auto", key_fp=fp, store=store, rotate=True)
    assert live[0] == second
    assert first not in live
    other = await candidate_profiles(
        llm, "auto", key_fp=llm_key_fingerprint("sk-other"), store=store, rotate=True
    )
    assert other[0] == first


@pytest.mark.asyncio
async def test_manual_banned_model_has_no_candidates():
    llm = load_config().llm
    store = MemoryBanStore()
    fp = llm_key_fingerprint("sk-byok")
    await store.ban_llm_model(fp, "qwen38_flash", QUOTA_EXHAUSTED)
    assert await candidate_profiles(
        llm, "qwen38_flash", key_fp=fp, store=store, rotate=False
    ) == []
    await store.mark_llm_key_dead(fp)
    assert await candidate_profiles(
        llm, "auto", key_fp=fp, store=store, rotate=True
    ) == ["ollama"]


class _QuotaProvider:
    def __init__(self, profile, code: str) -> None:
        self.profile = profile
        self.code = code
        self.calls = 0

    async def generate_response_stream(self, **_kwargs):
        self.calls += 1
        yield StreamEvent("error", {"code": self.code, "message": "quota"})


class _OkProvider:
    def __init__(self, profile) -> None:
        self.profile = profile
        self.calls = 0

    async def generate_response_stream(self, **_kwargs):
        self.calls += 1
        yield StreamEvent("content", {"delta": "ok"})
        yield StreamEvent("done", {"final_content": "ok", "history_tool_messages": []})


class _SqStatusProvider:
    def __init__(self, profile) -> None:
        self.profile = profile

    async def generate_response_stream(self, **_kwargs):
        final = (
            "Ответ (source:1).\n\n<SQ_STATUS_JSON>\n"
            '{"version":1,"items":[{"ref":"subquestion:1","status":"closed",'
            '"reason":"Данные найдены","source_refs":["source:1"]}]}\n'
            "</SQ_STATUS_JSON>"
        )
        yield StreamEvent("content", {"delta": final[:35]})
        yield StreamEvent("content", {"delta": final[35:]})
        yield StreamEvent("done", {"final_content": final, "history_tool_messages": []})


@pytest.mark.asyncio
async def test_staged_manager_hides_and_extracts_sq_status_block(monkeypatch):
    from server.llm.manager import LLMManager

    cfg = load_config()
    mgr = LLMManager(cfg)
    provider = _SqStatusProvider(cfg.llm.profiles.qwen38_flash)
    sources = SourceRegistry()
    sources.register("paper.pdf")
    monkeypatch.setattr(mgr, "provider_for", lambda _name=None: provider)
    monkeypatch.setattr(mgr, "_context_fits", lambda **_kwargs: True)
    monkeypatch.setattr(mgr, "_active_sources", lambda: sources)
    events = [
        event
        async for event in mgr.generate_response_stream(
            "question",
            profile_name="qwen38_flash",
            turn_context={
                "mode": "staged",
                "store": MemoryBanStore(),
                "active_sq_refs": ["subquestion:1"],
            },
        )
    ]
    streamed = "".join(
        str(event.data.get("delta") or "") for event in events if event.type == "content"
    )
    assert "SQ_STATUS_JSON" not in streamed
    done = next(event for event in events if event.type == "done")
    assert "SQ_STATUS_JSON" not in done.data["final_content"]
    assert "### Состояние исследовательских вопросов" in done.data["final_content"]
    assert done.data["_sq_assessments"][0]["status"] == "closed"


@pytest.mark.asyncio
async def test_auto_rotates_once_quota_dies_then_stays(monkeypatch):
    from server.llm.manager import LLMManager

    cfg = load_config()
    mgr = LLMManager(cfg)
    first, second = cfg.llm.auto_order[0], cfg.llm.auto_order[1]
    dying = getattr(cfg.llm.profiles, first)
    nxt = getattr(cfg.llm.profiles, second)
    providers = {
        first: _QuotaProvider(dying, QUOTA_EXHAUSTED),
        second: _OkProvider(nxt),
    }

    def _provider_for(name: str | None = None):
        key = name or first
        return providers[key]

    monkeypatch.setattr(mgr, "provider_for", _provider_for)
    monkeypatch.setattr(mgr, "_context_fits", lambda **_kwargs: True)
    store = MemoryBanStore()
    events = [
        event
        async for event in mgr.generate_response_stream(
            "hi",
            profile_name="auto",
            turn_context={"mode": "auto", "store": store},
        )
    ]
    assert providers[first].calls == 1
    assert providers[second].calls == 1
    assert any(event.type == "model" and event.data["id"] == second for event in events)
    assert events[-1].type == "done"
    assert events[-1].data["modelId"] == second
    banned = await store.banned_llm_models(llm_key_fingerprint(cfg.llm.profiles.qwen_cloud.api_key))
    assert first in banned

    events2 = [
        event
        async for event in mgr.generate_response_stream(
            "again",
            profile_name="auto",
            turn_context={"mode": "auto", "store": store},
        )
    ]
    assert providers[first].calls == 1
    assert any(event.type == "model" and event.data["id"] == second for event in events2)


@pytest.mark.asyncio
async def test_manual_model_does_not_rotate():
    from server.llm.manager import LLMManager

    cfg = load_config()
    mgr = LLMManager(cfg)
    flash = cfg.llm.profiles.qwen38_flash
    nxt = cfg.llm.profiles.qwen38_max
    providers = {
        "qwen38_flash": _QuotaProvider(flash, QUOTA_EXHAUSTED),
        "qwen38_max": _OkProvider(nxt),
    }
    mgr.provider_for = lambda name=None: providers[name or "qwen38_flash"]  # type: ignore[method-assign]
    mgr._context_fits = lambda **_kwargs: True  # type: ignore[method-assign]
    store = MemoryBanStore()
    events = [
        event
        async for event in mgr.generate_response_stream(
            "hi",
            profile_name="qwen38_flash",
            turn_context={"mode": "auto", "store": store},
        )
    ]
    assert providers["qwen38_max"].calls == 0
    assert events[-1].type == "error"
    assert events[-1].data["code"] == QUOTA_EXHAUSTED


@pytest.mark.asyncio
async def test_rate_limit_does_not_ban_or_rotate():
    from server.llm.manager import LLMManager

    cfg = load_config()
    mgr = LLMManager(cfg)
    first, second = cfg.llm.auto_order[0], cfg.llm.auto_order[1]
    dying = getattr(cfg.llm.profiles, first)
    nxt = getattr(cfg.llm.profiles, second)
    providers = {
        first: _QuotaProvider(dying, RATE_LIMIT),
        second: _OkProvider(nxt),
    }
    mgr.provider_for = lambda name=None: providers[name or first]  # type: ignore[method-assign]
    mgr._context_fits = lambda **_kwargs: True  # type: ignore[method-assign]
    store = MemoryBanStore()
    events = [
        event
        async for event in mgr.generate_response_stream(
            "hi",
            profile_name="auto",
            turn_context={"mode": "auto", "store": store},
        )
    ]
    assert providers[second].calls == 0
    assert events[-1].type == "error"
    assert events[-1].data["code"] == RATE_LIMIT
    fp = llm_key_fingerprint((cfg.llm.profiles.qwen_cloud.api_key or "").strip())
    assert await store.banned_llm_models(fp) == set()


@pytest.mark.asyncio
async def test_app_store_bans_are_keyed_by_fingerprint(tmp_path):
    from server.core.app_store import AppStore

    store = AppStore(str(tmp_path / "bans.db"))
    await store.open()
    try:
        fp = llm_key_fingerprint("sk-same")
        await store.ban_llm_model(fp, "qwen38_flash", QUOTA_EXHAUSTED)
        assert "qwen38_flash" in await store.banned_llm_models(fp)
        assert await store.banned_llm_models(llm_key_fingerprint("sk-other")) == set()
        await store.mark_llm_key_dead(fp)
        assert await store.llm_key_is_dead(fp) is True
        assert await store.llm_key_is_dead(llm_key_fingerprint("sk-other")) is False
    finally:
        await store.close()

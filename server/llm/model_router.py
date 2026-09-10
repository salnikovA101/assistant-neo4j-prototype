"""Per-key DashScope model rotation. Bans are keyed by sha256 of the API key."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from server.utils.config import AUTO_PROFILE, LlmConfig, llm_profile

QUOTA_EXHAUSTED = "quota_exhausted"
RATE_LIMIT = "rate_limit"
AUTH = "auth"
KEY_DEAD = "key_dead"
OTHER = "other"

_OUTPUT_EVENT_TYPES = frozenset({"thinking", "content", "tool_call", "card_draft"})


class BanStore(Protocol):
    async def banned_llm_models(self, key_fp: str) -> set[str]: ...

    async def ban_llm_model(self, key_fp: str, profile_id: str, reason: str) -> None: ...

    async def mark_llm_key_dead(self, key_fp: str) -> None: ...

    async def llm_key_is_dead(self, key_fp: str) -> bool: ...


class MemoryBanStore:
    """Process-local bans when the request has no AppStore."""

    def __init__(self) -> None:
        self._bans: dict[str, set[str]] = {}
        self._dead: set[str] = set()

    async def banned_llm_models(self, key_fp: str) -> set[str]:
        return set(self._bans.get(key_fp, set()))

    async def ban_llm_model(self, key_fp: str, profile_id: str, reason: str) -> None:
        self._bans.setdefault(key_fp, set()).add(profile_id)

    async def mark_llm_key_dead(self, key_fp: str) -> None:
        self._dead.add(key_fp)

    async def llm_key_is_dead(self, key_fp: str) -> bool:
        return key_fp in self._dead


def llm_key_fingerprint(api_key: str | None) -> str:
    raw = (api_key or "").strip()
    if not raw:
        return "empty"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _error_blob(exc: BaseException) -> str:
    parts = [str(exc), str(getattr(exc, "message", "") or "")]
    body = getattr(exc, "body", None)
    if body is not None:
        try:
            parts.append(json.dumps(body, ensure_ascii=False) if not isinstance(body, str) else body)
        except (TypeError, ValueError):
            parts.append(str(body))
    code = getattr(exc, "code", None)
    if code:
        parts.append(str(code))
    return " ".join(parts).lower()


def classify_llm_error(exc: BaseException) -> str:
    """Map provider failures to router actions. Never returns the raw secret."""
    status = getattr(exc, "status_code", None)
    text = _error_blob(exc)
    if "freetieronly" in text or "free allocated quota exceeded" in text:
        return QUOTA_EXHAUSTED
    if "prepaidbilloverdue" in text or "postpaidbilloverdue" in text:
        return KEY_DEAD
    if "commoditynotpurchased" in text:
        return KEY_DEAD
    if status in (401,):
        return AUTH
    if status == 403:
        if "quota" in text or "allocationquota" in text:
            return QUOTA_EXHAUSTED
        return AUTH
    if status == 429:
        if "free allocated quota exceeded" in text or "freetier" in text:
            return QUOTA_EXHAUSTED
        return RATE_LIMIT
    return OTHER


def classify_error_event(data: dict[str, Any] | None) -> str:
    payload = data or {}
    code = str(payload.get("code") or "").strip()
    if code in {QUOTA_EXHAUSTED, RATE_LIMIT, AUTH, KEY_DEAD}:
        return code
    message = str(payload.get("message") or "")
    if not message:
        return OTHER

    class _Blob(Exception):
        def __init__(self, text: str) -> None:
            super().__init__(text)
            self.body = text

    return classify_llm_error(_Blob(message))


def is_cloud_profile(llm: LlmConfig, name: str) -> bool:
    key = (name or "").strip()
    if key == AUTO_PROFILE:
        return True
    if key in {(item or "").strip() for item in (llm.auto_order or [])}:
        return True
    profile = llm_profile(llm, key)
    family = str(getattr(profile, "think_family", "") or "").strip().lower() if profile else ""
    return family in {"qwen38", "qwen37", "deepseek_v4", "glm", "kimi"}


def display_name_for(llm: LlmConfig, name: str) -> str:
    profile = llm_profile(llm, name)
    if profile is None:
        return name
    return (profile.display_name or "").strip() or name


def effective_cloud_key(llm: LlmConfig, request_key: str | None) -> str:
    override = (request_key or "").strip()
    if override:
        return override
    source = getattr(llm.profiles, "qwen_cloud", None)
    return ((source.api_key if source else "") or "").strip()


async def candidate_profiles(
    llm: LlmConfig,
    requested: str,
    *,
    key_fp: str,
    store: BanStore | None,
    rotate: bool,
) -> list[str]:
    """Ordered concrete profiles to try for this request."""
    fallback = (llm.fallback_profile or "ollama").strip() or "ollama"
    dead = bool(store and await store.llm_key_is_dead(key_fp))
    banned: set[str] = set()
    if store is not None:
        banned = await store.banned_llm_models(key_fp)

    if rotate or requested == AUTO_PROFILE:
        names = [
            name
            for name in (llm.auto_order or [])
            if llm_profile(llm, name) is not None
        ]
        if dead:
            return [fallback] if llm_profile(llm, fallback) is not None else []
        live = [name for name in names if name not in banned]
        if not live:
            return [fallback] if llm_profile(llm, fallback) is not None else []
        if fallback and fallback not in live and llm_profile(llm, fallback) is not None:
            live.append(fallback)
        return live

    if not requested or llm_profile(llm, requested) is None:
        return []
    if requested == fallback:
        return [requested]
    if dead or requested in banned:
        return []
    return [requested]


def event_has_model_output(event_type: str) -> bool:
    return event_type in _OUTPUT_EVENT_TYPES

"""Small SLM helpers for the judge stage."""

from __future__ import annotations

from typing import Any


def resolve_slm_base_url(raw: str | None) -> str:
    """Prefer reachable local LM Studio; rewrite Docker-only hostnames on host OS."""
    url = (raw or "").strip().rstrip("/")
    if not url:
        url = "http://127.0.0.1:1234/v1"
    if "host.docker.internal" in url:
        url = url.replace("host.docker.internal", "127.0.0.1")
    if url.endswith("/v1"):
        return url
    return f"{url}/v1"


def resolve_tool_llm_profile(config: Any) -> Any:
    """Resolve tool_profile via attribute access (LlmProfiles is not a dict)."""
    profile_name = config.llm.tool_profile
    llm_profile = getattr(config.llm.profiles, profile_name, None)
    return llm_profile or config.llm.profiles.other

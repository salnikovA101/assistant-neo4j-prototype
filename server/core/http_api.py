"""HTTP request models, health payload, CORS regex — no pipeline import."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from server.core.db import get_driver
from server.core.sessions import SESSION_HEADER, resolve_session_id

CORS_ORIGIN_RE = r"https?://(localhost|127\.0\.0\.1)(:\d+)?$"
LLM_API_KEY_HEADER = "X-LLM-Api-Key"
_LLM_API_KEY_MIN_LEN = 8
_LLM_API_KEY_MAX_LEN = 512


class TextProcessBody(BaseModel):
    text: str
    reasoning_effort: Optional[str] = None
    search_depth: Optional[str] = None


class GraphVizBody(BaseModel):
    graph_run_id: str = Field(default="")


def session_id_from_request(request: Request) -> str:
    try:
        return resolve_session_id(request.headers.get(SESSION_HEADER))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def parse_llm_api_key_header(raw: str | None) -> str | None:
    """Return a BYOK override, or None to use the server env key.

    Invalid values raise 400 without echoing the secret.
    """
    if raw is None:
        return None
    key = raw.strip()
    if not key:
        return None
    if any(ch.isspace() for ch in key):
        raise HTTPException(status_code=400, detail="Invalid X-LLM-Api-Key")
    if not (_LLM_API_KEY_MIN_LEN <= len(key) <= _LLM_API_KEY_MAX_LEN):
        raise HTTPException(status_code=400, detail="Invalid X-LLM-Api-Key")
    return key


def llm_api_key_from_request(request: Request) -> str | None:
    return parse_llm_api_key_header(request.headers.get(LLM_API_KEY_HEADER))


async def build_health(pipeline: Any | None) -> tuple[int, dict[str, Any]]:
    """Neo4j connectivity + LLM object present. No live LLM roundtrip (UI polls)."""
    checks = {"pipeline": False, "llm": False, "neo4j": False}
    if pipeline is not None:
        checks["pipeline"] = True
        model = getattr(getattr(pipeline, "llm", None), "model", None)
        checks["llm"] = model is not None
    try:
        driver = get_driver()
        await driver.verify_connectivity()
        checks["neo4j"] = True
    except Exception:
        checks["neo4j"] = False

    ok = all(checks.values())
    return (
        200 if ok else 503,
        {"status": "ready" if ok else "degraded", "checks": checks},
    )

"""HTTP request models, health payload, CORS regex — no pipeline import."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException, Request
from pydantic import BaseModel, Field

from server.core.db import get_driver
from server.core.sessions import SESSION_HEADER, resolve_session_id

CORS_ORIGIN_RE = r"https?://(localhost|127\.0\.0\.1)(:\d+)?$"


class TextProcessBody(BaseModel):
    text: str
    reasoning_effort: Optional[str] = None


class GraphVizBody(BaseModel):
    graph_run_id: str = Field(default="")


def session_id_from_request(request: Request) -> str:
    try:
        return resolve_session_id(request.headers.get(SESSION_HEADER))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


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

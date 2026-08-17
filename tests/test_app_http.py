"""HTTP-layer practice: CORS, health checks, request bodies."""

from __future__ import annotations

import re

import pytest
from pydantic import ValidationError

from server.core.http_api import (
    CORS_ORIGIN_RE,
    GraphVizBody,
    TextProcessBody,
    build_health,
)


def test_cors_regex_is_localhost_only():
    assert "*" not in CORS_ORIGIN_RE
    assert "localhost" in CORS_ORIGIN_RE
    assert r"127\.0\.0\.1" in CORS_ORIGIN_RE
    rx = re.compile(CORS_ORIGIN_RE)
    assert rx.fullmatch("http://localhost:8000")
    assert rx.fullmatch("http://127.0.0.1:8000")
    assert rx.fullmatch("https://localhost")
    assert rx.fullmatch("http://localhost:3000")
    assert rx.fullmatch("https://evil.example") is None


def test_text_process_body_requires_text():
    with pytest.raises(ValidationError):
        TextProcessBody()
    parsed = TextProcessBody(text="  hello  ", reasoning_effort="low")
    assert parsed.text == "  hello  "
    assert parsed.reasoning_effort == "low"


def test_graph_viz_body_defaults():
    assert GraphVizBody().graph_run_id == ""
    assert GraphVizBody(graph_run_id="gr_abc").graph_run_id == "gr_abc"


@pytest.mark.asyncio
async def test_health_ready_when_deps_ok(monkeypatch):
    class Driver:
        async def verify_connectivity(self):
            return None

    monkeypatch.setattr("server.core.http_api.get_driver", lambda: Driver())

    class Pipeline:
        llm = type("L", (), {"model": object()})()

    code, body = await build_health(Pipeline())
    assert code == 200
    assert body["status"] == "ready"
    assert body["checks"] == {"pipeline": True, "llm": True, "neo4j": True}


@pytest.mark.asyncio
async def test_health_degraded_without_neo4j(monkeypatch):
    def boom():
        raise RuntimeError("Neo4j driver is not initialized")

    monkeypatch.setattr("server.core.http_api.get_driver", boom)
    code, body = await build_health(None)
    assert code == 503
    assert body["status"] == "degraded"
    assert body["checks"]["pipeline"] is False
    assert body["checks"]["llm"] is False
    assert body["checks"]["neo4j"] is False

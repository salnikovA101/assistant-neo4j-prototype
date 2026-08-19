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
    parse_llm_api_key_header,
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


def test_parse_llm_api_key_header():
    from fastapi import HTTPException

    assert parse_llm_api_key_header(None) is None
    assert parse_llm_api_key_header("") is None
    assert parse_llm_api_key_header("   ") is None
    assert parse_llm_api_key_header("ollama_abcdefgh") == "ollama_abcdefgh"

    with pytest.raises(HTTPException) as short:
        parse_llm_api_key_header("short")
    assert short.value.status_code == 400
    assert "ollama" not in str(short.value.detail).lower()

    with pytest.raises(HTTPException) as spaced:
        parse_llm_api_key_header("bad key\nvalue")
    assert spaced.value.status_code == 400
    assert "bad key" not in str(spaced.value.detail)

    with pytest.raises(HTTPException):
        parse_llm_api_key_header("x" * 513)


def test_ui_config_does_not_leak_api_key():
    from server.core.app import build_ui_config
    from server.utils.config import AppConfig, OpenAIProfile

    secret = "super-secret-ollama-key"
    profile = OpenAIProfile(
        api_key=secret,
        think=True,
        think_effort="high",
        think_efforts=["high", "off"],
    )

    class Pipeline:
        llm = type("L", (), {"model": type("M", (), {"profile": profile})()})()
        config = AppConfig()
        config.llm.current_profile = "ollama"

    payload = build_ui_config(Pipeline())
    dumped = str(payload)
    assert "api_key" not in payload
    assert secret not in dumped
    assert payload["current_profile"] == "ollama"
    assert payload["llm_key_configured"] is True


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

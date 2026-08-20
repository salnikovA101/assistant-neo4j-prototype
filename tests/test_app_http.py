"""HTTP-layer practice: CORS, health checks, request bodies, UI Basic auth."""

from __future__ import annotations

import base64
import re
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import RedirectResponse

from server.core.http_api import (
    CORS_ORIGIN_RE,
    GraphVizBody,
    TextProcessBody,
    UI_SESSION_COOKIE,
    build_health,
    expected_ui_credentials,
    is_public_auth_path,
    make_session_token,
    parse_basic_authorization,
    parse_llm_api_key_header,
    request_is_ui_authenticated,
    session_token_ok,
    set_ui_session_cookie,
    ui_auth_middleware,
    ui_basic_configured,
    ui_basic_ok,
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


def _basic_header(user: str, password: str) -> dict[str, str]:
    token = base64.b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")
    return {"Authorization": f"Basic {token}"}


def test_parse_basic_authorization():
    assert parse_basic_authorization(None) is None
    assert parse_basic_authorization("") is None
    assert parse_basic_authorization("Bearer abc") is None
    assert parse_basic_authorization("Basic !!!not-base64!!!") is None
    assert parse_basic_authorization("Basic " + base64.b64encode(b"nocolon").decode()) is None

    user, password = parse_basic_authorization(_basic_header("demo", "s3cret")["Authorization"])
    assert user == "demo"
    assert password == "s3cret"

    user, password = parse_basic_authorization(
        _basic_header("demo", "a:b:c")["Authorization"]
    )
    assert user == "demo"
    assert password == "a:b:c"


def test_ui_basic_ok():
    assert ui_basic_configured("demo", "secret") is True
    assert ui_basic_configured("demo", "") is False
    assert ui_basic_configured("demo", "   ") is False

    assert ui_basic_ok("demo", "secret", "demo", "secret") is True
    assert ui_basic_ok("demo", "wrong", "demo", "secret") is False
    assert ui_basic_ok("other", "secret", "demo", "secret") is False
    assert ui_basic_ok("demo", "secret", "demo", "") is False
    assert ui_basic_ok("", "", "demo", "secret") is False


@contextmanager
def _stub_client(user: str = "demo", password: str = "secret"):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ui_basic_user = user
        app.state.ui_basic_password = password
        yield

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(BaseHTTPMiddleware, dispatch=ui_auth_middleware)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/login")
    def login_page(request: Request):
        if request_is_ui_authenticated(request):
            return RedirectResponse("/ui/", status_code=303)
        return {"login": True}

    @app.post("/login")
    async def login_submit(request: Request):
        from urllib.parse import parse_qs

        raw = await request.body()
        parsed = parse_qs(raw.decode("utf-8"), keep_blank_values=True)
        username = (parsed.get("username") or [""])[0].strip()
        password = (parsed.get("password") or [""])[0]
        expected_user, expected_password = expected_ui_credentials(request)
        if not ui_basic_ok(
            username, password, expected_user, expected_password
        ):
            return RedirectResponse("/login?error=1", status_code=303)
        response = RedirectResponse("/ui/", status_code=303)
        set_ui_session_cookie(response, expected_user, expected_password)
        return response

    @app.get("/ui/")
    def ui():
        return {"ui": True}

    with TestClient(app) as client:
        yield client


def test_ui_auth_requires_basic():
    with _stub_client() as client:
        response = client.get("/health")
        assert response.status_code == 401
        assert "Basic" in response.headers.get("www-authenticate", "")
        assert "secret" not in response.text


def test_ui_auth_rejects_bad_password():
    with _stub_client(password="secret") as client:
        response = client.get("/health", headers=_basic_header("demo", "nope"))
        assert response.status_code == 401
        assert "secret" not in response.text
        assert "nope" not in response.text


def test_ui_auth_allows_valid():
    with _stub_client(password="secret") as client:
        response = client.get("/health", headers=_basic_header("demo", "secret"))
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        assert "secret" not in response.text


def test_ui_auth_empty_password_is_503():
    with _stub_client(password="") as client:
        response = client.get("/health")
        assert response.status_code == 503
        body = response.text
        assert "not configured" in body.lower()
        assert response.json()["error"]
        authed = client.get("/health", headers=_basic_header("demo", "anything"))
        assert authed.status_code == 503
        assert "anything" not in authed.text


def test_ui_auth_skips_options():
    with _stub_client() as client:
        response = client.options("/health")
        assert response.status_code != 401
        assert response.status_code != 503


def test_session_token_does_not_contain_password():
    token = make_session_token("demo", "super-secret")
    assert "super-secret" not in token
    assert session_token_ok(token, "demo", "super-secret") is True
    assert session_token_ok(token, "demo", "other") is False
    assert session_token_ok("demo.00" * 16, "demo", "super-secret") is False


def test_login_html_has_no_inline_script():
    html = (
        Path(__file__).resolve().parents[1] / "server" / "static" / "login.html"
    ).read_text(encoding="utf-8")
    assert "<script" not in html.lower()
    assert "super-secret" not in html
    assert 'value="demo"' not in html
    assert ' class="login-error" hidden' in html
    assert is_public_auth_path("/login") is True
    assert is_public_auth_path("/logout") is True
    assert is_public_auth_path("/ui/style.css") is True
    assert is_public_auth_path("/ui/icon.svg") is True
    assert is_public_auth_path("/ui/") is False
    assert is_public_auth_path("/health") is False
    assert is_public_auth_path("/ui/app.js") is False


def test_chat_html_uses_mobile_safe_viewport():
    root = Path(__file__).resolve().parents[1] / "server" / "static"
    html = (root / "index.html").read_text(encoding="utf-8")
    css = (root / "style.css").read_text(encoding="utf-8")
    login = (root / "login.html").read_text(encoding="utf-8")
    js = (root / "app.js").read_text(encoding="utf-8")

    assert "viewport-fit=cover" in html
    assert "interactive-widget=resizes-content" in html
    assert "viewport-fit=cover" in login
    assert "width: 100vw" not in css
    assert "height: 100vh" not in css
    assert "100dvh" in css
    assert "safe-area-inset-top" in css
    assert "safe-area-inset-bottom" in css
    assert "visualViewport" in js
    assert "composer-actions" in html
    assert 'id="effort-label">high<' not in html
    assert ".message.assistant .message-text:empty" in css
    assert "min(76rem" in css
    assert "--chrome-bottom" in css
    assert "grid-template-columns" in css
    assert "ResizeObserver" in js
    assert "setSize" in js
    assert "copyTextToClipboard" in js
    assert 'id="composer-reveal"' in html
    assert "composer-collapsed" in css
    assert "syncComposerCollapse" in js


def test_browser_ui_redirects_to_login():
    with _stub_client() as client:
        response = client.get("/ui/", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/login"


def test_login_page_is_public():
    with _stub_client() as client:
        response = client.get("/login")
        assert response.status_code == 200
        assert "secret" not in response.text


def test_login_form_sets_httponly_cookie():
    with _stub_client(password="secret") as client:
        bad = client.post(
            "/login",
            data={"username": "demo", "password": "nope"},
            follow_redirects=False,
        )
        assert bad.status_code == 303
        assert "error=1" in bad.headers["location"]
        assert UI_SESSION_COOKIE not in bad.cookies
        assert "nope" not in bad.text
        assert "secret" not in bad.text

        ok = client.post(
            "/login",
            data={"username": "demo", "password": "secret"},
            follow_redirects=False,
        )
        assert ok.status_code == 303
        assert ok.headers["location"] == "/ui/"
        cookie = ok.cookies.get(UI_SESSION_COOKIE)
        assert cookie
        assert "secret" not in cookie
        flags = ok.headers.get("set-cookie", "").lower()
        assert "httponly" in flags
        assert "samesite=lax" in flags

        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok"}


def test_ui_app_js_is_not_public():
    with _stub_client() as client:
        response = client.get("/ui/app.js", follow_redirects=False)
        assert response.status_code == 303
        assert response.headers["location"] == "/login"


def test_forged_session_cookie_is_rejected():
    with _stub_client() as client:
        client.cookies.set(UI_SESSION_COOKIE, "demo." + "ab" * 32)
        response = client.get("/health")
        assert response.status_code == 401
        assert "secret" not in response.text

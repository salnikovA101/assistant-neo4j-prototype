"""HTTP-layer practice: CORS, health checks, request bodies, UI Basic auth."""

from __future__ import annotations

import base64
import re
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from pydantic import ValidationError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse, RedirectResponse

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
    ui_cache_control_middleware,
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
    parsed = TextProcessBody(
        text="  hello  ", reasoning_effort="low", profile="ollama_gptoss"
    )
    assert parsed.text == "  hello  "
    assert parsed.reasoning_effort == "low"
    assert parsed.profile == "ollama_gptoss"


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
    assert str(short.value.detail) == "Некорректный ключ LLM. Проверьте значение в настройках."

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
    assert "username" in payload
    assert payload["models"][0]["id"] == "ollama"
    assert "api_key" not in payload["models"][0]


def test_ui_config_models_catalog_hides_secrets():
    from server.core.app import build_ui_config
    from server.utils.config import load_config

    cfg = load_config()
    ollama = cfg.llm.profiles.ollama

    class Pipeline:
        llm = type("L", (), {"model": type("M", (), {"profile": ollama})()})()
        config = cfg

    payload = build_ui_config(Pipeline())
    dumped = str(payload)
    ids = [item["id"] for item in payload["models"]]
    assert ids[0] == "auto"
    assert ids[1:] == [
        "qwen38_max",
        "qwen38_2_4t",
        "deepseek_v4_pro",
        "kimi_k3",
        "glm_52",
        "qwen37_max",
        "qwen38_27b",
        "qwen38_flash",
        "qwen37_plus",
        "qwen37_flash",
    ]
    assert "ollama" not in ids
    assert "ollama_gptoss" not in ids
    assert "qwen_cloud" not in ids
    auto = payload["models"][0]
    flash = next(item for item in payload["models"] if item["id"] == "qwen38_flash")
    kimi = next(item for item in payload["models"] if item["id"] == "kimi_k3")
    glm = next(item for item in payload["models"] if item["id"] == "glm_52")
    assert auto["label"] == "Авто"
    assert auto["reasoning_effort_options"] == []
    assert flash["label"] == "Qwen 3.8 Flash"
    assert flash["reasoning_effort"] == "medium"
    assert flash["reasoning_effort_options"] == ["low", "medium", "xhigh", "off"]
    assert kimi["reasoning_effort_options"] == ["high"]
    assert "none" in glm["reasoning_effort_options"]
    assert "max" in glm["reasoning_effort_options"]
    assert payload["current_profile"] == "auto"
    assert payload["reasoning_effort_options"] == []
    assert "api_key" not in dumped
    secret = (ollama.api_key or "").strip()
    if secret:
        assert secret not in dumped
    gptoss_key = (cfg.llm.profiles.ollama_gptoss.api_key or "").strip()
    if gptoss_key:
        assert gptoss_key not in dumped
    qwen_key = (cfg.llm.profiles.qwen_cloud.api_key or "").strip()
    if qwen_key:
        assert qwen_key not in dumped


def test_request_think_effort_follows_selected_profile():
    from server.core.app import _request_profile_name, _request_think_effort
    from server.utils.config import load_config

    cfg = load_config()

    class Pipeline:
        config = cfg
        llm = type(
            "L",
            (),
            {"model": type("M", (), {"profile": cfg.llm.profiles.ollama})()},
        )()

    pipeline = Pipeline()
    gptoss_off = TextProcessBody(
        text="hi", profile="qwen37_max", reasoning_effort="off"
    )
    gptoss_name = _request_profile_name(pipeline, gptoss_off)
    assert gptoss_name == "qwen37_max"
    assert _request_think_effort(pipeline, gptoss_off, gptoss_name) == "off"

    gptoss_low = TextProcessBody(
        text="hi", profile="qwen37_max", reasoning_effort="low"
    )
    assert _request_think_effort(pipeline, gptoss_low, gptoss_name) is None

    auto_body = TextProcessBody(text="hi", profile="auto", reasoning_effort="high")
    auto_name = _request_profile_name(pipeline, auto_body)
    assert auto_name == "auto"
    assert _request_think_effort(pipeline, auto_body, auto_name) is None

    kimi = TextProcessBody(text="hi", profile="kimi_k3", reasoning_effort="high")
    kimi_name = _request_profile_name(pipeline, kimi)
    assert kimi_name == "kimi_k3"
    assert _request_think_effort(pipeline, kimi, kimi_name) == "high"

    qwen_xhigh = TextProcessBody(
        text="hi", profile="qwen38_flash", reasoning_effort="xhigh"
    )
    qwen_name = _request_profile_name(pipeline, qwen_xhigh)
    assert qwen_name == "qwen38_flash"
    assert _request_think_effort(pipeline, qwen_xhigh, qwen_name) == "xhigh"

    unknown = TextProcessBody(text="hi", profile="other", reasoning_effort="xhigh")
    with pytest.raises(HTTPException) as exc:
        _request_profile_name(pipeline, unknown)
    assert exc.value.status_code == 400
    assert "Неизвестная модель" in str(exc.value.detail)

    from server.core.app import _ensure_turn_options

    stale_auto = TextProcessBody(text="hi", profile="auto", reasoning_effort="high")
    assert _ensure_turn_options(pipeline, stale_auto) == "auto"


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
    assert is_public_auth_path("/healthz") is True
    assert is_public_auth_path("/logout") is False
    assert is_public_auth_path("/ui/login.css") is True
    assert is_public_auth_path("/ui/icon.svg") is True
    assert is_public_auth_path("/ui/") is False
    assert is_public_auth_path("/health") is False
    assert is_public_auth_path("/ui/app.js") is False
    assert is_public_auth_path("/ui/assets/index.js") is True


def test_chat_html_uses_mobile_safe_viewport():
    root = Path(__file__).resolve().parents[1]
    html = (root / "web" / "index.html").read_text(encoding="utf-8")
    css = (root / "web" / "src" / "styles.css").read_text(encoding="utf-8")
    login = (root / "server" / "static" / "login.html").read_text(encoding="utf-8")

    assert "viewport-fit=cover" in html
    assert "interactive-widget=resizes-content" in html
    assert "viewport-fit=cover" in login
    assert "width: 100vw" not in css
    assert "100dvh" in css
    assert "safe-area-inset-top" in css
    assert "safe-area-inset-bottom" in css


def test_stable_ui_assets_are_not_cached_across_rebuilds():
    app = FastAPI()
    app.add_middleware(BaseHTTPMiddleware, dispatch=ui_cache_control_middleware)

    @app.get("/ui/assets/index.js")
    async def stable_script():
        return JSONResponse({"ok": True})

    @app.get("/ui/assets/font-hash.woff2")
    async def hashed_font():
        return JSONResponse({"ok": True})

    @app.get("/api/example")
    async def api_response():
        return JSONResponse({"ok": True})

    with TestClient(app) as client:
        assert client.get("/ui/assets/index.js").headers["cache-control"] == "no-store"
        assert "cache-control" not in client.get("/ui/assets/font-hash.woff2").headers
        assert "cache-control" not in client.get("/api/example").headers


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

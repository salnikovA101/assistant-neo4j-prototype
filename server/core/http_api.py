"""HTTP request models, health payload, CORS regex, UI auth — no pipeline import."""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
import secrets
from typing import Any, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
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


UI_BASIC_REALM = "Neo4j Assistant"
UI_SESSION_COOKIE = "ui_session"
_PUBLIC_EXACT = frozenset({"/login", "/logout"})
_PUBLIC_FILES = frozenset({"/ui/style.css", "/ui/icon.svg"})


def parse_basic_authorization(header: str | None) -> tuple[str, str] | None:
    """Parse ``Basic base64(user:pass)``. Invalid input → None (caller sends 401)."""
    if not header:
        return None
    parts = header.strip().split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "basic":
        return None
    try:
        raw = base64.b64decode(parts[1].encode("ascii"), validate=True)
        decoded = raw.decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    if ":" not in decoded:
        return None
    user, password = decoded.split(":", 1)
    return user, password


def ui_basic_configured(_user: str, password: str) -> bool:
    return bool((password or "").strip())


def _const_eq(left: str, right: str) -> bool:
    a = left.encode("utf-8")
    b = right.encode("utf-8")
    if len(a) != len(b):
        return False
    try:
        return secrets.compare_digest(a, b)
    except (TypeError, ValueError):
        return False


def ui_basic_ok(
    user: str,
    password: str,
    expected_user: str,
    expected_password: str,
) -> bool:
    if not ui_basic_configured(expected_user, expected_password):
        return False
    return _const_eq(user, expected_user) and _const_eq(
        password, expected_password
    )


def unauthorized_basic_response(realm: str = UI_BASIC_REALM) -> Response:
    return Response(
        content="Unauthorized",
        status_code=401,
        media_type="text/plain",
        headers={"WWW-Authenticate": f'Basic realm="{realm}", charset="UTF-8"'},
    )


def password_not_configured_response() -> JSONResponse:
    return JSONResponse(
        {"error": "UI password not configured"},
        status_code=503,
    )


def make_session_token(user: str, password: str) -> str:
    """HMAC of the login name; key is the UI password. Cookie never stores the password."""
    digest = hmac.new(
        password.encode("utf-8"),
        f"ui-session|{user}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{user}.{digest}"


def session_token_ok(
    raw: str | None, expected_user: str, expected_password: str
) -> bool:
    if not raw or not ui_basic_configured(expected_user, expected_password):
        return False
    expected = make_session_token(expected_user, expected_password)
    return _const_eq(raw, expected)


def is_public_auth_path(path: str) -> bool:
    normalized = path.rstrip("/") or "/"
    if normalized in _PUBLIC_EXACT or path in _PUBLIC_EXACT:
        return True
    return path in _PUBLIC_FILES


def expected_ui_credentials(request: Request) -> tuple[str, str]:
    user = str(
        getattr(request.app.state, "ui_basic_user", "demo") or "demo"
    ).strip() or "demo"
    password = str(
        getattr(request.app.state, "ui_basic_password", "") or ""
    ).strip()
    return user, password


def request_is_ui_authenticated(request: Request) -> bool:
    expected_user, expected_password = expected_ui_credentials(request)
    if not ui_basic_configured(expected_user, expected_password):
        return False
    if session_token_ok(
        request.cookies.get(UI_SESSION_COOKIE),
        expected_user,
        expected_password,
    ):
        return True
    parsed = parse_basic_authorization(request.headers.get("Authorization"))
    if parsed is None:
        return False
    return ui_basic_ok(
        parsed[0], parsed[1], expected_user, expected_password
    )


def set_ui_session_cookie(response: Response, user: str, password: str) -> None:
    response.set_cookie(
        UI_SESSION_COOKIE,
        make_session_token(user, password),
        httponly=True,
        samesite="lax",
        path="/",
        secure=False,
    )
    response.headers["Cache-Control"] = "no-store"


def clear_ui_session_cookie(response: Response) -> None:
    response.delete_cookie(UI_SESSION_COOKIE, path="/")
    response.headers["Cache-Control"] = "no-store"


def unauthenticated_response(request: Request) -> Response:
    """Browsers hitting /ui get the login page; API/curl get HTTP Basic 401."""
    path = request.url.path
    if request.method in ("GET", "HEAD") and path.startswith("/ui"):
        return RedirectResponse(url="/login", status_code=303)
    return unauthorized_basic_response()


async def ui_auth_middleware(request: Request, call_next):
    """Cookie or HTTP Basic on every path except login assets and CORS preflight."""
    if request.method == "OPTIONS":
        return await call_next(request)
    if is_public_auth_path(request.url.path):
        return await call_next(request)

    expected_user, expected_password = expected_ui_credentials(request)
    if not ui_basic_configured(expected_user, expected_password):
        return password_not_configured_response()

    if request_is_ui_authenticated(request):
        return await call_next(request)

    return unauthenticated_response(request)


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

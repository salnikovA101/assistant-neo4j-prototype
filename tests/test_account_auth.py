from __future__ import annotations

import base64
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware

from server.core.app_store import AppStore
from server.core.http_api import (
    UI_SESSION_COOKIE,
    set_account_session_cookie,
    ui_auth_middleware,
    workspace_session_cookie,
)


def test_database_cookie_basic_and_origin_guard(tmp_path):
    state: dict[str, str] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store = AppStore(
            str(tmp_path / "auth.db"),
            workspaces={"packaging": "packaging-run", "kefir": "kefir-run"},
        )
        await store.open()
        user = await store.create_user("worker", "a sufficiently long password")
        kefir_user = await store.create_user(
            "kefir-worker", "another sufficiently long password", workspace="kefir"
        )
        state["token"] = await store.create_session(user.id)
        state["kefir_token"] = await store.create_session(kefir_user.id)
        app.state.app_store = store
        app.state.workspace_run_ids = {
            "packaging": "packaging-run",
            "kefir": "kefir-run",
        }
        app.state.auth_trusted_origins = ""
        yield
        await store.close()

    app = FastAPI(lifespan=lifespan)
    app.add_middleware(BaseHTTPMiddleware, dispatch=ui_auth_middleware)

    @app.get("/private")
    async def private(request: Request):
        return {"username": request.state.account_user.username}

    @app.post("/private")
    async def private_post(request: Request):
        return {"username": request.state.account_user.username}

    with TestClient(app) as client:
        workspace_headers = {"X-Workspace": "packaging"}
        assert client.get("/private").status_code == 404
        assert client.get(
            "/private", headers={"X-Workspace": "unknown"}
        ).status_code == 404
        assert client.get("/ui/unknown/", follow_redirects=False).status_code == 404
        assert client.get("/private", headers=workspace_headers).status_code == 401

        client.cookies.set(UI_SESSION_COOKIE, state["token"])
        assert client.get("/private", headers=workspace_headers).json() == {"username": "worker"}
        assert client.get(
            "/private", headers={"X-Workspace": "kefir"}
        ).status_code == 401
        client.cookies.set(workspace_session_cookie("kefir"), state["kefir_token"])
        assert client.get(
            "/private", headers={"X-Workspace": "kefir"}
        ).json() == {"username": "kefir-worker"}
        assert client.get("/private", headers=workspace_headers).json() == {"username": "worker"}
        rejected = client.post(
            "/private",
            headers={"Origin": "https://evil.example", **workspace_headers},
        )
        assert rejected.status_code == 403

        client.cookies.clear()
        basic = base64.b64encode(b"worker:a sufficiently long password").decode()
        response = client.post(
            "/private",
            headers={
                "Authorization": f"Basic {basic}",
                "Origin": "https://evil.example",
                **workspace_headers,
            },
        )
        assert response.status_code == 200


def test_account_cookie_attributes():
    response = JSONResponse({"ok": True})
    set_account_session_cookie(response, "raw-token", secure=True, max_age_days=30)
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=lax" in cookie
    assert "secure" in cookie
    assert "max-age=2592000" in cookie
    assert "ui_session_packaging=" in cookie

import asyncio
import hashlib
import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import parse_qs, quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from server.utils.config import load_config, resolve_request_profile, ui_selectable_profiles
from server.core.db import get_driver
from server.core.app_store import AccountUser, AppStore
from server.core.http_api import (
    CORS_ORIGIN_RE,
    GraphExploreBody,
    GraphVizBody,
    TextProcessBody,
    account_user_from_request,
    build_health,
    clear_ui_session_cookie,
    llm_api_key_from_request,
    login_attempt_limiter,
    session_id_from_request,
    set_account_session_cookie,
    ui_auth_middleware,
)
from server.core.pipeline import ServerPipeline
from server.core.sessions import session_store
from server.core.turn_state import (
    DEFAULT_SEARCH_DEPTH,
    SEARCH_DEPTHS,
    parse_search_depth,
)
from server.llm.base import parse_ui_think_effort, profile_think_efforts
from server.tools.graph_explore import build_graph_explore_payload
from server.tools.graph_viz import build_graph_viz_payload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

_LOGIN_PAGE = Path(__file__).resolve().parents[1] / "static" / "login.html"
_MAX_LOGIN_BODY = 4096
_LOGIN_SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": (
        "default-src 'none'; "
        "style-src 'self' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'none'"
    ),
}


async def _form_fields(request: Request) -> dict[str, str]:
    """Parse urlencoded login fields. Do not log: the body may contain the UI password."""
    raw = await request.body()
    if not raw or len(raw) > _MAX_LOGIN_BODY:
        return {}
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return {}
    parsed = parse_qs(text, keep_blank_values=True, max_num_fields=16)
    return {key: (values[0] if values else "") for key, values in parsed.items()}


def _login_html(show_error: bool, *, no_accounts: bool = False) -> HTMLResponse:
    html = _LOGIN_PAGE.read_text(encoding="utf-8")
    if show_error:
        html = html.replace(' class="login-error" hidden', ' class="login-error"', 1)
    if no_accounts:
        html = html.replace(
            ' id="login-setup" class="login-error" hidden',
            ' id="login-setup" class="login-error"',
            1,
        )
    return HTMLResponse(html, headers=_LOGIN_SECURITY_HEADERS)


def _request_profile_name(
    pipeline: ServerPipeline, body: TextProcessBody
) -> str:
    return resolve_request_profile(pipeline.config.llm, body.profile)


def _request_think_effort(
    pipeline: ServerPipeline, body: TextProcessBody, profile_name: str
) -> str | None:
    profile = getattr(pipeline.config.llm.profiles, profile_name, None)
    if profile is None:
        profile = pipeline.llm.provider_for(profile_name).profile
    return parse_ui_think_effort(
        body.reasoning_effort, profile_think_efforts(profile)
    )


class ConversationPatchBody(BaseModel):
    title: str = Field(default="", max_length=200)


def _current_user(request: Request) -> AccountUser:
    user = getattr(request.state, "account_user", None)
    if not isinstance(user, AccountUser):
        raise HTTPException(status_code=401, detail="Unauthorized")
    return user


def _conversation_session_key(user_id: str, conversation_id: str) -> str:
    """Opaque cache key accepted by SessionStore without exposing account ids."""
    raw = f"{user_id}\0{conversation_id}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


async def _owned_session(request: Request) -> tuple[AccountUser, str, str]:
    user = _current_user(request)
    conversation_id = session_id_from_request(request)
    store: AppStore = request.app.state.app_store
    if not await store.conversation_owned(user.id, conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    session_key = _conversation_session_key(user.id, conversation_id)
    return user, conversation_id, session_key


async def _hydrate_session(
    store: AppStore,
    user: AccountUser,
    conversation_id: str,
    session_key: str,
    history_len: int,
) -> None:
    turns, sources = await store.load_model_context(user.id, conversation_id, history_len)
    session_store.hydrate(session_key, history_len, turns, sources)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом: загрузка моделей при старте, выгрузка при остановке."""
    config = load_config()

    app_store = AppStore(config.app_db_path)
    await app_store.open()
    app.state.app_store = app_store
    app.state.auth_cookie_secure = bool(config.auth_cookie_secure)
    app.state.auth_session_days = max(1, int(config.auth_session_days))
    app.state.auth_trusted_origins = config.auth_trusted_origins

    if config.debug_mode:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        for name in [
            "httpx",
            "faster_whisper",
            "faster_qwen3_tts",
            "qwen_tts",
            "huggingface_hub",
            "neo4j",
        ]:
            logging.getLogger(name).setLevel(logging.ERROR)

    rid = (config.run_id or "").strip()
    if not rid:
        logger.warning(
            "run_id is empty: V6 ANN/bridges search the full vector index"
        )
    else:
        logger.info("V6 corpus run_id=%s", rid)
    if not config.rerank_enabled:
        logger.warning("rerank_enabled=false: S2b keeps ANN order by sim")
    else:
        logger.info("V6 S2b rerank enabled")

    account_count = await app_store.active_user_count()
    logger.info("SQLite accounts ready: active_users=%s db=%s", account_count, config.app_db_path)

    pipeline: ServerPipeline | None = None
    pipeline_started = False
    try:
        logger.info("Инициализация ServerPipeline...")
        pipeline = ServerPipeline(config)
        await pipeline.startup()
        pipeline_started = True
        app.state.pipeline = pipeline
        if config.audio_enabled:
            logger.info("Voice Assistant Server готов!")
        else:
            logger.info("Text backend готов (STT/TTS отключены)!")
        yield
    finally:
        logger.info("Завершение работы...")
        if pipeline is not None and pipeline_started:
            await pipeline.shutdown()
        await app_store.close()


app = FastAPI(title="Voice Assistant Server", lifespan=lifespan)

# Last added middleware runs first. Auth inner, CORS outer so 401 gets CORS headers.
app.add_middleware(BaseHTTPMiddleware, dispatch=ui_auth_middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=CORS_ORIGIN_RE,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "X-Session-Id",
        "X-LLM-Api-Key",
    ],
    expose_headers=[
        "Recognized-Text",
        "LLM-Response",
        "Sample-Rate",
        "Channels",
        "Sample-Width",
    ],
)


def _audio_disabled_response() -> JSONResponse:
    return JSONResponse(
        {"error": "Аудио отключено (audio_enabled=false). Используйте /process_text_test"},
        status_code=503,
    )


def _turn_id(raw: str | None) -> str:
    if not raw:
        return str(uuid.uuid4())
    try:
        return str(uuid.UUID(raw))
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=400, detail="Invalid turn_id") from exc


async def _persistent_stream(
    request: Request,
    body: TextProcessBody,
    *,
    user: AccountUser,
    conversation_id: str,
    session_key: str,
    assistant_message_id: str,
):
    pipeline: ServerPipeline = request.app.state.pipeline
    store: AppStore = request.app.state.app_store
    profile_name = _request_profile_name(pipeline, body)
    think_effort = _request_think_effort(pipeline, body, profile_name)
    search_depth = parse_search_depth(body.search_depth)
    started = time.monotonic()
    answer = ""
    thinking = ""
    tools: list[dict] = []
    steps: list[dict] = []
    raw_content = ""
    history_tools: list[dict] = []
    graph_run_id = ""
    graph_chains: list[dict] = []
    graph_chain_count = 0
    terminal = False

    def update_tool(data: dict, done: bool = False) -> None:
        nonlocal tools, steps
        tool_id = str(data.get("id") or (tools[-1]["id"] if tools else "tool"))
        existing = next((item for item in tools if item.get("id") == tool_id), None)
        if done:
            card = dict(existing or {"id": tool_id, "name": "ask_subgraph"})
            card.update({
                "status": "error" if data.get("ok") is False else "done",
                "result": str(data.get("result") or data.get("preview") or ""),
            })
        else:
            card = {
                "id": tool_id,
                "name": str(data.get("name") or "ask_subgraph"),
                "status": "running",
                "args": data.get("arguments", data.get("args")),
            }
        tools = [item for item in tools if item.get("id") != tool_id] + [card]
        step = {"kind": "tool", **card}
        steps = [item for item in steps if item.get("kind") != "tool" or item.get("id") != tool_id] + [step]

    async def persist(status: str) -> None:
        payload: dict[str, object] = {
            "thinking": thinking,
            "tools": tools,
            "steps": steps,
            "elapsedSec": max(1, round(time.monotonic() - started)),
        }
        if graph_run_id:
            payload["graphRunId"] = graph_run_id
            payload["graphChainCount"] = graph_chain_count
        session = session_store.get_or_create(session_key, pipeline.config.llm.history_len)
        await store.finish_turn(
            conversation_id,
            assistant_message_id,
            text=answer,
            status=status,
            payload=payload,
            raw_text=raw_content if status == "done" else "",
            tool_messages=history_tools if status == "done" else [],
            sources=session.sources.snapshot(),
            graph_run_id=graph_run_id,
            graph_chains=graph_chains,
        )

    try:
        async for event in pipeline.process_text_stream(
            body.text.strip(),
            request,
            think_effort=think_effort,
            session_id=session_key,
            search_depth=search_depth,
            api_key=llm_api_key_from_request(request),
            profile_name=profile_name,
        ):
            data = dict(event.data)
            if event.type == "thinking":
                delta = str(data.get("delta") or "")
                thinking += delta
                if steps and steps[-1].get("kind") == "think":
                    steps[-1] = {"kind": "think", "text": str(steps[-1].get("text") or "") + delta}
                else:
                    steps.append({"kind": "think", "text": delta})
            elif event.type == "content":
                answer += str(data.get("delta") or "")
            elif event.type == "content_rewind":
                rewind = str(data.get("text") or "")
                if rewind and answer.endswith(rewind):
                    answer = answer[:-len(rewind)]
            elif event.type == "tool_call":
                update_tool(data)
            elif event.type == "tool_result":
                update_tool(data, done=True)
            elif event.type == "done":
                answer = str(data.get("final_content") or answer)
                raw_content = str(data.pop("_raw_content", "") or "")
                history_tools = data.pop("_history_tool_messages", []) or []
                graph_chains = data.pop("_graph_chains", []) or []
                graph_run_id = str(data.get("graph_run_id") or "")
                graph_chain_count = int(data.get("graph_chain_count") or 0)
                await persist("done")
                terminal = True
            elif event.type == "error":
                answer = str(data.get("message") or "Ошибка LLM")
                await persist("error")
                terminal = True
            event.data = data
            yield event.to_sse()
    except Exception:
        logger.exception("Failed persistent stream conversation=%s", conversation_id)
        if not terminal:
            answer = answer or "Поток оборвался"
            await persist("error")
            terminal = True
        raise
    finally:
        if not terminal:
            await asyncio.shield(persist("aborted"))


async def _run_persistent_text(request: Request, body: TextProcessBody) -> str:
    pipeline: ServerPipeline = request.app.state.pipeline
    store: AppStore = request.app.state.app_store
    user, conversation_id, session_key = await _owned_session(request)
    await _hydrate_session(store, user, conversation_id, session_key, pipeline.config.llm.history_len)
    try:
        _, assistant_message_id = await store.begin_turn(
            user.id, conversation_id, _turn_id(body.turn_id), body.text.strip()
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail="Duplicate turn_id") from exc
    async for _ in _persistent_stream(
        request,
        body,
        user=user,
        conversation_id=conversation_id,
        session_key=session_key,
        assistant_message_id=assistant_message_id,
    ):
        pass
    detail = await store.get_conversation(user.id, conversation_id)
    if detail:
        message = next(
            (item for item in detail["messages"] if item["id"] == assistant_message_id),
            None,
        )
        if message:
            return str(message.get("text") or "")
    return ""


@app.post("/process")
async def process_audio(request: Request):
    """
    Принимает WAV-аудио, возвращает стрим PCM-чанков.

    Метаданные (распознанный текст и ответ LLM) передаются в заголовках:
    - Recognized-Text: URL-encoded распознанный текст
    - LLM-Response: URL-encoded ответ LLM
    """
    pipeline: ServerPipeline = request.app.state.pipeline
    if not pipeline.config.audio_enabled:
        return _audio_disabled_response()

    wav_bytes = await request.body()

    if not wav_bytes:
        return JSONResponse({"error": "Пустое тело запроса"}, status_code=400)

    user, conversation_id, session_key = await _owned_session(request)
    store: AppStore = request.app.state.app_store
    await _hydrate_session(
        store, user, conversation_id, session_key, pipeline.config.llm.history_len
    )
    recognized, answer = await pipeline.process_audio(wav_bytes, session_id=session_key)

    if not recognized:
        return JSONResponse({"error": "Речь не распознана"}, status_code=422)

    return StreamingResponse(
        pipeline.synthesize(answer, request),
        media_type="audio/pcm",
        headers={
            "Recognized-Text": quote(recognized, safe=""),
            "LLM-Response": quote(answer, safe=""),
            "Sample-Rate": "24000",
            "Channels": "1",
            "Sample-Width": "2",
        },
    )


@app.post("/stt")
async def stt_only(request: Request):
    """
    Только STT: принимает WAV-аудио, возвращает распознанный текст (JSON).
    Используется веб-клиентом для мгновенного отображения результата STT.
    """
    pipeline: ServerPipeline = request.app.state.pipeline
    if not pipeline.config.audio_enabled or pipeline.stt is None:
        return _audio_disabled_response()

    wav_bytes = await request.body()

    if not wav_bytes:
        return JSONResponse({"error": "Пустое тело запроса"}, status_code=400)

    text = await pipeline.stt.transcribe_bytes(wav_bytes)
    if not text:
        return JSONResponse({"error": "Речь не распознана"}, status_code=422)

    return JSONResponse({"text": text})


@app.post("/process_text")
async def process_text(request: Request, body: TextProcessBody):
    """
    Принимает текст JSON.
    При audio_enabled=true — стрим PCM; иначе — JSON {"answer": "..."} (как /process_text_test).
    """
    pipeline: ServerPipeline = request.app.state.pipeline
    text = body.text.strip()
    if not text:
        return JSONResponse({"error": "Пустой текст"}, status_code=400)

    answer = await _run_persistent_text(request, body)

    if not pipeline.config.audio_enabled:
        return JSONResponse({"answer": answer})

    return StreamingResponse(
        pipeline.synthesize(answer, request),
        media_type="audio/pcm",
        headers={
            "LLM-Response": quote(answer, safe=""),
            "Sample-Rate": "24000",
            "Channels": "1",
            "Sample-Width": "2",
        },
    )


@app.post("/process_text_stream")
async def process_text_stream(request: Request, body: TextProcessBody):
    """
    SSE stream of assistant events: thinking, tool_call, tool_result, content, done, error.

    Request body: {"text": "вопрос пользователя", "profile": "ollama"|"ollama_gptoss"|"qwen_cloud",
    "reasoning_effort": "<profile think_efforts>", "search_depth": "low"|"medium"|"high"}
    """
    text = body.text.strip()
    if not text:
        return JSONResponse({"error": "Пустой текст"}, status_code=400)
    pipeline: ServerPipeline = request.app.state.pipeline
    store: AppStore = request.app.state.app_store
    user, conversation_id, session_key = await _owned_session(request)
    await _hydrate_session(
        store, user, conversation_id, session_key, pipeline.config.llm.history_len
    )
    try:
        _, assistant_message_id = await store.begin_turn(
            user.id, conversation_id, _turn_id(body.turn_id), text
        )
    except ValueError:
        return JSONResponse({"error": "Этот запрос уже был отправлен"}, status_code=409)

    return StreamingResponse(
        _persistent_stream(
            request,
            body,
            user=user,
            conversation_id=conversation_id,
            session_key=session_key,
            assistant_message_id=assistant_message_id,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/process_text_test")
async def process_text_test(request: Request, body: TextProcessBody):
    """
    Принимает текст JSON, возвращает ответ LLM (без TTS).
    Специально для скриптов тестирования.

    Request body: {"text": "вопрос пользователя", "profile": "ollama"|"ollama_gptoss"|"qwen_cloud",
    "reasoning_effort": "<profile think_efforts>", "search_depth": "low"|"medium"|"high"}
    """
    pipeline: ServerPipeline = request.app.state.pipeline
    text = body.text.strip()
    if not text:
        return JSONResponse({"error": "Пустой текст"}, status_code=400)

    answer = await _run_persistent_text(request, body)

    return JSONResponse({"answer": answer})


@app.post("/clear_history")
async def clear_history(request: Request):
    """Сбрасывает историю чата и контекст этой вкладки."""
    user, conversation_id, session_key = await _owned_session(request)
    store: AppStore = request.app.state.app_store
    await store.clear_conversation(user.id, conversation_id)
    session_store.drop(session_key)
    return JSONResponse({"status": "ok"})


@app.get("/health")
async def health(request: Request):
    """Готовность: pipeline + LLM-объект + Neo4j verify_connectivity."""
    pipeline = getattr(request.app.state, "pipeline", None)
    status_code, payload = await build_health(pipeline)
    return JSONResponse(payload, status_code=status_code)


def _ui_profile_for(pipeline: ServerPipeline, name: str):
    """Yaml profile for the catalog. Runtime only if that id is missing in yaml."""
    llm_cfg = pipeline.config.llm
    cfg_profile = getattr(llm_cfg.profiles, name, None)
    if cfg_profile is not None:
        return cfg_profile
    if name == llm_cfg.current_profile:
        return getattr(getattr(pipeline.llm, "model", None), "profile", None)
    return None


def _ui_model_entry(name: str, profile) -> dict:
    options = list(profile_think_efforts(profile))
    default_effort = parse_ui_think_effort(profile.think_effort, options)
    if default_effort is None:
        default_effort = options[0] if options else ""
    label = (getattr(profile, "display_name", None) or "").strip() or name
    return {
        "id": name,
        "label": label,
        "think": bool(profile.think) and bool(options),
        "reasoning_effort": default_effort,
        "reasoning_effort_options": options,
    }


def build_ui_config(pipeline: ServerPipeline) -> dict:
    """Public UI defaults. Never includes API keys."""
    llm_cfg = pipeline.config.llm
    models = []
    for name in ui_selectable_profiles(llm_cfg):
        profile = _ui_profile_for(pipeline, name)
        if profile is None:
            continue
        models.append(_ui_model_entry(name, profile))

    default_name = llm_cfg.current_profile
    default_profile = _ui_profile_for(pipeline, default_name)
    if default_profile is None and models:
        default_name = models[0]["id"]
        default_profile = _ui_profile_for(pipeline, default_name)
    if default_profile is None:
        default_profile = getattr(getattr(pipeline.llm, "model", None), "profile", None)

    options = list(profile_think_efforts(default_profile)) if default_profile else []
    default_effort = (
        parse_ui_think_effort(default_profile.think_effort, options)
        if default_profile
        else None
    )
    if default_effort is None:
        default_effort = options[0] if options else ""

    ollama = getattr(llm_cfg.profiles, "ollama", None)
    key_configured = bool((getattr(ollama, "api_key", None) or "").strip())
    if not key_configured and default_profile is not None:
        key_configured = bool((default_profile.api_key or "").strip())
    if not key_configured:
        runtime = getattr(getattr(pipeline.llm, "model", None), "profile", None)
        if runtime is not None:
            key_configured = bool((runtime.api_key or "").strip())

    return {
        "think": bool(default_profile and default_profile.think) and bool(options),
        "reasoning_effort": default_effort,
        "reasoning_effort_options": options,
        "search_depth": DEFAULT_SEARCH_DEPTH,
        "search_depth_options": list(SEARCH_DEPTHS),
        "max_searches_per_answer": (
            max(1, int(default_profile.max_turns)) if default_profile else 2
        ),
        "audio_enabled": bool(pipeline.config.audio_enabled),
        "current_profile": default_name,
        "llm_key_configured": key_configured,
        "username": "",
        "models": models,
    }


@app.get("/ui_config")
async def ui_config(request: Request):
    """Defaults for the web UI (reasoning effort, search depth, audio)."""
    pipeline: ServerPipeline = request.app.state.pipeline
    payload = build_ui_config(pipeline)
    payload["username"] = _current_user(request).username
    return payload


@app.get("/api/me")
async def account_me(request: Request):
    user = _current_user(request)
    return {"id": user.id, "username": user.username}


@app.get("/api/conversations")
async def conversations_list(request: Request, limit: int = 50, before: str | None = None):
    user = _current_user(request)
    store: AppStore = request.app.state.app_store
    try:
        return await store.list_conversations(user.id, limit=limit, before=before)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/conversations")
async def conversations_create(request: Request):
    user = _current_user(request)
    store: AppStore = request.app.state.app_store
    return JSONResponse(await store.create_conversation(user.id), status_code=201)


@app.get("/api/conversations/{conversation_id}")
async def conversations_get(request: Request, conversation_id: str):
    user = _current_user(request)
    store: AppStore = request.app.state.app_store
    payload = await store.get_conversation(user.id, conversation_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return payload


@app.patch("/api/conversations/{conversation_id}")
async def conversations_patch(
    request: Request, conversation_id: str, body: ConversationPatchBody
):
    user = _current_user(request)
    store: AppStore = request.app.state.app_store
    if not await store.rename_conversation(user.id, conversation_id, body.title):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "ok"}


@app.delete("/api/conversations/{conversation_id}")
async def conversations_delete(request: Request, conversation_id: str):
    user = _current_user(request)
    store: AppStore = request.app.state.app_store
    if not await store.delete_conversation(user.id, conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    session_store.drop(_conversation_session_key(user.id, conversation_id))
    return {"status": "ok"}


@app.post("/graph_viz")
async def get_graph_viz(request: Request, body: GraphVizBody):
    """Hydrate accepted chains for the lightweight graph modal. No LLM calls."""
    user = _current_user(request)
    store: AppStore = request.app.state.app_store
    chains = await store.get_graph_run(user.id, body.graph_run_id)
    if chains is None:
        return JSONResponse(
            {"error": "Graph run not found"},
            status_code=404,
        )

    payload = await build_graph_viz_payload(get_driver(), chains)
    return JSONResponse(payload)


@app.post("/graph_explore")
async def graph_explore(request: Request, body: GraphExploreBody):
    """Text search + limit over the corpus graph. No Cypher from the client, no LLM."""
    pipeline: ServerPipeline = request.app.state.pipeline
    payload = await build_graph_explore_payload(
        get_driver(),
        q=body.q,
        limit=body.limit,
        field=body.field,
        run_id=(pipeline.config.run_id or "").strip(),
    )
    return JSONResponse(payload)


@app.get("/login")
async def login_page(request: Request):
    """Visual login form. Public. Already-authed users go to the chat."""
    if await account_user_from_request(request):
        return RedirectResponse("/ui/", status_code=303)
    store: AppStore = request.app.state.app_store
    return _login_html(
        show_error="error" in request.query_params,
        no_accounts=await store.user_count() == 0,
    )


@app.post("/login")
async def login_submit(request: Request):
    fields = await _form_fields(request)
    username = (fields.get("username") or "").strip()
    password = fields.get("password") or ""
    ip = request.client.host if request.client else "unknown"
    if not login_attempt_limiter.allowed(ip, username):
        return RedirectResponse("/login?error=1", status_code=303)
    store: AppStore = request.app.state.app_store
    user = await store.authenticate(username, password)
    if user is None:
        login_attempt_limiter.failure(ip, username)
        return RedirectResponse("/login?error=1", status_code=303)
    login_attempt_limiter.success(ip, username)
    token = await store.create_session(
        user.id, lifetime_days=request.app.state.auth_session_days
    )
    response = RedirectResponse("/ui/", status_code=303)
    set_account_session_cookie(
        response,
        token,
        secure=request.app.state.auth_cookie_secure,
        max_age_days=request.app.state.auth_session_days,
    )
    return response


@app.post("/logout")
async def logout(request: Request):
    store: AppStore = request.app.state.app_store
    await store.revoke_session(request.cookies.get("ui_session"))
    response = RedirectResponse("/login", status_code=303)
    clear_ui_session_cookie(response)
    return response


def _ui_static_dir() -> str:
    dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if dist.is_dir() and (dist / "index.html").is_file():
        return str(dist)
    return "server/static"


# Веб-интерфейс: http://localhost:8000/ui/
app.mount("/ui", StaticFiles(directory=_ui_static_dir(), html=True), name="ui")

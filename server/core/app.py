import logging
import sys
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from server.utils.config import load_config
from server.core.db import get_driver
from server.core.graph_runs import graph_run_store
from server.core.http_api import (
    CORS_ORIGIN_RE,
    GraphVizBody,
    TextProcessBody,
    build_health,
    llm_api_key_from_request,
    session_id_from_request,
)
from server.core.pipeline import ServerPipeline
from server.core.turn_state import (
    DEFAULT_SEARCH_DEPTH,
    SEARCH_DEPTHS,
    parse_search_depth,
)
from server.llm.base import parse_ui_think_effort, profile_think_efforts
from server.tools.graph_viz import build_graph_viz_payload

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def _request_think_effort(
    pipeline: ServerPipeline, body: TextProcessBody
) -> str | None:
    profile = pipeline.llm.model.profile
    return parse_ui_think_effort(
        body.reasoning_effort, profile_think_efforts(profile)
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом: загрузка моделей при старте, выгрузка при остановке."""
    config = load_config()

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

    logger.info("Инициализация ServerPipeline...")
    pipeline = ServerPipeline(config)
    await pipeline.startup()

    app.state.pipeline = pipeline
    if config.audio_enabled:
        logger.info("Voice Assistant Server готов!")
    else:
        logger.info("Text backend готов (STT/TTS отключены)!")

    yield

    logger.info("Завершение работы...")
    await pipeline.shutdown()


app = FastAPI(title="Voice Assistant Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=CORS_ORIGIN_RE,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-Session-Id", "X-LLM-Api-Key"],
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

    recognized, answer = await pipeline.process_audio(
        wav_bytes, session_id=session_id_from_request(request)
    )

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
    think_effort = _request_think_effort(pipeline, body)
    search_depth = parse_search_depth(body.search_depth)

    if not text:
        return JSONResponse({"error": "Пустой текст"}, status_code=400)

    answer = await pipeline.process_text(
        text,
        think_effort=think_effort,
        session_id=session_id_from_request(request),
        search_depth=search_depth,
        api_key=llm_api_key_from_request(request),
    )

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

    Request body: {"text": "вопрос пользователя", "reasoning_effort": "<profile think_efforts>",
    "search_depth": "low"|"medium"|"high"}
    """
    pipeline: ServerPipeline = request.app.state.pipeline
    text = body.text.strip()
    think_effort = _request_think_effort(pipeline, body)
    search_depth = parse_search_depth(body.search_depth)

    if not text:
        return JSONResponse({"error": "Пустой текст"}, status_code=400)

    session_id = session_id_from_request(request)

    async def event_generator():
        async for event in pipeline.process_text_stream(
            text,
            request,
            think_effort=think_effort,
            session_id=session_id,
            search_depth=search_depth,
            api_key=llm_api_key_from_request(request),
        ):
            yield event.to_sse()

    return StreamingResponse(
        event_generator(),
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

    Request body: {"text": "вопрос пользователя", "reasoning_effort": "<profile think_efforts>",
    "search_depth": "low"|"medium"|"high"}
    """
    pipeline: ServerPipeline = request.app.state.pipeline
    text = body.text.strip()
    think_effort = _request_think_effort(pipeline, body)
    search_depth = parse_search_depth(body.search_depth)

    if not text:
        return JSONResponse({"error": "Пустой текст"}, status_code=400)

    answer = await pipeline.process_text(
        text,
        think_effort=think_effort,
        session_id=session_id_from_request(request),
        search_depth=search_depth,
        api_key=llm_api_key_from_request(request),
    )

    return JSONResponse({"answer": answer})


@app.post("/clear_history")
async def clear_history(request: Request):
    """Сбрасывает историю чата и контекст этой вкладки."""
    pipeline: ServerPipeline = request.app.state.pipeline
    pipeline.clear_history(session_id=session_id_from_request(request))
    return JSONResponse({"status": "ok"})


@app.get("/health")
async def health(request: Request):
    """Готовность: pipeline + LLM-объект + Neo4j verify_connectivity."""
    pipeline = getattr(request.app.state, "pipeline", None)
    status_code, payload = await build_health(pipeline)
    return JSONResponse(payload, status_code=status_code)


def build_ui_config(pipeline: ServerPipeline) -> dict:
    """Public UI defaults. Never includes API keys."""
    profile = pipeline.llm.model.profile
    options = list(profile_think_efforts(profile))
    default_effort = parse_ui_think_effort(profile.think_effort, options)
    if default_effort is None:
        default_effort = options[0] if options else ""
    return {
        "think": bool(profile.think) and bool(options),
        "reasoning_effort": default_effort,
        "reasoning_effort_options": options,
        "search_depth": DEFAULT_SEARCH_DEPTH,
        "search_depth_options": list(SEARCH_DEPTHS),
        "max_searches_per_answer": max(1, int(profile.max_turns)),
        "audio_enabled": bool(pipeline.config.audio_enabled),
        "current_profile": pipeline.config.llm.current_profile,
        "llm_key_configured": bool((profile.api_key or "").strip()),
    }


@app.get("/ui_config")
async def ui_config(request: Request):
    """Defaults for the web UI (reasoning effort, search depth, audio)."""
    pipeline: ServerPipeline = request.app.state.pipeline
    return build_ui_config(pipeline)


@app.post("/graph_viz")
async def get_graph_viz(body: GraphVizBody):
    """Hydrate accepted chains for the lightweight graph modal. No LLM calls."""
    chains = graph_run_store.get(body.graph_run_id)
    if chains is None:
        return JSONResponse(
            {"error": "Graph run not found or expired"},
            status_code=404,
        )

    payload = await build_graph_viz_payload(get_driver(), chains)
    return JSONResponse(payload)


# Веб-интерфейс: http://localhost:8000/ui/
app.mount("/ui", StaticFiles(directory="server/static", html=True), name="ui")

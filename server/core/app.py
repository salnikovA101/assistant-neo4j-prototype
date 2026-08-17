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
    session_id_from_request,
)
from server.core.pipeline import ServerPipeline
from server.core.turn_state import (
    DEFAULT_SEARCH_DEPTH,
    SEARCH_DEPTHS,
    parse_search_depth,
)
from server.llm.base import UI_THINK_EFFORTS, parse_ui_think_effort
from server.tools.graph_viz import build_graph_viz_payload
from server.utils.tracing import init_tracing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом: загрузка моделей при старте, выгрузка при остановке."""
    config = load_config()
    init_tracing()

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
    allow_headers=["Content-Type", "Accept", "X-Session-Id"],
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
    think_effort = parse_ui_think_effort(body.reasoning_effort)
    search_depth = parse_search_depth(body.search_depth)

    if not text:
        return JSONResponse({"error": "Пустой текст"}, status_code=400)

    answer = await pipeline.process_text(
        text,
        think_effort=think_effort,
        session_id=session_id_from_request(request),
        search_depth=search_depth,
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

    Request body: {"text": "вопрос пользователя", "reasoning_effort": "xhigh"|"medium"|"low",
    "search_depth": "low"|"medium"|"high"}
    """
    pipeline: ServerPipeline = request.app.state.pipeline
    text = body.text.strip()
    think_effort = parse_ui_think_effort(body.reasoning_effort)
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

    Request body: {"text": "вопрос пользователя", "reasoning_effort": "xhigh"|"medium"|"low",
    "search_depth": "low"|"medium"|"high"}
    """
    pipeline: ServerPipeline = request.app.state.pipeline
    text = body.text.strip()
    think_effort = parse_ui_think_effort(body.reasoning_effort)
    search_depth = parse_search_depth(body.search_depth)

    if not text:
        return JSONResponse({"error": "Пустой текст"}, status_code=400)

    answer = await pipeline.process_text(
        text,
        think_effort=think_effort,
        session_id=session_id_from_request(request),
        search_depth=search_depth,
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


@app.get("/ui_config")
async def ui_config(request: Request):
    """Defaults for the web UI (reasoning effort, search depth, audio)."""
    pipeline: ServerPipeline = request.app.state.pipeline
    profile = pipeline.llm.model.profile
    default_effort = parse_ui_think_effort(profile.think_effort) or "xhigh"
    raw_effort = (profile.think_effort or "").strip().lower()
    supports_levels = raw_effort not in {"on", "off", "none"}
    return {
        "think": bool(profile.think) and supports_levels,
        "reasoning_effort": default_effort,
        "reasoning_effort_options": list(UI_THINK_EFFORTS),
        "search_depth": DEFAULT_SEARCH_DEPTH,
        "search_depth_options": list(SEARCH_DEPTHS),
        "max_searches_per_answer": max(1, int(profile.max_turns)),
        "audio_enabled": bool(pipeline.config.audio_enabled),
    }


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

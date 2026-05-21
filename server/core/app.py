import logging
import sys
from contextlib import asynccontextmanager
from urllib.parse import quote

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from server.utils.config import load_config
from server.core.pipeline import ServerPipeline
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
    logger.info("Voice Assistant Server готов!")

    yield

    logger.info("Завершение работы...")
    await pipeline.shutdown()


app = FastAPI(title="Voice Assistant Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Recognized-Text", "LLM-Response", "Sample-Rate", "Channels", "Sample-Width"],
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
    wav_bytes = await request.body()

    if not wav_bytes:
        return JSONResponse({"error": "Пустое тело запроса"}, status_code=400)

    recognized, answer = await pipeline.process_audio(wav_bytes)

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
    wav_bytes = await request.body()

    if not wav_bytes:
        return JSONResponse({"error": "Пустое тело запроса"}, status_code=400)

    text = await pipeline.stt.transcribe_bytes(wav_bytes)
    if not text:
        return JSONResponse({"error": "Речь не распознана"}, status_code=422)

    return JSONResponse({"text": text})


@app.post("/process_text")
async def process_text(request: Request):
    """
    Принимает текст JSON, возвращает стрим PCM-чанков.

    Request body: {"text": "вопрос пользователя"}
    """
    pipeline: ServerPipeline = request.app.state.pipeline
    data = await request.json()
    text = data.get("text", "").strip()

    if not text:
        return JSONResponse({"error": "Пустой текст"}, status_code=400)

    answer = await pipeline.process_text(text)

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


@app.post("/process_text_test")
async def process_text_test(request: Request):
    """
    Принимает текст JSON, возвращает ответ LLM (без TTS).
    Специально для скриптов тестирования.

    Request body: {"text": "вопрос пользователя"}
    """
    pipeline: ServerPipeline = request.app.state.pipeline
    data = await request.json()
    text = data.get("text", "").strip()

    if not text:
        return JSONResponse({"error": "Пустой текст"}, status_code=400)

    answer = await pipeline.process_text(text)

    return JSONResponse({"answer": answer})


@app.get("/health")
async def health():
    """Проверка готовности сервера."""
    return {"status": "ready"}


# Веб-интерфейс: http://localhost:8000/ui/
app.mount("/ui", StaticFiles(directory="server/static", html=True), name="ui")


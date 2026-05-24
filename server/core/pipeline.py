import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import Request

from server.utils.config import AppConfig
from server.llm.manager import LLMManager
from server.stt.provider import STTProvider
from server.utils.tracing import (
    OI_INPUT_VALUE,
    OI_SPAN_KIND,
    OISpanKind,
    get_tracer,
    set_span_error,
    set_span_ok,
)
from server.tts.manager import TTSManager

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class ServerPipeline:
    """
    Серверный пайплайн обработки: STT → LLM → TTS.

    Заменяет core/assistant.py — без sounddevice, keyboard, threading.
    Принимает данные по HTTP, возвращает PCM-чанки.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.stt = STTProvider(config.stt)
        self.llm = LLMManager(config)
        self.tts = TTSManager(config.tts)
        self._last_request_has_graph = False
        self._last_answer = ""

    async def startup(self) -> None:
        """Прогрев всех моделей при старте сервера."""
        await self.llm.warmup()
        logger.info("ServerPipeline готов")

    async def shutdown(self) -> None:
        """Освобождение ресурсов при остановке сервера."""
        await self.llm.unload()
        self.tts.unload()
        self.llm.tools.graph_filter.close()
        logger.info("Ресурсы освобождены")

    async def process_audio(self, wav_bytes: bytes) -> Tuple[Optional[str], str]:
        """
        Обрабатывает аудио: STT → LLM.

        Args:
            wav_bytes: WAV-файл в байтах.

        Returns:
            Tuple[recognized_text, llm_answer]
        """
        with tracer.start_as_current_span("process_audio") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.CHAIN)
            span.set_attribute(
                OI_INPUT_VALUE, f"Audio data, size: {len(wav_bytes)} bytes"
            )

            text = await self.stt.transcribe_bytes(wav_bytes)
            if not text:
                set_span_error(span, "Речь не распознана")
                return None, ""

            logger.info(f"STT: {text}")

            try:
                sq = self.llm.tools.graph_qa.successful_queries
                last_before = sq[-1] if sq else None
                answer = await asyncio.wait_for(
                    self.llm.generate_response(user_text=text),
                    timeout=self.config.server.llm_timeout,
                )
                last_after = sq[-1] if sq else None
                self._last_request_has_graph = (
                    last_after is not None and last_after is not last_before
                )
                self._last_answer = answer
                display_answer = answer.split("GRAPH_NODES:")[0].strip()
                logger.info(f"LLM: {display_answer}")
                set_span_ok(span, display_answer)
                return text, display_answer
            except Exception as e:
                self._last_request_has_graph = False
                set_span_error(span, str(e))
                raise

    async def process_text(self, text: str) -> str:
        """
        Обрабатывает текстовый ввод: LLM (без STT).

        Args:
            text: Текст от пользователя.

        Returns:
            Ответ LLM.
        """
        with tracer.start_as_current_span("process_text") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.CHAIN)
            span.set_attribute(OI_INPUT_VALUE, text)
            logger.info(f"Текст: {text}")

            try:
                sq = self.llm.tools.graph_qa.successful_queries
                last_before = sq[-1] if sq else None
                answer = await asyncio.wait_for(
                    self.llm.generate_response(user_text=text),
                    timeout=self.config.server.llm_timeout,
                )
                last_after = sq[-1] if sq else None
                self._last_request_has_graph = (
                    last_after is not None and last_after is not last_before
                )
                self._last_answer = answer
                result = answer.split("GRAPH_NODES:")
                display_answer = result[0].strip()
                nodes = None
                if len(result) > 1:
                    nodes = result[1].strip()
                logger.info(f"LLM: {display_answer}\n{nodes}")
                set_span_ok(span, display_answer)
                return display_answer
            except Exception as e:
                self._last_request_has_graph = False
                set_span_error(span, str(e))
                raise

    async def synthesize(
        self, text: str, request: Optional[Request] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        TTS: текст → стрим PCM-чанков (int16, 24kHz, mono).

        Args:
            text: Текст для синтеза.
            request: Объект запроса для проверки отключения клиента (barge in).

        Yields:
            bytes: PCM-чанки.
        """
        with tracer.start_as_current_span("synthesize") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.TOOL)
            span.set_attribute(OI_INPUT_VALUE, text)
            try:
                async for chunk in self.tts.synthesize_stream(text):
                    if request and await request.is_disconnected():
                        logger.info("Клиент отключился (barge in). Остановка TTS.")
                        break
                    yield chunk
                set_span_ok(span, "Audio stream completed")
            except Exception as e:
                set_span_error(span, str(e))
                logger.warning(f"TTS стрим прерван из-за ошибки: {e}")

    @property
    def has_graph(self) -> bool:
        """Показывает, был ли сгенерирован граф в последнем ответе."""
        return self._last_request_has_graph

    async def get_graph_data(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Извлекает данные графа для визуализации по последнему успешному Cypher-запросу.

        Возвращает пустой граф, если текущий запрос не породил нового
        успешного Cypher (чтобы не показывать устаревший граф от прошлого вопроса).

        Returns:
            Словарь {nodes: [...], edges: [...]}.
        """
        if not self._last_request_has_graph:
            logger.info(
                "Текущий запрос не породил нового Cypher — граф не отображается"
            )
            return {"nodes": [], "edges": []}

        graph_qa = self.llm.tools.graph_qa
        last_question, last_cypher = graph_qa.successful_queries[-1]
        logger.info(f"Визуализация графа по запросу: {last_cypher}")

        graph_data = await self.llm.tools.graph_filter.build_viz_graph(
            assistant_answer=self._last_answer,
            original_cypher=last_cypher,
        )
        return graph_data

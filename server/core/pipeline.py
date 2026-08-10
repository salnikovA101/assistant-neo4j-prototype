import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import Request

from server.utils.config import AppConfig
from server.llm.manager import LLMManager
from server.utils.tracing import (
    OI_INPUT_VALUE,
    OI_SPAN_KIND,
    OISpanKind,
    get_tracer,
    set_span_error,
    set_span_ok,
)

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class ServerPipeline:
    """
    Серверный пайплайн обработки: (STT) → LLM → (TTS).

    При audio_enabled=false работает только как текстовый бэкенд (LLM + Neo4j).
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.llm = LLMManager(config)
        self.stt = None
        self.tts = None

        if config.audio_enabled:
            from server.stt.provider import STTProvider
            from server.tts.manager import TTSManager

            self.stt = STTProvider(config.stt)
            self.tts = TTSManager(config.tts)
        else:
            logger.info("Аудио отключено (audio_enabled=false): STT/TTS не загружаются")

        self._last_request_has_graph = False
        self._last_new_queries = []
        self._last_answer = ""

    async def startup(self) -> None:
        """Прогрев всех моделей при старте сервера."""
        from server.core.db import init_driver

        neo4j_config = self.config.neo4j
        init_driver(neo4j_config.uri, neo4j_config.user, neo4j_config.password)
        logger.info("Neo4j driver инициализирован")

        await self.llm.warmup()
        logger.info("ServerPipeline готов")

    async def shutdown(self) -> None:
        """Освобождение ресурсов при остановке сервера."""
        from server.core.db import close_driver

        await self.llm.unload()
        if self.tts is not None:
            self.tts.unload()
        self.llm.tools.graph_filter.close()
        await close_driver()
        logger.info("Ресурсы освобождены")

    def clear_history(self) -> None:
        """Сброс истории диалога и контекста."""
        self.llm.clear_history()
        self._last_request_has_graph = False
        self._last_new_queries = []
        self._last_answer = ""
        logger.info("История разговора и контекст сброшены")

    async def process_audio(self, wav_bytes: bytes) -> Tuple[Optional[str], str]:
        """
        Обрабатывает аудио: STT → LLM.

        Args:
            wav_bytes: WAV-файл в байтах.

        Returns:
            Tuple[recognized_text, llm_answer]
        """
        if self.stt is None:
            raise RuntimeError("STT отключён (audio_enabled=false)")

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
                sq_len_before = len(sq)
                answer = await asyncio.wait_for(
                    self.llm.generate_response(user_text=text),
                    timeout=self.config.server.llm_timeout,
                )
                sq_len_after = len(sq)
                self._last_new_queries = list(sq)[sq_len_before:sq_len_after]
                self._last_request_has_graph = len(self._last_new_queries) > 0
                self._last_answer = answer
                display_answer = answer.strip()
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
                sq_len_before = len(sq)
                answer = await asyncio.wait_for(
                    self.llm.generate_response(user_text=text),
                    timeout=self.config.server.llm_timeout,
                )
                sq_len_after = len(sq)
                self._last_new_queries = list(sq)[sq_len_before:sq_len_after]
                self._last_request_has_graph = len(self._last_new_queries) > 0
                self._last_answer = answer
                display_answer = answer.strip()
                logger.info(f"LLM: {display_answer}")
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
        if self.tts is None:
            raise RuntimeError("TTS отключён (audio_enabled=false)")

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
        if not getattr(self, "_last_new_queries", None):
            logger.info(
                "Текущий запрос не породил нового Cypher — граф не отображается"
            )
            return {"nodes": [], "edges": []}

        merged_data = {"nodes": [], "edges": []}
        seen_nodes = set()
        seen_edges = set()

        for question, cypher in self._last_new_queries:
            logger.info(f"Визуализация графа по запросу: {cypher}")
            graph_data = await self.llm.tools.graph_filter.build_viz_graph(
                assistant_answer=self._last_answer,
                original_cypher=cypher,
            )
            
            for node in graph_data.get("nodes", []):
                if node["id"] not in seen_nodes:
                    seen_nodes.add(node["id"])
                    merged_data["nodes"].append(node)
                    
            for edge in graph_data.get("edges", []):
                if edge["id"] not in seen_edges:
                    seen_edges.add(edge["id"])
                    merged_data["edges"].append(edge)

        return merged_data

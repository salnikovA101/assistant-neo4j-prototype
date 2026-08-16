import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from fastapi import Request

from server.utils.config import AppConfig
from server.llm.manager import LLMManager
from server.llm.stream_events import StreamEvent
from server.core.graph_runs import (
    current_graph_collector,
    graph_run_store,
    new_graph_collector,
    reset_graph_collector,
)
from server.tools.source_registry import filter_chains_by_source_files
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

    async def process_text(
        self, text: str, think_effort: Optional[str] = None
    ) -> str:
        """
        Обрабатывает текстовый ввод: LLM (без STT).

        Args:
            text: Текст от пользователя.
            think_effort: Optional per-request reasoning_effort override.

        Returns:
            Ответ LLM.
        """
        with tracer.start_as_current_span("process_text") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.CHAIN)
            span.set_attribute(OI_INPUT_VALUE, text)
            if think_effort:
                span.set_attribute("think_effort", think_effort)
            logger.info("Текст: %s effort=%s", text, think_effort or "-")

            try:
                sq = self.llm.tools.graph_qa.successful_queries
                sq_len_before = len(sq)
                answer = await asyncio.wait_for(
                    self.llm.generate_response(
                        user_text=text, think_effort=think_effort
                    ),
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

    async def process_text_stream(
        self,
        text: str,
        request: Optional[Request] = None,
        think_effort: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream LLM events (thinking / tools / content / done) for text input.
        """
        with tracer.start_as_current_span("process_text_stream") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.CHAIN)
            span.set_attribute(OI_INPUT_VALUE, text)
            if think_effort:
                span.set_attribute("think_effort", think_effort)
            logger.info(
                "Текст (stream): %s effort=%s",
                text,
                think_effort or "-",
            )

            sq = self.llm.tools.graph_qa.successful_queries
            sq_len_before = len(sq)
            final_content = ""
            collector_token = new_graph_collector()

            try:
                async for event in self.llm.generate_response_stream(
                    user_text=text, think_effort=think_effort
                ):
                    if request and await request.is_disconnected():
                        logger.info("Клиент отключился — остановка LLM stream")
                        break

                    if event.type == "done":
                        final_content = event.data.get("final_content") or final_content
                        sq_len_after = len(sq)
                        self._last_new_queries = list(sq)[sq_len_before:sq_len_after]
                        cited = event.data.get("cited_source_files") or []
                        graph_chains = filter_chains_by_source_files(
                            current_graph_collector() or [],
                            cited,
                        )
                        graph_run_id = (
                            graph_run_store.put(graph_chains) if graph_chains else ""
                        )
                        graph_chain_count = len(graph_chains) if graph_run_id else 0
                        self._last_request_has_graph = (
                            bool(graph_run_id) or len(self._last_new_queries) > 0
                        )
                        self._last_answer = final_content
                        event = StreamEvent(
                            "done",
                            {
                                "final_content": final_content,
                                "has_graph": self._last_request_has_graph,
                                "graph_run_id": graph_run_id,
                                "graph_chain_count": graph_chain_count,
                            },
                        )
                        set_span_ok(span, final_content)
                        logger.info(f"LLM (stream): {final_content}")
                    elif event.type == "error":
                        self._last_request_has_graph = False
                        set_span_error(span, event.data.get("message", "error"))

                    yield event
            except Exception as e:
                self._last_request_has_graph = False
                set_span_error(span, str(e))
                yield StreamEvent("error", {"message": str(e)})
            finally:
                reset_graph_collector(collector_token)

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


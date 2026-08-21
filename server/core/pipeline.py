import asyncio
import logging
from typing import AsyncGenerator, Optional, Tuple

from fastapi import Request

from server.utils.config import AppConfig
from server.llm.manager import LLMManager
from server.llm.stream_events import StreamEvent
from server.llm.base import public_llm_error_message
from server.core.graph_runs import (
    current_graph_collector,
    graph_run_store,
    new_graph_collector,
    reset_graph_collector,
)
from server.core.sessions import bind_conversation, session_store
from server.tools.source_registry import filter_chains_by_source_files

logger = logging.getLogger(__name__)


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

    def clear_history(self, session_id: Optional[str] = None) -> None:
        """Сброс истории и source-реестра одной вкладки (или дефолтного fallback)."""
        if session_id:
            session_store.clear(session_id)
            logger.info("История сессии %s сброшена", session_id)
            return
        self.llm.clear_history()
        logger.info("История разговора и контекст сброшены")

    async def process_audio(
        self, wav_bytes: bytes, session_id: Optional[str] = None
    ) -> Tuple[Optional[str], str]:
        """
        Обрабатывает аудио: STT → LLM.

        Args:
            wav_bytes: WAV-файл в байтах.
            session_id: Per-tab conversation id (X-Session-Id).

        Returns:
            Tuple[recognized_text, llm_answer]
        """
        if self.stt is None:
            raise RuntimeError("STT отключён (audio_enabled=false)")

        async with bind_conversation(session_id, self.config.llm.history_len):
            text = await self.stt.transcribe_bytes(wav_bytes)
            if not text:
                return None, ""

            logger.info(f"STT: {text}")

            answer = await asyncio.wait_for(
                self.llm.generate_response(user_text=text),
                timeout=self.config.server.llm_timeout,
            )
            display_answer = answer.strip()
            logger.info(f"LLM: {display_answer}")
            return text, display_answer

    async def process_text(
        self,
        text: str,
        think_effort: Optional[str] = None,
        session_id: Optional[str] = None,
        search_depth: Optional[str] = None,
        api_key: Optional[str] = None,
        profile_name: Optional[str] = None,
    ) -> str:
        """
        Обрабатывает текстовый ввод: LLM (без STT).

        Args:
            text: Текст от пользователя.
            think_effort: Optional per-request reasoning_effort override.
            session_id: Per-tab conversation id (X-Session-Id).
            search_depth: UI-selected ask_subgraph depth (low|medium|high).
            profile_name: UI-selected LLM profile id (ollama / ollama_gptoss / qwen_cloud).

        Returns:
            Ответ LLM.
        """
        async with bind_conversation(session_id, self.config.llm.history_len):
            logger.info(
                "Текст: %s profile=%s effort=%s depth=%s",
                text,
                profile_name or "-",
                think_effort or "-",
                search_depth or "-",
            )

            answer = await asyncio.wait_for(
                self.llm.generate_response(
                    user_text=text,
                    think_effort=think_effort,
                    search_depth=search_depth,
                    api_key=api_key,
                    profile_name=profile_name,
                ),
                timeout=self.config.server.llm_timeout,
            )
            display_answer = answer.strip()
            logger.info(f"LLM: {display_answer}")
            return display_answer

    async def process_text_stream(
        self,
        text: str,
        request: Optional[Request] = None,
        think_effort: Optional[str] = None,
        session_id: Optional[str] = None,
        search_depth: Optional[str] = None,
        api_key: Optional[str] = None,
        profile_name: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """
        Stream LLM events (thinking / tools / content / done) for text input.
        """
        async with bind_conversation(session_id, self.config.llm.history_len):
            logger.info(
                "Текст (stream): %s profile=%s effort=%s depth=%s",
                text,
                profile_name or "-",
                think_effort or "-",
                search_depth or "-",
            )

            final_content = ""
            collector_token = new_graph_collector()

            try:
                async for event in self.llm.generate_response_stream(
                    user_text=text,
                    think_effort=think_effort,
                    search_depth=search_depth,
                    api_key=api_key,
                    profile_name=profile_name,
                ):
                    if request and await request.is_disconnected():
                        logger.info("Клиент отключился — остановка LLM stream")
                        break

                    if event.type == "done":
                        final_content = (
                            event.data.get("final_content") or final_content
                        )
                        cited = event.data.get("cited_source_files") or []
                        graph_chains = filter_chains_by_source_files(
                            current_graph_collector() or [],
                            cited,
                        )
                        graph_run_id = (
                            graph_run_store.put(graph_chains) if graph_chains else ""
                        )
                        graph_chain_count = (
                            len(graph_chains) if graph_run_id else 0
                        )
                        event = StreamEvent(
                            "done",
                            {
                                "final_content": final_content,
                                "graph_run_id": graph_run_id,
                                "graph_chain_count": graph_chain_count,
                            },
                        )
                        logger.info(f"LLM (stream): {final_content}")

                    yield event
            except Exception as e:
                yield StreamEvent("error", {"message": public_llm_error_message(e)})
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

        try:
            async for chunk in self.tts.synthesize_stream(text):
                if request and await request.is_disconnected():
                    logger.info("Клиент отключился (barge in). Остановка TTS.")
                    break
                yield chunk
        except Exception as e:
            logger.warning(f"TTS стрим прерван из-за ошибки: {e}")


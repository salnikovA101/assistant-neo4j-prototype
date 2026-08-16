import logging
from typing import AsyncIterator

from server.utils.config import AppConfig
from server.utils.constants import LLMProviderType
from server.llm.base import BaseLLMProvider
from server.llm.history_manager import HistoryManager
from server.llm.prompt_loader import PromptLoader
from server.llm.providers.openai_provider import OpenAIProvider
from server.llm.stream_events import StreamEvent
from server.tools.registry import Tools
from server.tools.source_registry import extract_cited_source_files, render_citations
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

_PROVIDER_MAP: dict[LLMProviderType, type[BaseLLMProvider]] = {
    LLMProviderType.OPENAI: OpenAIProvider,
}


class LLMManager:
    """
    Менеджер для работы с LLM провайдером.

    Инициализирует нужный провайдер по полю `provider` из конфига профиля
    (LLMProviderType). Делегирует вызовы generate_response, unload и warmup
    активному провайдеру.
    """

    def __init__(self, config: AppConfig) -> None:
        self.config = config.llm
        self.prompt_manager = PromptLoader(
            self.config.prompt_folder,
            config.tts.mode,
            audio_enabled=config.audio_enabled,
        )
        self.history_manager = HistoryManager(self.config.history_len)
        self.tools = Tools(config)
        self.model: BaseLLMProvider = self._load(self.config.current_profile)

    async def generate_response(
        self, user_text: str, think_effort: str | None = None
    ) -> str:
        with tracer.start_as_current_span("generate_response") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.CHAIN)
            span.set_attribute(OI_INPUT_VALUE, user_text)
            span.set_attribute("user_text", user_text[:200])

            try:
                text = ""
                async for event in self.generate_response_stream(
                    user_text, think_effort=think_effort
                ):
                    if event.type == "done":
                        text = event.data.get("final_content") or text
                    elif event.type == "error":
                        msg = event.data.get("message", "unknown")
                        set_span_error(span, msg)
                        return f"Ошибка: {msg}"
                set_span_ok(span, text)
                return text
            except Exception as e:
                set_span_error(span, str(e))
                raise

    async def generate_response_stream(
        self, user_text: str, think_effort: str | None = None
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream assistant events. History is updated only after a successful done
        with non-empty final_content (not on abort/error).
        """
        with tracer.start_as_current_span("generate_response_stream") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.CHAIN)
            span.set_attribute(OI_INPUT_VALUE, user_text)
            span.set_attribute("user_text", user_text[:200])
            if think_effort:
                span.set_attribute("think_effort", think_effort)

            prompt = self.prompt_manager.get_system_prompt()
            history = self.history_manager.get_history()
            logger.debug(prompt)
            logger.debug(history)

            final_content = ""
            try:
                async for event in self.model.generate_response_stream(
                    user_text=user_text,
                    prompt=prompt,
                    history=history,
                    tools=self.tools.get_openai_tools(),
                    tool_map=self.tools.get_tool_map(),
                    think_effort=think_effort,
                ):
                    if event.type == "done":
                        final_content = (
                            event.data.get("final_content") or final_content
                        )
                        # History keeps raw (source:N) plus compact tool receipts.
                        # User/SSE get [n] + ### Источники; UNIT stays in live UI only.
                        self.history_manager.add_entry(
                            user_text,
                            final_content,
                            tool_messages=event.data.get("history_tool_messages")
                            or [],
                        )
                        cited = extract_cited_source_files(
                            final_content, self.tools.source_registry
                        )
                        display = render_citations(
                            final_content, self.tools.source_registry
                        )
                        event = StreamEvent(
                            "done",
                            {
                                "final_content": display,
                                "cited_source_files": cited,
                                "has_graph": event.data.get("has_graph", False),
                            },
                        )
                        set_span_ok(span, display)
                    elif event.type == "error":
                        set_span_error(span, event.data.get("message", "error"))
                    yield event
            except Exception as e:
                set_span_error(span, str(e))
                yield StreamEvent("error", {"message": str(e)})

    def clear_history(self) -> None:
        """Очищает историю диалога и контекст запросов к БД."""
        self.history_manager.clear_history()
        self.tools.clear_history()

    async def unload(self) -> None:
        await self.model.unload()

    async def warmup(self) -> None:
        await self.model.warmup()

    def _load(self, name: str) -> BaseLLMProvider:
        profile = getattr(self.config.profiles, name, None)
        if not profile:
            raise ValueError(f"Профиль LLM '{name}' не найден в конфигурации")

        cls = _PROVIDER_MAP.get(profile.provider)
        if cls is None:
            raise ValueError(
                f"Неизвестный провайдер '{profile.provider}'. "
                f"Доступные: {[e.value for e in LLMProviderType]}"
            )

        logger.info(f"LLM провайдер: {cls.__name__} (профиль: '{name}')")
        return cls(profile)

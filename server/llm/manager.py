import logging
from typing import AsyncIterator

from server.utils.config import AppConfig, resolve_request_profile
from server.utils.constants import LLMProviderType
from server.core.sessions import current_session
from server.core.turn_state import bind_turn, searches_state
from server.llm.base import BaseLLMProvider, public_llm_error_message
from server.llm.history_manager import HistoryManager
from server.llm.prompt_loader import PromptLoader
from server.llm.providers.openai_provider import OpenAIProvider
from server.llm.stream_events import StreamEvent
from server.tools.registry import Tools
from server.tools.source_registry import (
    SourceRegistry,
    citation_stats,
    extract_cited_source_files,
    render_citations,
)
logger = logging.getLogger(__name__)

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
        self._providers: dict[str, BaseLLMProvider] = {}
        self.model: BaseLLMProvider = self.provider_for(self.config.current_profile)

    def provider_for(self, name: str | None = None) -> BaseLLMProvider:
        """Return a cached provider for a UI-selectable profile (or the default)."""
        key = resolve_request_profile(self.config, name)
        cached = self._providers.get(key)
        if cached is not None:
            return cached
        provider = self._load(key)
        self._providers[key] = provider
        return provider

    def _active_history(self) -> HistoryManager:
        sess = current_session()
        return sess.history if sess else self.history_manager

    def _active_sources(self) -> SourceRegistry:
        sess = current_session()
        return sess.sources if sess else self.tools.source_registry

    async def generate_response(
        self,
        user_text: str,
        think_effort: str | None = None,
        search_depth: str | None = None,
        api_key: str | None = None,
        profile_name: str | None = None,
    ) -> str:
        text = ""
        async for event in self.generate_response_stream(
            user_text,
            think_effort=think_effort,
            search_depth=search_depth,
            api_key=api_key,
            profile_name=profile_name,
        ):
            if event.type == "done":
                text = event.data.get("final_content") or text
            elif event.type == "error":
                msg = event.data.get("message", "unknown")
                return f"Ошибка: {msg}"
        return text

    async def generate_response_stream(
        self,
        user_text: str,
        think_effort: str | None = None,
        search_depth: str | None = None,
        api_key: str | None = None,
        profile_name: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream assistant events. History is updated only after a successful done
        with non-empty final_content (not on abort/error).
        """
        prompt = self.prompt_manager.get_system_prompt()
        history_manager = self._active_history()
        sources = self._active_sources()
        history = history_manager.get_history()
        provider = self.provider_for(profile_name)
        logger.debug(prompt)
        logger.debug(history)

        final_content = ""
        max_searches = max(1, int(provider.profile.max_turns))
        try:
            with bind_turn(search_depth, max_searches=max_searches):
                async for event in provider.generate_response_stream(
                    user_text=user_text,
                    prompt=prompt,
                    history=history,
                    tools=self.tools.get_openai_tools(),
                    tool_map=self.tools.get_tool_map(),
                    think_effort=think_effort,
                    api_key=api_key,
                ):
                    if event.type == "done":
                        final_content = (
                            event.data.get("final_content") or final_content
                        )
                        # History keeps raw (source:N) plus compact tool receipts.
                        # User/SSE get [n] + ### Источники; UNIT stays in live UI only.
                        history_manager.add_entry(
                            user_text,
                            final_content,
                            tool_messages=event.data.get("history_tool_messages")
                            or [],
                        )
                        cited = extract_cited_source_files(final_content, sources)
                        display = render_citations(final_content, sources)
                        self._log_turn_quality(final_content, sources)
                        event = StreamEvent(
                            "done",
                            {
                                "final_content": display,
                                "cited_source_files": cited,
                            },
                        )
                    yield event
        except Exception as e:
            yield StreamEvent("error", {"message": public_llm_error_message(e)})

    def _log_turn_quality(self, answer: str, sources: SourceRegistry) -> None:
        """Groundedness signal per turn: citations, invented ids, searches spent."""
        stats = citation_stats(answer, sources)
        used, limit = searches_state()
        logger.info(
            "Turn quality: searches=%s/%s citations=%s unknown=%s uncited_lines=%s",
            used,
            limit,
            stats.known,
            stats.unknown,
            stats.uncited_claim_lines,
        )
        if stats.unknown:
            logger.warning(
                "Ответ содержит %s неизвестных источников — помечены %s",
                stats.unknown,
                "[?]",
            )

    def clear_history(self) -> None:
        """Очищает историю диалога и контекст запросов к БД."""
        self.history_manager.clear_history()
        self.tools.clear_history()

    async def unload(self) -> None:
        seen: list[BaseLLMProvider] = []
        for provider in self._providers.values():
            if provider in seen:
                continue
            seen.append(provider)
            await provider.unload()

    async def warmup(self) -> None:
        await self.provider_for(self.config.current_profile).warmup()

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

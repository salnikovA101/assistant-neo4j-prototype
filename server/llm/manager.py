import logging

from server.utils.config import AppConfig
from server.utils.constants import LLMProviderType
from server.llm.base import BaseLLMProvider
from server.llm.history_manager import HistoryManager
from server.llm.prompt_loader import PromptLoader
from server.llm.providers.openai_provider import OpenAIProvider
from server.tools.registry import Tools
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
        self.prompt_manager = PromptLoader(self.config.prompt_folder, config.tts.mode)
        self.history_manager = HistoryManager(self.config.history_len)
        self.tools = Tools(config)
        self.model: BaseLLMProvider = self._load(self.config.current_profile)

    async def generate_response(self, user_text: str) -> str:
        with tracer.start_as_current_span("generate_response") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.CHAIN)
            span.set_attribute(OI_INPUT_VALUE, user_text)
            span.set_attribute("user_text", user_text[:200])

            prompt = self.prompt_manager.get_system_prompt()
            history = self.history_manager.get_history()
            logger.debug(prompt)
            logger.debug(history)

            try:
                text = await self.model.generate_response(
                    user_text=user_text,
                    prompt=prompt,
                    history=history,
                    tools=self.tools.get_openai_tools(),
                    tool_map=self.tools.get_tool_map(),
                )
                self.history_manager.add_entry(user_text, text)
                set_span_ok(span, text)
                return text
            except Exception as e:
                set_span_error(span, str(e))
                raise

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

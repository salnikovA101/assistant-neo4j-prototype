import logging

from server.llm.base import BaseLLMProvider

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """
    Провайдер для OpenAI-совместимых API (OpenAI, Gemini, OpenRouter и др.).

    Управление жизненным циклом модели (load/unload) не поддерживается —
    модель живёт на стороне провайдера.
    """

    async def unload(self) -> None:
        logger.debug("[OpenAIProvider] выгрузка не нужна")

    async def warmup(self) -> None:
        logger.debug("[OpenAIProvider] прогрев не нужен")

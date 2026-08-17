import logging
from collections.abc import Callable
from typing import Any

from server.core.turn_state import search_depth
from server.tools.source_registry import SourceRegistry
from server.tools.subgraph_search import SubgraphSearchAgent
from server.utils.config import AppConfig
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


class Tools:
    """
    Класс-регистратор инструментов (Tools/Functions) для LLM-ассистента.
    """

    def __init__(self, config: AppConfig):
        """
        Инициализирует реестр инструментов.

        Args:
            config (AppConfig): Полный объект конфигурации приложения.
        """
        _ = config
        self.source_registry = SourceRegistry()
        self.subgraph_search = SubgraphSearchAgent(
            source_registry=self.source_registry,
        )

    async def ask_subgraph(
        self,
        subquestions: list[str],
        **ignored: Any,
    ) -> str:
        """
        Search the English knowledge graph (articles, patents, regulations).

        Args:
            subquestions: 1–6 English declarative statements, one per aspect.
            ignored: tolerated legacy/hallucinated arguments (e.g. `effort`);
                search depth comes from the UI, not from the model.
        """
        with tracer.start_as_current_span("ask_subgraph") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.TOOL)
            span.set_attribute(
                OI_INPUT_VALUE,
                " | ".join(str(s) for s in (subquestions or [])[:6])[:500],
            )
            span.set_attribute("search_depth", search_depth())
            sqs = [str(s).strip() for s in (subquestions or []) if str(s).strip()]
            if ignored:
                logger.info("ask_subgraph: игнорируем аргументы модели %s", list(ignored))
            logger.info(
                "Вызов ask_subgraph depth=%s n_sq=%s",
                search_depth(),
                len(sqs),
            )
            for i, text in enumerate(sqs, 1):
                logger.info("  sq%s: %s", i, text)

            try:
                result = await self.subgraph_search.query(subquestions=subquestions)
                set_span_ok(span, result)
                return result
            except Exception as e:
                set_span_error(span, str(e))
                raise

    def clear_history(self) -> None:
        """Очищает сессионный source-реестр."""
        self.source_registry.clear()

    def get_tools_list(self) -> list[Callable]:
        """
        Возвращает список всех доступных функций-инструментов.
        """
        return [self.ask_subgraph]

    def get_tool_map(self) -> dict[str, Callable]:
        """
        Создает словарь соответствия имен функций их объектам.
        """
        return {func.__name__: func for func in self.get_tools_list()}

    def get_openai_tools(self) -> list[dict[str, Any]]:
        """
        Возвращает список инструментов в формате JSON Schema для OpenAI SDK.
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "ask_subgraph",
                    "description": (
                        "Search the English knowledge graph for food technology "
                        "(starter cultures, freshness indicators, smart packaging). "
                        "Call it when the question names a product, substance, "
                        "culture, process or goal. Returns evidence UNITs: tours of "
                        "cards, each card a triple plus its verbatim quote and "
                        "(source:N). Search depth is set in the UI, not here."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subquestions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 6,
                                "description": (
                                    "1–6 English declarative statements, no '?' and "
                                    "no Russian. Each one runs a separate search "
                                    "over quotes, so each must cover a different "
                                    "aspect of the question — paraphrases return "
                                    "the same evidence. "
                                    "GOOD: 'Lactic acid bacteria acidify milk during "
                                    "cottage cheese production.' "
                                    "BAD: 'What starter cultures are used?'"
                                ),
                            },
                        },
                        "required": ["subquestions"],
                    },
                },
            },
        ]

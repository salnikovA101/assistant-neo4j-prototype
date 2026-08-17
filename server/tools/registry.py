import logging
from collections.abc import Callable
from typing import Any

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
        effort: str = "medium",
    ) -> str:
        """
        Search the English knowledge graph (articles, patents, regulations).

        At most TWO calls; high+high is forbidden, any other pair is allowed.
        Call 1: 1–6 English statements from the user question only.
        Call 2: missing field or follow-up from returned chains, not a
        repeat of call 1.

        Returns UNIT blocks. Cite (source:N); no [n] / ### Источники.

        Args:
            subquestions: 1–6 English statements. Call 1: only classes
                from the user question. Call 2 may use names from
                returned chains.
            effort: low (mine 10 / emit 5), medium (15 / 10), high (20 / 15).
        """
        with tracer.start_as_current_span("ask_subgraph") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.TOOL)
            span.set_attribute(
                OI_INPUT_VALUE,
                " | ".join(str(s) for s in (subquestions or [])[:6])[:500],
            )
            span.set_attribute("effort", effort or "medium")
            sqs = [str(s).strip() for s in (subquestions or []) if str(s).strip()]
            logger.info(
                "Вызов ask_subgraph effort=%s n_sq=%s",
                effort,
                len(sqs),
            )
            for i, text in enumerate(sqs, 1):
                logger.info("  sq%s: %s", i, text)

            try:
                result = await self.subgraph_search.query(
                    subquestions=subquestions,
                    effort=effort,
                )
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
                        "Call when the question has a product and/or goal. If neither "
                        "is named, do not call — ask one clarifying question. "
                        "BUDGET: two calls max; high+high is forbidden, any other "
                        "pair is allowed. Prior ask_subgraph calls in history are "
                        "the previous turn and do not spend this budget. "
                        "Call 1: 1–6 English declarative statements; orthogonal "
                        "aspects from the question, not paraphrases and not empty "
                        "axes; only names the user said; no Russian. GOOD lines "
                        "are syntax, not default entities. Follow-up that points "
                        "at the previous assistant answer (expand, close GAPS, "
                        "add a field): names from that answer and its GAPS axes "
                        "are in scope for call 1. Do not copy subquestion strings "
                        "from prior ask_subgraph calls in history; write new "
                        "statements for missing fields. "
                        "Call 2: 1–3 statements for a missing field (dose / matrix / "
                        "regulation) or names from returned chains this turn; if "
                        "call 1 was empty, repeat the same question classes "
                        "without new names. "
                        "RETURNS: evidence blocks. One block = one system = one "
                        "table row. Cite (source:N). Do not write [n], PDF names, "
                        "or ### Источники. Do not name block labels in the answer."
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
                                    "1–6 English declarative statements; orthogonal "
                                    "aspects from the question, not paraphrases. "
                                    "No '?', no Russian. Call 1: only names and "
                                    "classes the user said (substance, gas, number, "
                                    "strain, matrix, plant, subclass — not only a "
                                    "dye). Follow-up pointing at the previous "
                                    "assistant answer: those names are in scope "
                                    "for call 1. Call 2 may use names from returned "
                                    "chains this turn. "
                                    "GOOD: 'Lactic acid bacteria are used as starter cultures "
                                    "for cottage cheese production.' "
                                    "GOOD: 'Freshness indicators change color "
                                    "in packaged food.' "
                                    "BAD: several restatements of one sentence. "
                                    "BAD (call 1): a name the user did not mention "
                                    "and that is not in the previous assistant answer. "
                                    "BAD: copying a previous ask_subgraph "
                                    "subquestion string from history. "
                                    "BAD (follow-up call 1): drop names from your "
                                    "previous answer and search only a class."
                                ),
                            },
                            "effort": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                                "description": (
                                    "Search budget: how many evidence blocks to mine/emit. "
                                    "low=mine 10 emit 5 (narrow fact), medium=15/10 "
                                    "(default), high=20/15 only if the user asked a list "
                                    "or comparison of many entities. Choose by question "
                                    "WIDTH. Max two calls; high+high is forbidden. "
                                    "Do not pick high only to mine more blocks."
                                ),
                            },
                        },
                        "required": ["subquestions"],
                    },
                },
            },
        ]

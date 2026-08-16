import logging
from typing import Any, Callable, Dict, List

from server.utils.config import AppConfig
from server.tools.graph_qa import GraphQA
from server.tools.source_registry import SourceRegistry
from server.tools.subgraph_search import SubgraphSearchAgent
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
        cypher_profile_name = config.llm.cypher_profile
        llm_profile = getattr(config.llm.profiles, cypher_profile_name, None)

        if not llm_profile:
            logger.warning(
                f"Профиль {cypher_profile_name} не найден. Используем профиль по умолчанию."
            )
            llm_profile = getattr(config.llm.profiles, config.llm.current_profile)

        tool_profile_name = config.llm.tool_profile or cypher_profile_name
        tool_llm_profile = getattr(config.llm.profiles, tool_profile_name, None)
        if not tool_llm_profile:
            logger.warning(
                f"Профиль {tool_profile_name} не найден. Используем cypher-профиль."
            )
            tool_llm_profile = llm_profile

        self.graph_qa = GraphQA(
            config.neo4j, llm_profile, config.llm.history_len, config.run_id, config.limit
        )
        self.source_registry = SourceRegistry()
        self.subgraph_search = SubgraphSearchAgent(
            tool_llm_profile,
            config.run_id,
            source_registry=self.source_registry,
        )

    async def ask_database(self, question: str) -> str:
        """
        Queries the knowledge graph database in natural language.
        Use for ANY question about entities, relationships, properties, or paths in the graph.
        Returns structured data including provenance: evidence (verbatim quote), source_file.

        Args:
            question (str): Natural language question to the database.
        """
        with tracer.start_as_current_span("ask_database") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.TOOL)
            span.set_attribute(OI_INPUT_VALUE, question)
            span.set_attribute("question", question)
            logger.info(f"Вызов инструмента: ask_database с вопросом '{question}'")

            try:
                result = await self.graph_qa.query(question)
                set_span_ok(span, result)
                return result
            except Exception as e:
                set_span_error(span, str(e))
                raise

    async def ask_subgraph(
        self,
        subquestions: list[str],
        effort: str = "medium",
    ) -> str:
        """
        Search the English knowledge graph (articles, patents, regulations).

        At most TWO calls: medium+medium or high+low (second is not high).
        Call 1: 1–6 English statements. Call 2: missing field (dose / matrix /
        regulation), not a repeat of call 1.

        Returns UNIT blocks. Cite (source:N); no [n] / ### Источники.

        Args:
            subquestions: 1–6 English statements. Only classes and user
                constraints; do not invent entity names.
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
        """Очищает историю успешных Cypher-запросов и сессионный source-реестр."""
        self.graph_qa.successful_queries.clear()
        self.source_registry.clear()

    def get_tools_list(self) -> List[Callable]:
        """
        Возвращает список всех доступных функций-инструментов.
        """
        return [self.ask_subgraph]

    def get_tool_map(self) -> Dict[str, Callable]:
        """
        Создает словарь соответствия имен функций их объектам.
        """
        return {func.__name__: func for func in self.get_tools_list()}

    def get_openai_tools(self) -> List[Dict[str, Any]]:
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
                        "BUDGET: two calls max — medium+medium or high+low (second "
                        "call is not high). "
                        "Call 1: 1–6 English declarative statements; only classes and "
                        "constraints from the user; no invented dye/strain/gas names; "
                        "no Russian. Orthogonal beats paraphrases. "
                        "Call 2: 1–3 statements for a missing field (dose / matrix / "
                        "regulation), not a repeat of call 1. "
                        "RETURNS: UNIT chains. Cite (source:N). Do not write [n], "
                        "PDF names, or ### Источники. One UNIT is one system; "
                        "do not mix facts across UNITs."
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
                                    "1–6 English declarative statements (product / process / "
                                    "matrix class / sensor or culture class). No '?', no "
                                    "checklists, no Russian. Only classes and user constraints; "
                                    "never invent entity names. "
                                    "GOOD: 'Lactic acid bacteria are used as starter cultures "
                                    "for cottage cheese production.' "
                                    "GOOD: 'Freshness indicators change color "
                                    "in packaged food.' "
                                    "BAD: 'Bromocresol green is embedded in agar to detect "
                                    "spoilage.' (named a dye the user did not)."
                                ),
                            },
                            "effort": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                                "description": (
                                    "Search budget: how many UNIT chains to mine/emit. "
                                    "low=mine 10 emit 5 (narrow fact), medium=15/10 "
                                    "(default), high=20/15 only if the user asked a list "
                                    "or comparison of many entities. Choose by question "
                                    "WIDTH. Max two calls: medium+medium or high+low "
                                    "(second call is not high)."
                                ),
                            },
                        },
                        "required": ["subquestions"],
                    },
                },
            },
        ]

import logging
from typing import Any, Callable, Dict, List

from server.utils.config import AppConfig
from server.tools.graph_filter import GraphFilterAgent
from server.tools.graph_qa import GraphQA
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
        self.subgraph_search = SubgraphSearchAgent(tool_llm_profile, config.run_id)
        self.graph_filter = GraphFilterAgent(
            config.neo4j, llm_profile, config.run_id, config.limit
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

    async def ask_subgraph(self, question: str) -> str:
        """
        Runs the V4 multi-stage graph retrieval pipeline for scientific questions.
        Prefer this for mechanisms, pathways, and multi-hop evidence synthesis.
        Each hop includes evidence and source_file. Cite user-facing claims as [n]
        mapped to ### Источники ([n] source_file); never cite [path N] indices.

        Args:
            question (str): User's natural language question (English).
        """
        with tracer.start_as_current_span("ask_subgraph") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.TOOL)
            span.set_attribute(OI_INPUT_VALUE, question)
            span.set_attribute("question", question)
            logger.info(f"Вызов инструмента: ask_subgraph с вопросом '{question}'")

            try:
                result = await self.subgraph_search.query(question)
                set_span_ok(span, result)
                return result
            except Exception as e:
                set_span_error(span, str(e))
                raise

    def clear_history(self) -> None:
        """Очищает историю успешных Cypher-запросов."""
        self.graph_qa.successful_queries.clear()

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
                        "Primary knowledge-graph retrieval. Internally: "
                        "(1) decomposes into declarative scientific statements, "
                        "(2) embeds and finds vector anchors, "
                        "(3) GDS projection + PPR filter, "
                        "(4) prize-coverage paths with evidence + source_file per edge. "
                        "Return format: [path N] blocks with hop evidence and source_file. "
                        "In the user answer cite as [1], [2], ... and list "
                        "### Источники with [n] source_file.pdf — never cite path indices. "
                        "CRITICAL — question phrasing: "
                        "Pass ONE natural English scientific question about entities/mechanisms "
                        "(how X relates to Y). The pipeline embeds declarative facts — "
                        "NOT field checklists or keyword lists. "
                        "GOOD: 'How do bromophenol blue cellulose indicators respond to acetic acid "
                        "in fruit packaging headspace?' "
                        "GOOD: 'Which anthocyanin films indicate fruit or vegetable spoilage by color change?' "
                        "BAD: 'What systems detect spoilage based on acetic acid, lactic acid, CO2? "
                        "Include dyes, concentrations, matrices, colors, and placement.' "
                        "BAD: stuffing concentration/matrix/color/placement requirements into the question. "
                        "If numbers/colors are missing, ask a focused scientific follow-up about that system, "
                        "or mark as a gap — do not pack field checklists into one call. "
                        "First call ≈ user intent; follow-ups = one focused scientific gap each, same turn."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": (
                                    "One natural-language English scientific question "
                                    "(full sentence about a mechanism or entity relationship). "
                                    "Do NOT pass keyword lists or table-field checklists."
                                ),
                            }
                        },
                        "required": ["question"],
                    },
                },
            },
        ]

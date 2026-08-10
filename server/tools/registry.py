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

    async def ask_subgraph(
        self,
        subquestions: list[str],
        effort: str = "medium",
    ) -> str:
        """
        Search the English knowledge graph (articles, patents, regulations).

        Budget: at most TWO calls per user question — high+low or medium+medium.
        Call 1: decompose the question into 1–6 English declarative statements.
        Call 2 (only if gaps remain): 1–3 narrow statements naming the missing info;
        they do not have to re-decompose the original question.

        Returns UNIT blocks: SPINE = main directed edge path; FANS @Hub = extra
        facts about a hub node (fan leaves are NOT linked to each other).
        Each edge: Label: A -[REL: "evidence"]-> Label: B (source_file.pdf; conf=0-1).
        Answer only from these chains; cite [n] and end with ### Источники;
        state remaining GAPS honestly.

        Args:
            subquestions: 1–6 English declarative statements (light HyDE ok; no '?').
            effort: search budget — low (1 iteration), medium (2, default), high (3).
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
                        "Search the English knowledge graph (scientific articles, patents, "
                        "regulations) for food technology: starter cultures, strains, freshness "
                        "indicators, smart packaging. ALWAYS call it for any content question. "
                        "BUDGET: at most TWO calls per user question — high+low or medium+medium. "
                        "Call 1: decompose the user question into 1–6 English DECLARATIVE "
                        "statements (light HyDE allowed: extend with plausible general domain "
                        "facts, never invent specific numbers/substances). Call 2 ONLY if gaps "
                        "remain after call 1: 1–3 narrow statements naming the missing "
                        "information (need not re-decompose the original question). "
                        "RETURNS: UNIT blocks. SPINE = main directed path of edges "
                        "Label: A -[REL: \"verbatim evidence\"]-> Label: B (source_file.pdf; conf=0-1). "
                        "FANS @Hub = extra facts about a hub node; fan leaves are NOT linked "
                        "to each other — never infer Leaf1→Leaf2 from a shared hub. "
                        "Answer ONLY from these chains: never transfer properties between "
                        "entities, cite every fact as [n] and end with ### Источники listing "
                        "[n] source_file.pdf, then honestly list GAPS (what the DB did not cover). "
                        "If the user asks for a confidence score, derive it from the per-edge "
                        "conf of the edges behind the claim and state the method; never invent it."
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
                                    "1–6 English declarative statements: units of information "
                                    "you want from the DB. Atomic but contextual (product / "
                                    "process / substance class). No question marks, no field "
                                    "checklists, no Russian. "
                                    "GOOD: 'Colorimetric freshness indicators change color in "
                                    "response to volatile amines in packaged food headspace.' "
                                    "BAD: 'What dyes are used? Include concentration, matrix, "
                                    "color, placement.'"
                                ),
                            },
                            "effort": {
                                "type": "string",
                                "enum": ["low", "medium", "high"],
                                "description": (
                                    "Search budget = number of chain-selection iterations: "
                                    "low=1 (single narrow fact), medium=2 (default, typical "
                                    "question), high=3 (broad multi-entity / table request). "
                                    "Choose by question WIDTH. Max two calls total: "
                                    "high+low or medium+medium."
                                ),
                            },
                        },
                        "required": ["subquestions"],
                    },
                },
            },
        ]

import logging
from typing import Any, Callable, Dict, List

from server.utils.config import AppConfig
from server.tools.graph_filter import GraphFilterAgent
from server.tools.graph_qa import GraphQA
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
        self.graph_qa = GraphQA(
            config.neo4j, llm_profile, config.llm.history_len, config.run_id
        )
        self.graph_filter = GraphFilterAgent(config.neo4j, llm_profile, config.run_id)

    async def ask_database(self, question: str) -> str:
        """
        Queries the knowledge graph database in natural language.
        Use for ANY question about entities, relationships, properties, or paths in the graph.
        Returns structured data including provenance: evidence (verbatim quote), source_file, chunk_id.

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

    def get_tools_list(self) -> List[Callable]:
        """
        Возвращает список всех доступных функций-инструментов.
        """
        return [self.ask_database]

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
                    "name": "ask_database",
                    "description": (
                        "Queries the knowledge graph database in natural language. "
                        "Use for ANY question about entities, relationships, properties, or paths in the graph. "
                        "Returns structured data including provenance fields: "
                        "evidence (verbatim quote from source), source_file (document reference), chunk_id."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": (
                                    "Natural language question to the database. "
                                    "IMPORTANT: The database contains English-only text. "
                                    "Always formulate the question in English, "
                                    "translating any non-English terms before calling this tool. "
                                    "Examples: 'What metabolites are affected by room temperature?', "
                                    "'Find the path between temperature and quality', "
                                    "'Which substances inhibit fermentation?'"
                                ),
                            }
                        },
                        "required": ["question"],
                    },
                },
            }
        ]

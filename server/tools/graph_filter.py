import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from server.tools.graph_viz import GraphVizExtractor
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

_CYPHER_BLOCK_RE = re.compile(r"```cypher\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_PROMPT_PATH = Path("prompts/graph_filter/system.md")


class GraphFilterAgent:
    """
    Агент для генерации Cypher-запроса визуализации по ответу ассистента.

    Принимает текст ответа ассистента и оригинальный Cypher-запрос,
    генерирует новый Cypher, который возвращает только упомянутые
    в ответе ноды и связи.

    Использует GraphVizExtractor для парсинга результатов Neo4j
    в формат {nodes, edges}.
    """

    def __init__(self, neo4j_config, llm_profile, run_id: str):
        self.run_id = run_id

        self.client = AsyncOpenAI(
            base_url=llm_profile.base_url,
            api_key=llm_profile.api_key or "api-key",
        )
        self.model = llm_profile.model

        self.schema = """
            Node labels and properties:
            - Microbe {name: STRING, leiden_community: INTEGER}
            - Metabolite {name: STRING, leiden_community: INTEGER}
            - EnvironmentCondition {name: STRING, leiden_community: INTEGER}

            Relationship types:
            - PRODUCES, CONSUMES, INHIBITS, STIMULATES, REQUIRES

            Valid relationships (Source -> RELATION -> Target):
            - Microbe -> PRODUCES|CONSUMES -> Metabolite|EnvironmentCondition
            - Microbe -> INHIBITS|STIMULATES -> Microbe
            - Microbe -> REQUIRES -> Metabolite|EnvironmentCondition
            - Metabolite -> INHIBITS|STIMULATES|PRODUCES|CONSUMES -> Metabolite
            - Metabolite -> INHIBITS|STIMULATES -> Microbe
            - Metabolite -> REQUIRES -> EnvironmentCondition
            IMPORTANT RULES: 
            1. EnvironmentCondition is NEVER the source of any relationship.
            2. INHIBITS and STIMULATES relationships NEVER target EnvironmentCondition.

            Relationship properties (apply to all relationships):
            - confidence: FLOAT  -- reliability score [0.0, 1.0]
            - evidence: STRING   -- verbatim quote from source document
            - source_file: STRING
            - chunk_id: STRING
            - run_id: STRING
        """

        self.viz_extractor = GraphVizExtractor(
            uri=neo4j_config.uri,
            user=neo4j_config.user,
            password=neo4j_config.password,
        )

        try:
            self._system_tpl = _PROMPT_PATH.read_text(encoding="utf-8")
            logger.info("GraphFilterAgent prompt loaded")
        except FileNotFoundError as e:
            logger.error(f"Ошибка загрузки промпта GraphFilterAgent: {e}")
            raise

        logger.info(
            f"GraphFilterAgent инициализирован: model={self.model}, run_id={self.run_id}"
        )

    def _extract_cypher(self, text: str) -> str:
        """Извлекает Cypher-запрос из ```cypher``` markdown-блока."""
        text = _THINK_TAG_RE.sub("", text).strip()
        match = _CYPHER_BLOCK_RE.search(text)
        if match:
            return match.group(1).strip()
        logger.warning(
            "Cypher-блок не найден в ответе GraphFilterAgent, используем весь текст"
        )
        return text.strip()

    async def _generate_viz_cypher(
        self, assistant_answer: str, original_cypher: str
    ) -> Optional[str]:
        """Генерирует Cypher для визуализации через LLM."""
        with tracer.start_as_current_span("generate_viz_cypher") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.CHAIN)
            span.set_attribute(OI_INPUT_VALUE, assistant_answer[:200])
            span.set_attribute("original_cypher", original_cypher)

            system = self._system_tpl.format(
                schema=self.schema,
                original_cypher=original_cypher,
                run_id=self.run_id,
            )

            user_content = (
                f"Assistant's answer:\n{assistant_answer}\n\n"
                f"Generate a Cypher query for graph visualization that shows "
                f"ONLY the entities mentioned in the answer above."
            )

            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0,
                )

                raw = response.choices[0].message.content or ""
                cypher = self._extract_cypher(raw)
                cypher = re.sub(
                    r"run_id\s*=\s*['\"][^'\"]+['\"]",
                    f"run_id = '{self.run_id}'",
                    cypher,
                )
                cypher = re.sub(
                    r"run_id\s*:\s*['\"][^'\"]+['\"]",
                    f"run_id: '{self.run_id}'",
                    cypher,
                )
                logger.info(f"GraphFilterAgent сгенерировал Cypher: {cypher}")
                set_span_ok(span, cypher)
                return cypher
            except Exception as e:
                logger.error(f"Ошибка генерации viz Cypher: {e}")
                set_span_error(span, str(e))
                return None

    async def build_viz_graph(
        self, assistant_answer: str, original_cypher: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Генерирует данные графа для визуализации.

        1. LLM генерирует точный Cypher по ответу ассистента.
        2. Выполняет Cypher в Neo4j.
        3. Парсит результат в {nodes, edges}.

        Args:
            assistant_answer: Текст ответа ассистента.
            original_cypher: Оригинальный Cypher-запрос из GraphQA.

        Returns:
            Словарь {nodes: [...], edges: [...]}.
        """
        with tracer.start_as_current_span("build_viz_graph") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.TOOL)
            span.set_attribute(OI_INPUT_VALUE, assistant_answer[:200])

            viz_cypher = await self._generate_viz_cypher(
                assistant_answer, original_cypher
            )
            if not viz_cypher:
                logger.warning("Не удалось сгенерировать viz Cypher, пустой граф")
                return {"nodes": [], "edges": []}

            try:
                graph_data = self.viz_extractor.execute_and_parse(viz_cypher)
                logger.info(
                    f"GraphFilterAgent: {len(graph_data['nodes'])} нод, "
                    f"{len(graph_data['edges'])} связей"
                )
                set_span_ok(
                    span,
                    f"nodes={len(graph_data['nodes'])}, edges={len(graph_data['edges'])}",
                )
                return graph_data
            except Exception as e:
                logger.error(f"Ошибка выполнения viz Cypher: {e}")
                set_span_error(span, str(e))
                return {"nodes": [], "edges": []}

    def close(self):
        """Освобождает ресурсы."""
        self.viz_extractor.close()

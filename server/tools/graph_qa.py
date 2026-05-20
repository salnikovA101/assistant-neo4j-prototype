import json
import logging
import re
from collections import deque
from pathlib import Path
from typing import Deque, Tuple

from langchain_neo4j import Neo4jGraph
from openai import AsyncOpenAI

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

DEFAULT_LIMIT = 20

_CYPHER_BLOCK_RE = re.compile(r"```cypher\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_PROMPTS_DIR = Path("prompts/graph_qa")


class SafeReadOnlyNeo4jGraph(Neo4jGraph):
    def query(self, query: str, params: dict = {}, session_params: dict = {}) -> list:
        """Выполняет запрос строго в Read-Only транзакции на уровне БД."""

        def read_tx(tx):
            result = tx.run(query, params)
            return [record.data() for record in result]

        safe_session_params = session_params.copy()
        safe_session_params["default_access_mode"] = "READ"

        with self._driver.session(**safe_session_params) as session:
            try:
                return session.execute_read(read_tx)
            except Exception as e:
                logger.error(f"БЛОКИРОВКА ИЛИ ОШИБКА ЗАПРОСА: {e}. Запрос: {query}")
                raise ValueError(
                    f"Запрос заблокирован базой данных (попытка изменения данных) или содержит ошибку: {e}"
                )


class GraphQA:
    """
    Асинхронный Text-to-Cypher pipeline.
    Генерирует Cypher-запрос через LLM, выполняет его в Neo4j,
    и возвращает сырые данные. При пустом результате — retry с фидбеком.
    """

    MAX_TURNS = 3

    def __init__(self, neo4j_config, llm_profile, max_len: int):
        """
        Инициализирует подключение к Neo4j и async OpenAI клиент.
        """
        try:
            self.graph = SafeReadOnlyNeo4jGraph(
                url=neo4j_config.uri,
                username=neo4j_config.user,
                password=neo4j_config.password,
            )
            logger.info("SafeReadOnlyNeo4jGraph инициализирован")
        except Exception as e:
            logger.error(f"Ошибка подключения Neo4jGraph: {e}")
            raise

        self.client = AsyncOpenAI(
            base_url=llm_profile.base_url,
            api_key=llm_profile.api_key or "api-key",
        )
        self.model = llm_profile.model
        self.schema = self.graph.schema
        # Компактная схема
        # self.schema = """
        #     Node labels and properties:
        #     - Microbe {name: STRING, leiden_community: INTEGER}
        #     - Metabolite {name: STRING, leiden_community: INTEGER}
        #     - EnvironmentCondition {name: STRING, leiden_community: INTEGER}

        #     Relationship types:
        #     - PRODUCES, CONSUMES, INHIBITS, STIMULATES, REQUIRES

        #     Relationship properties (apply to all relationships):
        #     - confidence: FLOAT  -- reliability score [0.0, 1.0]
        #     - evidence: STRING   -- verbatim quote from source document
        #     - source_file: STRING
        #     - chunk_id: STRING
        #     - run_id: STRING
        # """
        logger.info(self.schema)
        self.successful_queries: Deque[Tuple[str, str]] = deque(maxlen=max_len)

        try:
            self._cypher_system_tpl = (_PROMPTS_DIR / "cypher_system.md").read_text(
                encoding="utf-8"
            )
            self._cypher_examples = (_PROMPTS_DIR / "cypher_examples.md").read_text(
                encoding="utf-8"
            )
            logger.info("Cypher prompts loaded from prompts/graph_qa/")
        except FileNotFoundError as e:
            logger.error(f"Ошибка загрузки Cypher-промптов: {e}")
            raise

        logger.info(
            f"GraphQA инициализирован: model={self.model}, schema_len={len(self.schema)}"
        )

    def _format_history(self) -> str:
        """Форматирует историю успешных запросов для system prompt.

        Примеры из cypher_examples.md всегда присутствуют первыми как якорные примеры провенанса.
        История сессии дописывается следом, если есть успешные запросы.
        """
        if self.successful_queries:
            lines = ["Previous successful queries in this session (use as reference):"]
            for question, cypher in self.successful_queries:
                lines.append(f"  Question: {question}")
                lines.append(f"  Cypher: {cypher}")
            return "\n".join(lines)
        return ""

    def _extract_cypher(self, text: str) -> str:
        """Извлекает Cypher-запрос из ```cypher``` markdown-блока."""
        text = _THINK_TAG_RE.sub("", text).strip()
        match = _CYPHER_BLOCK_RE.search(text)
        if match:
            return match.group(1).strip()
        logger.warning(
            "Cypher-блок не найден в ответе LLM, используем весь текст как запрос."
        )
        return text.strip()

    async def _generate_cypher(self, question: str, feedback: str = "") -> str:
        """
        Единственный LLM-вызов: вопрос → Cypher.
        """
        with tracer.start_as_current_span("generate_cypher") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.CHAIN)
            span.set_attribute(OI_INPUT_VALUE, question)
            span.set_attribute("question", question)
            if feedback:
                span.set_attribute("feedback", feedback[:200])

            system = self._cypher_system_tpl.format(
                schema=self.schema,
                history=self._format_history(),
                limit=DEFAULT_LIMIT,
            )

            user_content = question
            if feedback:
                user_content = f"{question}\n\n{feedback}"

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
                logger.info(f"Сгенерирован Cypher: {cypher}")
                set_span_ok(span, cypher)
                return cypher
            except Exception as e:
                set_span_error(span, str(e))
                raise

    async def query(self, question: str) -> str:
        """
        Задает вопрос базе данных на естественном языке.

        Возвращает сырые данные из Neo4j в виде JSON-строки,
        либо сообщение об ошибке если все попытки неудачны.
        """
        with tracer.start_as_current_span("graph_qa_query") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.TOOL)
            span.set_attribute(OI_INPUT_VALUE, question)
            span.set_attribute("question", question)
            logger.info(f"GraphQA.query: {question}")

            feedback = ""

            try:
                for attempt in range(1, self.MAX_TURNS + 1):
                    logger.debug(f"Попытка {attempt}/{self.MAX_TURNS}")

                    try:
                        cypher = await self._generate_cypher(question, feedback)
                    except Exception as e:
                        logger.error(f"Ошибка генерации Cypher: {e}")
                        feedback = f"LLM returned an error: {e}. Try again."
                        continue

                    try:
                        result = self.graph.query(cypher)
                    except ValueError as e:
                        logger.warning(
                            f"Ошибка выполнения запроса (попытка {attempt}): {e}"
                        )
                        feedback = (
                            f"Your query `{cypher}` caused a database error: {e}. "
                            f"Write a corrected query."
                        )
                        continue

                    if result:
                        self.successful_queries.append((question, cypher))
                        data = json.dumps(result, ensure_ascii=False, default=str)
                        logger.info(f"Данные получены ({len(result)} записей)")
                        set_span_ok(span, data)
                        return data

                    logger.info(f"Пустой результат (попытка {attempt}), retry")
                    if attempt >= 2:
                        feedback = (
                            f"Your query `{cypher}` returned empty results again. "
                            f"The node might not exist with that exact name. "
                            f"Try using a case-insensitive regex pattern with =~ "
                            f"(e.g., `n.name =~ '(?i).*keyword.*'`) to find similar nodes."
                        )
                    else:
                        feedback = (
                            f"Your query `{cypher}` returned empty results. "
                            f"Rewrite it using a different approach."
                        )

                err_msg = "Не удалось найти данные в базе после нескольких попыток."
                set_span_error(span, err_msg)
                return err_msg

            except Exception as e:
                logger.error(f"Критическая ошибка в GraphQA: {e}")
                err_msg = f"Произошла ошибка при обращении к базе данных: {e}"
                set_span_error(span, err_msg)
                return err_msg

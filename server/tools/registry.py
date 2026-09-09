import logging
import re
from collections.abc import Callable
from typing import Any

from server.core.turn_state import current_turn, search_depth
from server.service_guide import load_service_guide
from server.tools.source_registry import SourceRegistry
from server.tools.subgraph_search import TOOL_ERROR, SubgraphSearchAgent
from server.utils.config import AppConfig

logger = logging.getLogger(__name__)


def _service_guide_tool_schema() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "get_service_guide",
            "description": (
                "Read the current Neo4j Assistant user guide. Call this when the "
                "user asks what the service or assistant can do, or how to use a UI "
                "feature, work mode, search, sources, research map, branches, cards, "
                "settings, or troubleshooting. Do not call it for domain research. "
                "When answering from this documentation, do not expose internal "
                "source markers or citation ids."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    }


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
        self.prompt_folder = config.llm.prompt_folder
        # Fail fast: the tool is part of every mode and must never be advertised
        # with a missing backing document. Calls still reload the file so the UI
        # and assistant see prompt-volume edits without a server restart.
        load_service_guide(self.prompt_folder)
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
            subquestions: 1–5 standalone neutral English questions, one per aspect.
            ignored: tolerated legacy/hallucinated arguments (e.g. `effort`);
                search depth comes from the UI, not from the model.
        """
        if subquestions is not None and not isinstance(subquestions, list):
            return (
                f"{TOOL_ERROR}: subquestions must be a JSON array of strings."
            )
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

        return await self.subgraph_search.query(subquestions=subquestions)

    async def advance_research(
        self,
        open_sq_refs: list[str] | None = None,
        new_subquestions: list[str] | None = None,
        **ignored: Any,
    ) -> str:
        """Start or continue staged SQs; new SQ still require the approval gate."""
        refs = [str(value).strip() for value in (open_sq_refs or []) if str(value).strip()]
        proposed = [str(value).strip() for value in (new_subquestions or []) if str(value).strip()]
        if len(refs) + len(proposed) > 5:
            return f"{TOOL_ERROR}: at most 5 SQ may be used in one graph call."
        if proposed:
            return (
                f"{TOOL_ERROR}: new staged SQ require user approval before retrieval."
            )
        turn = current_turn()
        if turn is None or turn.mode != "staged" or turn.store is None:
            return f"{TOOL_ERROR}: advance_research is available only in staged mode."
        resolved = await turn.store.open_agenda_subquestions(turn.checkpoint_id, refs)
        if len(resolved) != len(list(dict.fromkeys(refs))):
            return f"{TOOL_ERROR}: one or more SQ refs are unknown, hidden or closed."
        if not resolved:
            return f"{TOOL_ERROR}: select at least one open SQ."
        if ignored:
            logger.info("advance_research: игнорируем аргументы модели %s", list(ignored))
        return await self.subgraph_search.query(
            subquestions=[item["text"] for item in resolved]
        )

    def get_service_guide(self, **ignored: Any) -> str:
        """Return the shared user guide without adding it to the system prompt."""
        if ignored:
            logger.info(
                "get_service_guide: игнорируем аргументы модели %s", list(ignored)
            )
        # The service guide is product documentation, not a retrieved evidence
        # source.  Return it verbatim so citation rendering cannot turn its
        # prose into unresolved ``[?]`` markers.
        guide = load_service_guide(self.prompt_folder)
        # Keep implementation-only citation notation out of user-facing help,
        # even if it is accidentally present in a customized guide file.
        return re.sub(
            r"\(\s*source\s*:\s*(?:N|\d+)\s*\)",
            "",
            guide,
            flags=re.IGNORECASE,
        )

    def clear_history(self) -> None:
        """Очищает сессионный source-реестр."""
        self.source_registry.clear()

    def get_tools_list(self, mode: str = "auto") -> list[Callable]:
        """
        Возвращает список всех доступных функций-инструментов.
        """
        mode_tool = self.advance_research if mode == "staged" else self.ask_subgraph
        return [mode_tool, self.get_service_guide]

    def get_tool_map(self, mode: str = "auto") -> dict[str, Callable]:
        """
        Создает словарь соответствия имен функций их объектам.
        """
        return {func.__name__: func for func in self.get_tools_list(mode)}

    def get_openai_tools(self, mode: str = "auto") -> list[dict[str, Any]]:
        """
        Возвращает список инструментов в формате JSON Schema для OpenAI SDK.
        """
        if mode == "staged":
            return [
                {
                    "type": "function",
                    "function": {
                        "name": "advance_research",
                        "description": (
                            "Start or continue staged graph search for food technology "
                            "(starter cultures, freshness indicators, smart packaging). "
                            "Empty research-question list: put the first 1–5 standalone neutral English questions in "
                            "`new_subquestions` and send `open_sq_refs` as []. "
                            "Existing open items: pass their `subquestion:N` refs. "
                            "New search directions also go in `new_subquestions` and need "
                            "user approval. At least one of the two arrays must be non-empty. "
                            "At most 5 total SQ per call. "
                            "GOOD: 'Which starter cultures are used in cottage cheese "
                            "production?' BAD: 'What starter cultures are used?' (missing product context)"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "open_sq_refs": {
                                    "type": "array",
                                    "items": {"type": "string", "pattern": "^subquestion:[1-9][0-9]*$"},
                                    "maxItems": 5,
                                    "description": (
                                        "Open SQ refs exactly as listed in CURRENT RESEARCH QUESTIONS. "
                                        "Empty array or omit when starting a new research-question list."
                                    ),
                                },
                                "new_subquestions": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "maxItems": 5,
                                    "description": (
                                        "New standalone neutral English questions, preserve user constraints and unknowns; do not assume answers. No Russian. "
                                        "Required when the research-question list is empty or does not cover "
                                        "a necessary search direction. Empty array or omit "
                                        "when only existing open refs are searched."
                                    ),
                                },
                            },
                        },
                    },
                },
                _service_guide_tool_schema(),
            ]
        return [
            {
                "type": "function",
                "function": {
                    "name": "ask_subgraph",
                    "description": (
                        "Search the English knowledge graph for food technology "
                        "(starter cultures, freshness indicators, smart packaging). "
                        "Call it when the question names a product, substance, "
                        "culture, process, class or goal — including catalogs and "
                        "selection questions. Returns evidence UNITs: tours of "
                        "cards, each card a triple plus its evidence text and "
                        "(source:N). Search depth is set in the UI, not here."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "subquestions": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 1,
                                "maxItems": 5,
                                "description": (
                                    "1–5 standalone neutral English questions, "
                                    "no Russian. Each one runs a separate search "
                                    "over evidence texts, so each must cover a different "
                                    "aspect of the question — paraphrases return "
                                    "overlapping evidence. Preserve user constraints and unknowns; do not invent answers. "
                                    "GOOD: 'Which starter cultures are used in "
                                    "cottage cheese production?' "
                                    "BAD: 'What starter cultures are used?' (missing product context)"
                                ),
                            },
                        },
                        "required": ["subquestions"],
                    },
                },
            },
            _service_guide_tool_schema(),
        ]

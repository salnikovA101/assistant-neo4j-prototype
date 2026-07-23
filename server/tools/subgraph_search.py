import logging

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

# How many ranked paths to expose to the LLM agent
DEFAULT_TOP_K = 50


def _prize_key(path: dict) -> float:
    prize = path.get("prize_sum")
    if prize is not None:
        try:
            return float(prize)
        except (TypeError, ValueError):
            pass
    try:
        return -float(path.get("totalCost", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


class SubgraphSearchAgent:
    """
    Агент для поиска подграфов.
    Вызывает V4 pipeline (stages 1–6) и возвращает сериализованные пути.
    """

    def __init__(self, llm_profile, run_id: str, top_k: int = DEFAULT_TOP_K):
        self.run_id = run_id
        self.top_k = top_k
        logger.info(f"SubgraphSearchAgent инициализирован с run_id='{self.run_id}'")
        self.model = llm_profile.model
        from server.algorithm.pipeline import create_default_pipeline

        self.pipeline = create_default_pipeline(llm_profile)

    async def query(self, question: str) -> str:
        """Основной метод инструмента."""
        with tracer.start_as_current_span("subgraph_search_query") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.TOOL)
            span.set_attribute(OI_INPUT_VALUE, question)
            span.set_attribute("question", question)

            try:
                paths = await self.pipeline.run(question=question, top_k=self.top_k)
                paths = sorted(paths, key=_prize_key, reverse=True)

                if not paths:
                    res_str = "No relevant information found in the graph for this query."
                else:
                    lines = ["### Retrieved graph paths"]
                    for idx, path in enumerate(paths, 1):
                        text = (path.get("serialized_text") or "").strip()
                        if not text:
                            continue
                        lines.append(f"{idx}. {text}")

                    if len(lines) == 1:
                        res_str = "No relevant relationships found."
                    else:
                        res_str = "\n\n".join(lines)

                set_span_ok(span, res_str)
                return res_str

            except Exception as e:
                logger.error(f"Ошибка в SubgraphSearchAgent: {e}")
                err_msg = f"Произошла ошибка при поиске подграфов: {e}"
                set_span_error(span, err_msg)
                return err_msg

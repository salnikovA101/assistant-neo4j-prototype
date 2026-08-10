import logging
from typing import Any, Sequence

from server.core.db import get_driver
from server.tools.source_registry import (
    SourceRegistry,
    collect_source_files,
    remap_filenames_to_source_ids,
)
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


def _normalize_effort(effort: str | None) -> str:
    e = (effort or "medium").strip().lower()
    if e in {"low", "medium", "high"}:
        return e
    return "medium"


def _format_accepted_chains(
    accepted: Sequence[dict[str, Any]],
    registry: SourceRegistry | None = None,
) -> str:
    lines = [
        "### Retrieved evidence chains",
        "",
        "Edge format: Label: A -[REL: \"verbatim evidence\"]-> Label: B (source:N; conf=0-1). "
        "SPINE = main path (read top-down); FANS @Hub = extra hub facts, leaves NOT linked "
        "to each other. Cite claims as (source:N) or (source:1; source:2) — copy ids from "
        "edges; never invent ids; never write [n] or ### Источники (server adds those). "
        "Never cite UNIT indices as bibliography. Answer only from these chains; list GAPS honestly.",
        "",
    ]
    n = 0
    for item in accepted:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        if registry is not None:
            for sf in collect_source_files([item]):
                registry.register(sf)
            text = remap_filenames_to_source_ids(text, registry)
        n += 1
        # Renumber UNIT label for citation-friendly display
        if text.startswith("UNIT "):
            rest = text.split("\n", 1)
            body = rest[1] if len(rest) > 1 else ""
            score_part = ""
            head = rest[0]
            if "(score=" in head:
                score_part = "  " + head[head.index("(score=") :]
            text = f"UNIT [{n}]{score_part}\n{body}".rstrip()
        else:
            text = f"UNIT [{n}]\n{text}"
        lines.append(text)
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()

    if n == 0:
        return "No relevant chains found in the graph for these subquestions."
    return "\n".join(lines)


class SubgraphSearchAgent:
    """
    Runs the graph retrieval pipeline and returns accepted evidence chains.
    Decomposition and effort selection are done by the assistant LLM.
    """

    def __init__(
        self,
        llm_profile=None,
        run_id: str = "",
        source_registry: SourceRegistry | None = None,
    ):
        self.run_id = run_id
        self.llm_profile = llm_profile
        self.source_registry = source_registry if source_registry is not None else SourceRegistry()
        logger.info("SubgraphSearchAgent initialized run_id=%r", self.run_id)

    async def query(
        self,
        subquestions: list[str] | None = None,
        effort: str = "medium",
    ) -> str:
        """Run pipeline for assistant-supplied subquestions + effort."""
        sqs = [str(s).strip() for s in (subquestions or []) if str(s).strip()]
        effort_n = _normalize_effort(effort)

        with tracer.start_as_current_span("subgraph_search_query") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.TOOL)
            span.set_attribute(OI_INPUT_VALUE, " | ".join(sqs)[:500])
            span.set_attribute("effort", effort_n)
            span.set_attribute("n_subquestions", len(sqs))

            if not sqs:
                err = "ask_subgraph requires a non-empty subquestions list."
                set_span_error(span, err)
                return err

            try:
                from server.algorithm.pipeline import run

                driver = get_driver()
                payload = [
                    {"id": f"sq{i+1}", "text": text} for i, text in enumerate(sqs)
                ]
                result = await run(
                    driver,
                    subquestions=payload,
                    effort=effort_n,
                )
                accepted = result.get("accepted") or []
                res_str = _format_accepted_chains(accepted, self.source_registry)
                set_span_ok(span, res_str)
                return res_str
            except Exception as e:
                logger.exception("SubgraphSearchAgent failed")
                err_msg = f"Error in ask_subgraph: {e}"
                set_span_error(span, err_msg)
                return err_msg

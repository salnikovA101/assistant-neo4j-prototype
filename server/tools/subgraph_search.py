import logging
import re
from typing import Any, Sequence

from server.core.db import get_driver
from server.core.graph_runs import chain_unit_index, record_accepted_chains
from server.core.sessions import current_sources
from server.core.turn_state import (
    remember_subquestions,
    search_depth,
    searches_state,
    seen_subquestions,
    subquestion_key,
    take_search_slot,
)
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

# Machine-unambiguous prefixes: the assistant prompt maps each to one behaviour.
TOOL_ERROR = "TOOL_ERROR"
NO_RESULTS = "NO_RESULTS"

MAX_SUBQUESTIONS = 6
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def normalize_subquestions(
    raw: Sequence[Any] | None,
    seen: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Enforce the subquestion contract in code: English declarative statements,
    no duplicates inside a call or against earlier calls of the same turn,
    at most MAX_SUBQUESTIONS.

    Returns (usable statements, human-readable problems).
    """
    clean: list[str] = []
    problems: list[str] = []
    local_seen: set[str] = set()
    seen = seen or set()

    for item in raw or []:
        text = str(item).strip()
        if not text:
            continue
        if _CYRILLIC_RE.search(text):
            problems.append(f"not English, dropped: {text[:60]}")
            continue
        text = text.rstrip("?").strip()
        key = subquestion_key(text)
        if not key:
            continue
        if key in local_seen:
            problems.append(f"duplicate in this call, dropped: {text[:60]}")
            continue
        if key in seen:
            problems.append(f"already searched this turn, dropped: {text[:60]}")
            continue
        local_seen.add(key)
        clean.append(text)

    if len(clean) > MAX_SUBQUESTIONS:
        problems.append(
            f"{len(clean)} subquestions sent, only the first {MAX_SUBQUESTIONS} used"
        )
        clean = clean[:MAX_SUBQUESTIONS]

    return clean, problems


def _format_accepted_chains(
    accepted: Sequence[dict[str, Any]],
    registry: SourceRegistry | None = None,
) -> str:
    lines = [
        "### Retrieved evidence chains",
        "",
        "Format only (behaviour rules are in the system prompt):",
        "UNIT = one tour in walk order; consecutive cards share a vertex "
        "(A-B, then B-C).",
        "Card = `Label: A —REL→ Label: B`, next line = the verbatim quote with "
        "(source:N; conf=0-1 or None).",
        "@Hub = still at that vertex (a sibling edge), not the next process step.",
        "A quote may name more entities than the node labels do.",
        "conf and UNIT numbers are service fields; source:N is copied from the "
        "quote line.",
        "",
    ]
    units: list[str] = []
    first_unit: int | None = None
    last_unit = 0
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
        unit_n = chain_unit_index(item, n)
        if first_unit is None:
            first_unit = unit_n
        last_unit = unit_n
        if text.startswith("UNIT "):
            rest = text.split("\n", 1)
            body = rest[1] if len(rest) > 1 else ""
            text = f"UNIT [{unit_n}]\n{body}".rstrip()
        else:
            text = f"UNIT [{unit_n}]\n{text}"
        units.append(text)

    if n == 0:
        return f"{NO_RESULTS}: no evidence chains matched these subquestions."
    if first_unit and first_unit > 1:
        lines.append(
            f"UNIT numbers continue this turn: this batch is UNIT [{first_unit}]–[{last_unit}]. "
            "Do not reuse UNIT indices from an earlier ask_subgraph in this answer."
        )
        lines.append("")
    for i, text in enumerate(units):
        if i:
            lines.append("---")
            lines.append("")
        lines.append(text)
        lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


class SubgraphSearchAgent:
    """
    Runs the graph retrieval pipeline and returns accepted evidence chains.

    Decomposition is the assistant's job; search depth comes from the UI and the
    per-turn search budget is enforced here, not asked of the model.
    """

    def __init__(self, source_registry: SourceRegistry | None = None):
        self.source_registry = source_registry if source_registry is not None else SourceRegistry()
        logger.info("SubgraphSearchAgent initialized")

    async def query(self, subquestions: list[str] | None = None) -> str:
        """Validate subquestions, spend one search slot, run the pipeline."""
        sqs, problems = normalize_subquestions(subquestions, seen_subquestions())
        depth = search_depth()

        with tracer.start_as_current_span("subgraph_search_query") as span:
            span.set_attribute(OI_SPAN_KIND, OISpanKind.TOOL)
            span.set_attribute(OI_INPUT_VALUE, " | ".join(sqs)[:500])
            span.set_attribute("effort", depth)
            span.set_attribute("n_subquestions", len(sqs))
            if problems:
                span.set_attribute("input_problems", "; ".join(problems)[:500])

            if not sqs:
                err = (
                    f"{TOOL_ERROR}: no usable subquestions. Send 1-"
                    f"{MAX_SUBQUESTIONS} English declarative statements, each a "
                    "different aspect of the question."
                )
                if problems:
                    err = f"{err} Rejected: {'; '.join(problems)}."
                logger.warning("ask_subgraph rejected input: %s", problems)
                set_span_error(span, err)
                return err

            if not take_search_slot():
                used, limit = searches_state()
                err = (
                    f"{TOOL_ERROR}: search budget for this answer is spent "
                    f"({used}/{limit}). Answer now from the evidence already "
                    "retrieved."
                )
                set_span_error(span, err)
                return err

            remember_subquestions(sqs)
            used, limit = searches_state()
            span.set_attribute("search_slot", f"{used}/{limit}")

            try:
                from server.algorithm.params import merge_params
                from server.algorithm.pipeline import run
                from server.utils.config import load_config, retrieval_param_overrides

                driver = get_driver()
                payload = [
                    {"id": f"sq{i+1}", "text": text} for i, text in enumerate(sqs)
                ]
                result = await run(
                    driver,
                    subquestions=payload,
                    effort=depth,
                    params=merge_params(retrieval_param_overrides(load_config())),
                )
                if result.get("error"):
                    err_msg = f"{TOOL_ERROR}: {result['error']}"
                    detail = result.get("error_detail")
                    if detail:
                        err_msg = f"{err_msg}: {detail}"
                    set_span_error(span, err_msg)
                    return err_msg
                accepted = record_accepted_chains(result.get("accepted") or [])
                registry = current_sources() or self.source_registry
                res_str = _format_accepted_chains(accepted, registry)
                if problems:
                    res_str = f"{res_str}\n\n[Input note: {'; '.join(problems)}.]"
                set_span_ok(span, res_str)
                return res_str
            except Exception as e:
                logger.exception("SubgraphSearchAgent failed")
                err_msg = f"{TOOL_ERROR}: {e}"
                set_span_error(span, err_msg)
                return err_msg

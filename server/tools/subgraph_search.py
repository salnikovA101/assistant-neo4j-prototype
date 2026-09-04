import logging
import hashlib
import json
import re
from typing import Any, Sequence

from server.utils.constants import RETRIEVAL_STATE_VERSION
from server.core.db import get_driver
from server.core.graph_runs import chain_unit_index, record_accepted_chains
from server.core.sessions import current_sources
from server.core.turn_state import (
    current_turn,
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
logger = logging.getLogger(__name__)

# Machine-unambiguous prefixes: the assistant prompt maps each to one behaviour.
TOOL_ERROR = "TOOL_ERROR"
NO_RESULTS = "NO_RESULTS"

MAX_SUBQUESTIONS = 5
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
    if raw is not None and not isinstance(raw, (list, tuple)):
        return [], ["subquestions must be a JSON array of strings"]

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
        "Card = `Label: A —REL→ Label: B`, next line = the evidence text with "
        "(source:N; conf=0-1 or None).",
        "@Hub = still at that vertex (a sibling edge), not the next process step.",
        "Evidence text may name more entities than the node labels do.",
        "conf and UNIT numbers are service fields; source:N is copied from the "
        "evidence line.",
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

        if not sqs:
            err = (
                f"{TOOL_ERROR}: no usable subquestions. Send 1-"
                f"{MAX_SUBQUESTIONS} English declarative statements, each a "
                "different aspect of the question."
            )
            if problems:
                err = f"{err} Rejected: {'; '.join(problems)}."
            logger.warning("ask_subgraph rejected input: %s", problems)
            return err

        if not take_search_slot():
            used, limit = searches_state()
            return (
                f"{TOOL_ERROR}: search budget for this answer is spent "
                f"({used}/{limit}). Answer now from the evidence already "
                "retrieved."
            )

        remember_subquestions(sqs)

        try:
            from server.algorithm.params import merge_params
            from server.algorithm.pipeline import run
            from server.utils.config import load_config, retrieval_param_overrides

            driver = get_driver()
            turn = current_turn()
            params = merge_params(retrieval_param_overrides(load_config()))
            persistent = bool(
                turn
                and turn.store is not None
                and turn.user_id
                and turn.conversation_id
                and turn.checkpoint_id
            )
            if persistent:
                snapshot = await turn.store.conversation_source_snapshot(
                    turn.conversation_id
                )
                (current_sources() or self.source_registry).restore(snapshot)
                agenda = await turn.store.upsert_turn_subquestions(
                    turn.conversation_id,
                    turn.checkpoint_id,
                    sqs,
                    increment=True,
                    agenda_visible=turn.mode == "staged",
                )
                by_key = {
                    turn.store.canonical_subquestion(item["text"]): item
                    for item in agenda
                }
                # Agenda statuses are now tri-state. Both not_closed and partial
                # remain searchable; only explicitly closed SQs are excluded.
                open_ids = {item["id"] for item in agenda if item.get("status") != "closed"}
                payload = [
                    {
                        "id": by_key[turn.store.canonical_subquestion(text)]["id"],
                        "text": text,
                    }
                    for text in sqs
                    if turn.store.canonical_subquestion(text) in by_key
                    and by_key[turn.store.canonical_subquestion(text)]["id"] in open_ids
                ]
                if not payload:
                    return f"{NO_RESULTS}: all selected subquestions are closed in the current research-question list."
                retrieval = dict(turn.retrieval_state or {})
                master = dict(retrieval.get("s3Bundle") or {})
                master_graphs = dict(master.get("graphs") or {})
                missing = [item for item in payload if item["id"] not in master_graphs]
                if missing:
                    built = await run(
                        driver,
                        subquestions=missing,
                        effort=depth,
                        params=params,
                        emit_s3_bundle=True,
                        budget_override=0,
                    )
                    if built.get("error"):
                        result = built
                    else:
                        fresh = dict(built.get("s3_bundle") or {})
                        for key in ("graphs", "ann_keys", "rerank_keys"):
                            merged = dict(master.get(key) or {})
                            merged.update(fresh.get(key) or {})
                            master[key] = merged
                        sims = dict(master.get("ann_edge_sims") or {})
                        sims.update(fresh.get("ann_edge_sims") or {})
                        master["ann_edge_sims"] = sims
                        master["version"] = 2
                        master_graphs = dict(master.get("graphs") or {})
                        result = {}
                else:
                    result = {}
                if not result.get("error"):
                    selected_ids = {item["id"] for item in payload}
                    subset = dict(master)
                    subset["graphs"] = {
                        key: value for key, value in master_graphs.items() if key in selected_ids
                    }
                    for key in ("ann_keys", "rerank_keys"):
                        subset[key] = {
                            sid: value for sid, value in (master.get(key) or {}).items() if sid in selected_ids
                        }
                    result = await run(
                        driver,
                        subquestions=payload,
                        effort=depth,
                        params=params,
                        s3_bundle=subset,
                        carousel_state=retrieval.get("carousel") or {},
                        prior_signatures=retrieval.get("priorSignatures") or [],
                        manual_round=turn.mode == "staged",
                    )
                if not result.get("error"):
                    p_before = dict((retrieval.get("carousel") or {}).get("p_store") or {})
                    recorded = await turn.store.record_units(
                        turn.conversation_id,
                        turn.checkpoint_id,
                        result.get("accepted") or [],
                    )
                    result["accepted"] = recorded
                    result["accepted_all"] = recorded
                    source_files = [
                        path
                        for item in recorded
                        for path in collect_source_files([item])
                    ]
                    snapshot = await turn.store.merge_conversation_sources(
                        turn.conversation_id, source_files
                    )
                    (current_sources() or self.source_registry).restore(snapshot)
                    carousel = dict(result.get("carousel_state") or {})
                    retrieval.update(
                        {
                            "algorithmVersion": RETRIEVAL_STATE_VERSION,
                            "s3Bundle": master,
                            "carousel": carousel,
                            "priorSignatures": list(carousel.get("accepted_signatures") or []),
                            "lastMode": turn.mode,
                            "lastDepth": depth,
                            "lastSubquestionIds": [item["id"] for item in payload],
                            "lastTrace": result.get("trace") or {},
                            "pBefore": p_before,
                            "params": result.get("params") or params.to_dict(),
                            "paramsHash": hashlib.sha256(
                                json.dumps(
                                    result.get("params") or params.to_dict(),
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest(),
                            "corpusRevision": str(params.run_id or "all"),
                            "s3Hashes": {
                                key: hashlib.sha256(
                                    json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
                                ).hexdigest()
                                for key, value in (master.get("graphs") or {}).items()
                            },
                        }
                    )
                    turn.retrieval_state = retrieval
            else:
                payload = [
                    {"id": f"sq{i+1}", "text": text} for i, text in enumerate(sqs)
                ]
                result = await run(
                    driver,
                    subquestions=payload,
                    effort=depth,
                    params=params,
                )
            if result.get("error"):
                err_msg = f"{TOOL_ERROR}: {result['error']}"
                detail = result.get("error_detail")
                if detail:
                    err_msg = f"{err_msg}: {detail}"
                return err_msg
            accepted = record_accepted_chains(result.get("accepted") or [])
            registry = current_sources() or self.source_registry
            res_str = _format_accepted_chains(accepted, registry)
            if problems:
                res_str = f"{res_str}\n\n[Input note: {'; '.join(problems)}.]"
            return res_str
        except Exception as e:
            logger.exception("SubgraphSearchAgent failed")
            return f"{TOOL_ERROR}: {e}"

"""S6: unit judge (N sq_closed + K chain_needed) + session p / spine-seq updates."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from server.algorithm.models import UNIT_RULES, Chain, SubQuestion
from server.algorithm.params import Params
from server.algorithm.slm_utils import (
    resolve_slm_base_url,
    resolve_tool_llm_profile,
)
from server.utils.config import load_config

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """
You are an evidence auditor inside a multi-round GraphRAG retrieval loop over
a corpus of food-microbiology papers: lactic acid bacteria and starter
cultures, kefir, yogurt and cheese fermentation, probiotics and GI survival,
GABA and other metabolites, bacteriocins and milk-protein peptides, biogenic
amines, exopolysaccharides, spoilage and freshness indicators.

Each UNIT = a SPINE (ordered directed edges) plus optional FANS at a hub.
One edge is Label: A -[REL: "verbatim quote from a paper"]-> Label: B, with
node labels Microbe / Metabolite / StarterCulture / EnvironmentCondition and
REL one of INHIBITS / STIMULATES / REQUIRES / COMPOSED_OF / PRODUCES / CONSUMES.
Adjacent SPINE lines may share an endpoint (walk order); FANS @Hub show
hub-centric leaves as -[REL]-> Leaf (out) or <-[REL]- Leaf (in).
Only the quotes are evidence; entity names, node labels and REL types alone
prove nothing. Sibling FAN edges are not linked to each other. Never invent
facts. `score=` and the order of units are retrieval artifacts — ignore them.

Open subquestions are declarative statements that quotes must SUPPORT; they
are not questions to answer. The loop runs several retrieval rounds: whatever
you reject or leave open is retried with fresh evidence. Your two ways to
damage the loop are approving junk and closing a statement too early. Both
defaults are conservative: chain_needed defaults to false, sq_closed
defaults to open.

REASONING FORMAT (strict): your whole analysis is at most one line per
statement plus one line per unit, in exactly this shape:
  sq3 | components: strain + tolerance value | strain ok, value missing → open
  c1 | s=2 | "strongest quote fragment" | serves sq3
Never enumerate a unit's quotes, never write one line per quote, never
mention the same unit id twice. Find the strongest quote, score it, move on.
If you catch yourself listing quotes or writing "wait" / "re-check", stop
immediately and emit the final JSON. Running out of budget before the JSON
is the one failure you cannot recover from.

=== STEP 1 — targets ===
For each open statement, list its components: the entities or process, plus
EVERY condition, criterion, mechanism or outcome it asserts. A statement
with three components needs all three covered.

=== STEP 2 — score every unit 0/1/2 on its SINGLE strongest quote ===
2 = PAYLOAD. The quote carries a concrete, citable fact for some component:
  - a measured value, range or dose with units (pH, %, °C, h, mg/L, µg/ml,
    fold change, log CFU, MIC, read counts, p-value);
  - a named strain, isolate or collection code ("ATCC 12345", "strain M1",
    "subsp. lactis LB12");
  - a named enzyme, gene, pathway, metabolite, peptide or compound class
    (glutamate decarboxylase/GAD, bile salt hydrolase, arginine deiminase,
    urease, protease, kefiran, exopolysaccharide, bacteriocin, PLP);
  - a named assay, medium or test condition (simulated gastric fluid, MRS
    broth, in vitro incubation, agar diffusion, response-surface design);
  - an explicit selection, screening or acceptance criterion;
  - a directional or comparative result (increased, decreased, inhibited,
    tolerated, survived, highest, best, most abundant).
1 = SPECIFIC BUT THIN. Names concrete species or strains together with their
    explicit role in the statement's process, but gives no value, condition
    or mechanism.
0 = GLUE. Existence or intention claims ("can be used", "have been
    developed", "has received attention", "is widely used"); definitions and
    category statements; title-like or keyword strings that name a topic
    without reporting a finding; truncated fragments that assert nothing;
    quotes that merely echo the statement's own words.
A quote that is not a complete factual clause is never score 2.

=== STEP 3 — chain_needed (one boolean per unit, default false) ===
Set true only when you can point to the payload quote that earns it.
- score 2 → true, unless TWIN RULE or HARD FAIL applies.
- score 1 → true only if it adds a species or role that no unit already true
  in THIS batch supplies; otherwise false, keeping the most specific one.
- score 0 → false.
- TWIN RULE: if two units rest their score on the same finding (same quote
  or paraphrases of one result), keep only the more specific unit — the one
  with more score-2 quotes; the other is false.
- HARD FAIL → false whatever the score:
  - the REL label contradicts its own quote (e.g. INHIBITS quoting
    stimulation);
  - no edge touches any component of any open statement;
  - the strongest quote is a truncated fragment that asserts nothing;
  - the unit is off-topic for every open statement.
- ANTI-COLLAPSE: if these rules leave every unit false, re-check the unit
  with the strongest quote and set it true unless it is a HARD FAIL.

=== STEP 4 — sq_closed (one boolean per open statement, default open) ===
Close a statement only if EVERY component from STEP 1 is backed by score-2
quotes inside units you marked true in this batch, and you can name the
covering quote for each component. One quote may cover several components
only if it explicitly states each of them.
- partial coverage, adjacent topic, or support only by score-0/1 quotes →
  open;
- units marked false never close anything;
- approving units is NOT coverage: a batch of good units can still miss a
  component, so needed=true for every unit does not by itself close any
  statement;
- in the first rounds some component is usually still missing — close only
  when coverage is complete enough that one more retrieval round would add
  nothing;
- when unsure, leave it open: a redundant round is cheap, while closing
  early discards the remaining evidence for good.

=== CONSISTENCY ===
Decide chain_needed first, then sq_closed. If chain_needed is false for
every unit, then sq_closed must be false for every statement — closing
requires accepted units. Never emit all-false chain_needed together with
any sq_closed=true.

=== OUTPUT ===
Score and decide while you think; do not restate the analysis afterwards.
Emit exactly one JSON object and nothing else, with chain_needed first:
{"chain_needed":{"c1":true,"c2":false},"sq_closed":{"sq1":false,"sq2":true}}
""".strip()


def apply_p_event(
    p_store: dict[str, float],
    edge_keys: list[str],
    event: float,
    *,
    floor: float,
    protected: set[str] | None = None,
) -> None:
    prot = protected or set()
    for ek in edge_keys:
        if not ek or ek in prot:
            continue
        cur = float(p_store.get(ek, 1.0))
        p_store[ek] = max(floor, cur * event)


def _reasoning_text(message: Any) -> str:
    """Thinking trace, when the endpoint exposes one (kept for eval traces only)."""
    for key in ("reasoning_content", "reasoning"):
        val = getattr(message, key, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    extra = getattr(message, "model_extra", None) or {}
    for key in ("reasoning_content", "reasoning"):
        val = extra.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _parse_bool_map(raw: Any) -> dict[str, bool]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, bool] = {}
    for k, v in raw.items():
        if isinstance(v, bool):
            out[str(k)] = v
        elif isinstance(v, (int, float)):
            out[str(k)] = bool(v)
        elif isinstance(v, str):
            out[str(k)] = v.strip().lower() in ("1", "true", "yes")
    return out


def _parse_judge_json(content: str) -> tuple[dict[str, bool], dict[str, bool]] | None:
    content = (content or "").strip()
    m = re.search(r"```(?:json)?\n?(.*?)\n?```", content, re.DOTALL)
    text = m.group(1).strip() if m else content
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    sq = _parse_bool_map(data.get("sq_closed") or data.get("subquestions") or {})
    ch = _parse_bool_map(data.get("chain_needed") or data.get("chains") or {})
    return sq, ch


async def judge_flat(
    open_sqs: list[SubQuestion],
    chains: list[Chain],
    params: Params,
) -> tuple[dict[str, bool], dict[str, bool], str, bool]:
    """
    Returns (sq_closed, chain_needed, raw_content, ok).

    On API/parse failure: ok=False, maps all-false, do NOT postprocess
    (no accept, no p_reject, no spine-seq burn, leave sq open).
    Empty inputs are ok=True (nothing to judge).
    """
    sq_ids = [s.id for s in open_sqs]
    chain_ids = [c.chain_id for c in chains]
    empty_sq = {i: False for i in sq_ids}
    empty_ch = {i: False for i in chain_ids}
    if not open_sqs or not chains:
        return empty_sq, empty_ch, "", True

    sq_block = "\n".join(f"- {s.id}: {s.text}" for s in open_sqs)
    # chains already score-ascending from S5; best last
    units_block = "\n\n".join(c.format_unit(c.chain_id) for c in chains)
    user = (
        f"{UNIT_RULES}\n\n"
        f"OPEN SUBQUESTIONS:\n{sq_block}\n\n"
        f"EVIDENCE UNITS (order and score are retrieval artifacts; "
        f"judge each unit on its quotes):\n{units_block}\n"
    )

    try:
        config = load_config()
        llm_profile = resolve_tool_llm_profile(config)
        client = AsyncOpenAI(
            api_key=llm_profile.api_key or "EMPTY",
            base_url=resolve_slm_base_url(llm_profile.base_url),
        )
        req: dict[str, Any] = {
            "model": llm_profile.model,
            "temperature": params.judge_temperature,
            # "max_tokens": params.judge_max_tokens,
            "messages": [
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": user},
            ],
        }
        think = bool(getattr(llm_profile, "think", False))
        effort = str(getattr(llm_profile, "think_effort", "high") or "high").strip().lower()
        if think:
            req["reasoning_effort"] = effort if effort != "max" else "high"
            req["extra_body"] = {
                "reasoning": {"enabled": True, "effort": effort},
                "thinking": {"type": "enabled"},
                "chat_template_kwargs": {"enable_thinking": True},
            }
            token = str(getattr(llm_profile, "think_token", "") or "").strip()
            if token and token not in JUDGE_PROMPT:
                req["messages"][0]["content"] = f"{token}\n{JUDGE_PROMPT}"
        else:
            req["reasoning_effort"] = "none"
            req["extra_body"] = {
                "reasoning": {"enabled": False, "effort": "none"},
                "thinking": {"type": "disabled"},
            }
        resp = await client.chat.completions.create(**req)
        message = resp.choices[0].message
        raw = message.content or ""
        thinking = _reasoning_text(message)
    except Exception as e:
        logger.warning("judge failed: %s", e)
        return empty_sq, empty_ch, "", False

    # Verdict first so the trace stays readable when truncated; parse only `raw`
    # (thinking text carries braces that would break JSON extraction).
    trace = f"{raw}\n\n--- thinking ---\n{thinking}" if thinking else raw

    parsed = _parse_judge_json(raw)
    if parsed is None:
        logger.warning("judge JSON parse fail")
        return empty_sq, empty_ch, trace, False

    sq_map, ch_map = parsed
    sq_closed = {i: bool(sq_map.get(i, False)) for i in sq_ids}
    chain_needed = {i: bool(ch_map.get(i, False)) for i in chain_ids}

    if not any(chain_needed.values()):
        sq_closed = {i: False for i in sq_ids}

    return sq_closed, chain_needed, trace, True


def postprocess_and_update_p(
    *,
    open_sqs: list[SubQuestion],
    all_sqs: list[SubQuestion],
    chains: list[Chain],
    sq_closed: dict[str, bool],
    chain_needed: dict[str, bool],
    accepted: list[Chain],
    used_edges: set[str],
    seen_spine_seqs: set[tuple[str, ...]],
    p_store: dict[str, float],
    params: Params,
) -> tuple[list[Chain], list[Chain]]:
    """
    Append needed units to accepted; update p; mark sq closed.

    Dedup of units is only via spine_evidence_seq in S5 (exact spine copies
    never reach the judge). Here every judged unit records its seq; every
    needed=true unit is accepted even if edge_keys overlap prior accepts.
    used_edges still tracks accepted edges for p_reject protection.

    Returns (newly_accepted, rejected_chains).
    """
    if not any(chain_needed.values()):
        sq_closed = {k: False for k in sq_closed}

    for sq in all_sqs:
        if sq_closed.get(sq.id):
            sq.closed = True

    newly: list[Chain] = []
    rejected: list[Chain] = []
    next_i = len(accepted) + 1

    for c in chains:
        seq = c.spine_evidence_seq()
        seen_spine_seqs.add(seq)

        if chain_needed.get(c.chain_id, False):
            cloned = Chain(
                chain_id=f"a{next_i}",
                edge_keys=list(c.edge_keys),
                score=c.score,
                source_graph=c.source_graph,
                source_graphs=list(c.source_graphs),
                edges=list(c.edges),
                fans={h: list(fl) for h, fl in c.fans.items()},
                fan_hub_names=dict(c.fan_hub_names),
                text=c.format_unit(f"a{next_i}"),
            )
            next_i += 1
            accepted.append(cloned)
            newly.append(cloned)
            used_edges.update(cloned.all_edge_keys())
        else:
            rejected.append(c)

    for c in newly:
        apply_p_event(
            p_store,
            c.all_edge_keys(),
            params.p_accept,
            floor=params.p_floor,
            protected=None,
        )
    protected = set(used_edges)
    for c in rejected:
        apply_p_event(
            p_store,
            c.all_edge_keys(),
            params.p_reject,
            floor=params.p_floor,
            protected=protected,
        )

    return newly, rejected

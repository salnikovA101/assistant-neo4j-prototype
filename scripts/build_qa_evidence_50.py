#!/usr/bin/env python3
"""Mine grounded evidence packs for QA dataset (no remote LLM).

Pipeline (anti-hallucination):
  1) Index dump JSON
  2) Mine evidence packs deterministically (easy/medium/hard)
  3) Questions are written by Cursor agents / human from packs only
     (see tests/qa_build/pack_batch_*.json → questions_batch_*.json)
  4) Validate + assemble → tests/qa_evidence_50.json

Default: --mine-only (does not call OpenRouter or any API).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DUMP = ROOT / "tests" / "db_dump_full_corpus_20260713.json"
OUT_DIR = ROOT / "tests" / "qa_build"
FINAL_OUT = ROOT / "tests" / "qa_evidence_50.json"

logger = logging.getLogger("build_qa_evidence")

THEME_RE = re.compile(
    r"spoil|fresh|volatil|ammonia|amine|tvbn|h2s|sulfide|packag|indicator|"
    r"color|colour|sensor|metabolite|biogenic|histamine|putrescine|cadaverine|"
    r"trimethylamine|lactic|ferment|shelf.?life|storage|peptide|casein|"
    r"antimicrobial|inhibits|kefir|dairy|milk|cheese|pathogen|bacteria|"
    r"gaba|glutamate|proteoly|VOC|organic acid|pH|acid",
    re.I,
)

BLOCKLIST_RE = re.compile(
    r"\bolivier\b|\bolivye\b|\bsalat oliv\b|\biceberg lettuce\b|"
    r"\bpotassium permanganate\b|\bpermanganate\b",
    re.I,
)

JUNK_EV_RE = re.compile(
    r"^(review|article|supplementary|figure|table|http|doi:|"
    r"microbial community variations|the many faces of)",
    re.I,
)

QUOTAS = {"easy": 20, "medium": 15, "hard": 15}
TIER_RANGE = {
    "easy": (1, 5),
    "medium": (6, 15),
    "hard": (16, 30),
}


@dataclass
class EdgeRec:
    edge_id: str
    rel_type: str
    start_name: str
    end_name: str
    evidence: str
    chunk_id: str
    source_file: str


@dataclass
class Candidate:
    candidate_id: str
    tier: str
    theme: str
    evidence: list[str]
    entities: list[str]
    edge_ids: list[str]
    source_files: list[str]
    chunk_ids: list[str]
    question: str = ""
    reject_reason: str = ""


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _is_junk_evidence(ev: str) -> bool:
    e = _norm(ev)
    if len(e) < 25:
        return True
    if len(e.split()) < 5:
        return True
    if JUNK_EV_RE.search(e):
        return True
    # title-like: mostly Capitalized, no verb-ish lowercase run
    if e.endswith(".pdf"):
        return True
    return False


def load_edges(dump_path: Path) -> list[EdgeRec]:
    raw = json.loads(dump_path.read_text(encoding="utf-8"))
    nodes = {n["id"]: n for n in raw["nodes"]}

    def name(nid: str) -> str:
        p = (nodes.get(nid) or {}).get("properties") or {}
        return _norm(str(p.get("name") or ""))

    edges: list[EdgeRec] = []
    for r in raw["relationships"]:
        p = r.get("properties") or {}
        ev = _norm(str(p.get("evidence") or ""))
        if not ev or _is_junk_evidence(ev):
            continue
        sf = p.get("source_file")
        if not sf:
            sfs = p.get("source_files") or []
            sf = sfs[0] if sfs else ""
        edges.append(
            EdgeRec(
                edge_id=str(r.get("id") or ""),
                rel_type=str(r.get("type") or ""),
                start_name=name(r["start"]),
                end_name=name(r["end"]),
                evidence=ev,
                chunk_id=_norm(str(p.get("chunk_id") or "")),
                source_file=_norm(str(sf or "")),
            )
        )
    return edges


def build_index(edges: list[EdgeRec]) -> dict[str, Any]:
    by_chunk: dict[str, list[EdgeRec]] = defaultdict(list)
    by_source: dict[str, list[EdgeRec]] = defaultdict(list)
    by_entity: dict[str, list[EdgeRec]] = defaultdict(list)
    evidence_set = set()
    for e in edges:
        evidence_set.add(e.evidence)
        if e.chunk_id:
            by_chunk[e.chunk_id].append(e)
        if e.source_file:
            by_source[e.source_file].append(e)
        if e.start_name:
            by_entity[e.start_name.lower()].append(e)
        if e.end_name:
            by_entity[e.end_name.lower()].append(e)
    return {
        "edges": edges,
        "by_chunk": dict(by_chunk),
        "by_source": dict(by_source),
        "by_entity": dict(by_entity),
        "evidence_set": evidence_set,
    }


def _theme_score(edges: list[EdgeRec]) -> tuple[int, str]:
    blob = " ".join(
        f"{e.start_name} {e.rel_type} {e.end_name} {e.evidence}" for e in edges
    )
    hits = THEME_RE.findall(blob)
    if not hits:
        return 0, "general"
    top = Counter(h.lower() for h in hits).most_common(1)[0][0]
    return len(hits), top


def _unique_evidence(edges: list[EdgeRec]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for e in edges:
        if e.evidence not in seen:
            seen.add(e.evidence)
            out.append(e.evidence)
    return out


def _entities(edges: list[EdgeRec]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for e in edges:
        for n in (e.start_name, e.end_name):
            if n and n.lower() not in seen:
                seen.add(n.lower())
                names.append(n)
    return names


def _cid(tier: str, evidence: list[str]) -> str:
    h = hashlib.sha1("||".join(evidence).encode("utf-8")).hexdigest()[:10]
    return f"{tier}_{h}"


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def mine_candidates(index: dict[str, Any], target_pool: int = 120) -> list[Candidate]:
    edges: list[EdgeRec] = index["edges"]
    by_chunk: dict[str, list[EdgeRec]] = index["by_chunk"]
    by_source: dict[str, list[EdgeRec]] = index["by_source"]
    by_entity: dict[str, list[EdgeRec]] = index["by_entity"]

    cands: list[Candidate] = []
    seen_ids: set[str] = set()

    def add(tier: str, pack: list[EdgeRec], theme: str) -> None:
        ev = _unique_evidence(pack)
        lo, hi = TIER_RANGE[tier]
        if not (lo <= len(ev) <= hi):
            return
        # trim hard packs to <=30 unique by keeping highest theme edges first
        if len(ev) > 30:
            pack = pack[:80]
            ev = _unique_evidence(pack)[:30]
            # rebuild pack to match evidence
            keep = set(ev)
            pack = [e for e in pack if e.evidence in keep]
            ev = _unique_evidence(pack)
            if not (lo <= len(ev) <= hi):
                return
        cid = _cid(tier, ev)
        if cid in seen_ids:
            return
        # diversity vs existing
        for c in cands:
            if c.tier == tier and _jaccard(c.evidence, ev) > 0.7:
                return
        seen_ids.add(cid)
        score, auto_theme = _theme_score(pack)
        cands.append(
            Candidate(
                candidate_id=cid,
                tier=tier,
                theme=theme or auto_theme,
                evidence=ev,
                entities=_entities(pack)[:40],
                edge_ids=[e.edge_id for e in pack if e.edge_id][:80],
                source_files=sorted({e.source_file for e in pack if e.source_file}),
                chunk_ids=sorted({e.chunk_id for e in pack if e.chunk_id}),
            )
        )

    # --- EASY: single edge or 2-hop path ---
    # Prefer thematic edges
    thematic = [e for e in edges if THEME_RE.search(f"{e.start_name} {e.end_name} {e.evidence}")]
    pool = thematic if len(thematic) > 200 else edges
    # single-edge easy
    for e in pool:
        if e.rel_type in {"PRODUCES", "INHIBITS", "REQUIRES", "STIMULATES", "CONSUMES"}:
            add("easy", [e], e.rel_type.lower())
        if len([c for c in cands if c.tier == "easy"]) >= 50:
            break

    # 2-hop easy: share endpoint name
    by_node: dict[str, list[EdgeRec]] = defaultdict(list)
    for e in thematic[:2500]:
        if e.start_name:
            by_node[e.start_name.lower()].append(e)
        if e.end_name:
            by_node[e.end_name.lower()].append(e)
    for node, elist in list(by_node.items()):
        if len(elist) < 2:
            continue
        # take two edges with different evidence
        elist = sorted(elist, key=lambda x: x.evidence)
        for i in range(min(4, len(elist))):
            for j in range(i + 1, min(8, len(elist))):
                e1, e2 = elist[i], elist[j]
                if e1.evidence == e2.evidence:
                    continue
                pack = [e1, e2]
                # maybe add one more distinct
                for e3 in elist[j + 1 : j + 4]:
                    if e3.evidence not in {e1.evidence, e2.evidence}:
                        pack.append(e3)
                        break
                add("easy", pack, "path")
        if len([c for c in cands if c.tier == "easy"]) >= 80:
            break

    # --- MEDIUM: chunk packs ---
    chunk_items = sorted(by_chunk.items(), key=lambda kv: len(_unique_evidence(kv[1])), reverse=True)
    for cid, elist in chunk_items:
        evn = len(_unique_evidence(elist))
        score, theme = _theme_score(elist)
        if score < 2:
            continue
        if 6 <= evn <= 15:
            add("medium", elist, theme)
        elif evn > 15:
            # take thematic subset
            themed = [
                e
                for e in elist
                if THEME_RE.search(f"{e.start_name} {e.end_name} {e.evidence}")
            ]
            if len(_unique_evidence(themed)) >= 6:
                add("medium", themed[:40], theme)
        if len([c for c in cands if c.tier == "medium"]) >= 50:
            break

    # entity neighborhoods medium
    for ename, elist in sorted(by_entity.items(), key=lambda kv: len(kv[1]), reverse=True)[:400]:
        score, theme = _theme_score(elist)
        if score < 3:
            continue
        # diversify evidence
        uniq_edges: list[EdgeRec] = []
        seen_ev: set[str] = set()
        for e in elist:
            if e.evidence in seen_ev:
                continue
            seen_ev.add(e.evidence)
            uniq_edges.append(e)
            if len(uniq_edges) >= 15:
                break
        n = len(_unique_evidence(uniq_edges))
        if 6 <= n <= 15:
            add("medium", uniq_edges, theme)
        if len([c for c in cands if c.tier == "medium"]) >= 70:
            break

    # --- HARD: large chunks or source unions ---
    for cid, elist in chunk_items:
        ev = _unique_evidence(elist)
        score, theme = _theme_score(elist)
        if score < 4:
            continue
        if 16 <= len(ev) <= 30:
            add("hard", elist, theme)
        elif len(ev) > 30:
            # keep first 30 unique by walking edges
            keep: list[EdgeRec] = []
            seen: set[str] = set()
            # prefer thematic
            ordered = sorted(
                elist,
                key=lambda e: (
                    0 if THEME_RE.search(f"{e.start_name} {e.end_name} {e.evidence}") else 1,
                    e.evidence,
                ),
            )
            for e in ordered:
                if e.evidence in seen:
                    continue
                seen.add(e.evidence)
                keep.append(e)
                if len(seen) >= 28:
                    break
            add("hard", keep, theme)
        if len([c for c in cands if c.tier == "hard"]) >= 40:
            break

    # source-file thematic unions for hard
    for sf, elist in sorted(by_source.items(), key=lambda kv: len(kv[1]), reverse=True):
        score, theme = _theme_score(elist)
        if score < 8:
            continue
        # sample connected-ish: pick top entities in this source and their edges
        ent_count: Counter[str] = Counter()
        for e in elist:
            ent_count[e.start_name.lower()] += 1
            ent_count[e.end_name.lower()] += 1
        top_ents = {n for n, _ in ent_count.most_common(12)}
        sub = [
            e
            for e in elist
            if e.start_name.lower() in top_ents or e.end_name.lower() in top_ents
        ]
        # unique evidence cap
        keep: list[EdgeRec] = []
        seen: set[str] = set()
        for e in sorted(
            sub,
            key=lambda x: (
                0 if THEME_RE.search(f"{x.start_name} {x.end_name} {x.evidence}") else 1,
                x.evidence,
            ),
        ):
            if e.evidence in seen:
                continue
            seen.add(e.evidence)
            keep.append(e)
            if len(seen) >= 28:
                break
        add("hard", keep, theme)
        if len([c for c in cands if c.tier == "hard"]) >= 60:
            break

    # balance pool
    by_tier: dict[str, list[Candidate]] = defaultdict(list)
    for c in cands:
        by_tier[c.tier].append(c)

    # prefer higher theme score order already roughly ok; shuffle within by hash
    selected: list[Candidate] = []
    for tier, need in (("easy", 45), ("medium", 35), ("hard", 35)):
        items = by_tier[tier]
        items.sort(key=lambda c: (-len(c.entities), c.candidate_id))
        selected.extend(items[:need])

    logger.info(
        "mined candidates: total=%s easy=%s medium=%s hard=%s",
        len(selected),
        sum(1 for c in selected if c.tier == "easy"),
        sum(1 for c in selected if c.tier == "medium"),
        sum(1 for c in selected if c.tier == "hard"),
    )
    return selected[:target_pool]


QUESTION_SYSTEM = """You write evaluation questions for a scientific knowledge graph.

Rules:
- English only.
- Use ONLY facts present in the provided EVIDENCE texts. Do not invent chemicals, organisms, products, regulations, or numbers.
- The question must be answerable using the evidence list alone.
- Style: domain scientific Q/A (food tech / fermentation / metabolites / antimicrobial peptides / spoilage markers / sensors when present).
- Prefer concrete entities named in the evidence.
- For easy: one focused fact or short relation.
- For medium: ask to list/relate several linked facts.
- For hard: multi-part compositional question requiring synthesizing many evidence snippets (markers, producers, effects, conditions) — still only from evidence.
- Do NOT mention olive salad, Olivier, potassium permanganate, or any entity absent from evidence.
- Output JSON only: {"question": "..."}
"""


async def generate_questions(
    candidates: list[Candidate],
    *,
    batch_size: int = 5,
    concurrency: int = 4,
) -> list[Candidate]:
    # local imports to keep script runnable for mining-only
    import sys

    sys.path.insert(0, str(ROOT))
    from server.algorithm.slm_utils import resolve_slm_base_url, resolve_tool_llm_profile
    from server.utils.config import load_config

    config = load_config()
    profile = resolve_tool_llm_profile(config)
    client = AsyncOpenAI(
        api_key=profile.api_key or "EMPTY",
        base_url=resolve_slm_base_url(profile.base_url),
    )
    model = profile.model
    sem = asyncio.Semaphore(concurrency)

    async def one(c: Candidate) -> Candidate:
        ents = ", ".join(c.entities[:15])
        ev_block = "\n".join(f"- {e}" for e in c.evidence)
        user = (
            f"TIER: {c.tier} (n_evidence={len(c.evidence)})\n"
            f"ENTITIES: {ents}\n"
            f"THEME: {c.theme}\n"
            f"EVIDENCE:\n{ev_block}\n\n"
            "Write one English question answerable only from EVIDENCE."
        )
        async with sem:
            try:
                params: dict[str, Any] = {
                    "model": model,
                    "temperature": 0.0,
                    "max_tokens": 400,
                    "messages": [
                        {"role": "system", "content": QUESTION_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                }
                if hasattr(profile, "think") and not profile.think:
                    params["reasoning_effort"] = "none"
                resp = await client.chat.completions.create(**params)
                raw = (resp.choices[0].message.content or "").strip()
                q = _parse_question(raw)
                if not q:
                    c.reject_reason = "parse_fail"
                else:
                    c.question = q
            except Exception as exc:  # noqa: BLE001
                c.reject_reason = f"llm_error:{exc}"
                logger.warning("%s llm failed: %s", c.candidate_id, exc)
        return c

    out: list[Candidate] = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        logger.info("LLM batch %s-%s / %s", i + 1, i + len(batch), len(candidates))
        done = await asyncio.gather(*[one(c) for c in batch])
        out.extend(done)
    return out


def _parse_question(raw: str) -> str:
    text = raw.strip()
    # strip fences
    if "```" in text:
        text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    try:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            data = json.loads(text[start : end + 1])
            q = _norm(str(data.get("question") or ""))
            if q:
                return q
    except json.JSONDecodeError:
        pass
    # fallback: first line with ?
    for line in text.splitlines():
        line = _norm(line.strip(" -*\"'"))
        if "?" in line and len(line) > 20:
            return line
    return ""


def validate_and_select(
    candidates: list[Candidate],
    evidence_set: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accepted: list[dict[str, Any]] = []
    rejects: Counter[str] = Counter()
    per_source: Counter[str] = Counter()
    selected: list[Candidate] = []

    # sort: prefer themed + more entities within tier
    order = sorted(
        candidates,
        key=lambda c: (
            0 if c.tier == "easy" else 1 if c.tier == "medium" else 2,
            -len(c.entities),
            -len(c.evidence),
            c.candidate_id,
        ),
    )

    tier_counts = {t: 0 for t in QUOTAS}

    for c in order:
        if tier_counts[c.tier] >= QUOTAS[c.tier]:
            continue
        if not c.question:
            rejects["no_question"] += 1
            continue
        if BLOCKLIST_RE.search(c.question):
            rejects["blocklist_question"] += 1
            continue
        # evidence membership
        bad = [e for e in c.evidence if e not in evidence_set]
        if bad:
            rejects["evidence_not_in_dump"] += 1
            continue
        lo, hi = TIER_RANGE[c.tier]
        n = len(c.evidence)
        if not (lo <= n <= hi):
            rejects["tier_count"] += 1
            continue
        # entity grounding: at least one entity name appears in question (case-insensitive)
        qlow = c.question.lower()
        hits = [e for e in c.entities if e.lower() in qlow]
        min_hits = 1 if c.tier == "easy" else (2 if c.tier == "medium" else 2)
        # soft: if no direct entity substring, allow if a distinctive token from evidence is in Q
        if len(hits) < min_hits:
            # try partial tokens length>=5 from entities
            soft = []
            for e in c.entities:
                toks = [t for t in re.split(r"[^a-zA-Z0-9]+", e.lower()) if len(t) >= 5]
                if any(t in qlow for t in toks):
                    soft.append(e)
            hits = soft
        if len(hits) < 1:
            rejects["no_entity_grounding"] += 1
            continue
        # diversity
        if any(_jaccard(c.evidence, s.evidence) > 0.5 for s in selected):
            rejects["diversity"] += 1
            continue
        # source cap
        primary = c.source_files[0] if c.source_files else ""
        if primary and per_source[primary] >= 4:
            rejects["source_cap"] += 1
            continue

        selected.append(c)
        tier_counts[c.tier] += 1
        if primary:
            per_source[primary] += 1
        accepted.append(
            {
                "id": f"q{len(accepted)}",
                "question": c.question,
                "evidence": c.evidence,
                "difficulty": c.tier,
                "n_evidence": len(c.evidence),
            }
        )

    report = {
        "accepted": len(accepted),
        "tier_counts": tier_counts,
        "rejects": dict(rejects),
        "sources_used": dict(per_source),
        "shortfall": {t: QUOTAS[t] - tier_counts[t] for t in QUOTAS},
    }
    return accepted, report


def fill_shortfall_template(
    index: dict[str, Any],
    accepted: list[dict[str, Any]],
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    """Deterministic English template questions if LLM shortfall (still grounded)."""
    edges: list[EdgeRec] = index["edges"]
    have = {a["id"] for a in accepted}
    used_ev = [a["evidence"] for a in accepted]
    out = list(accepted)

    def ok_new(ev: list[str], tier: str) -> bool:
        lo, hi = TIER_RANGE[tier]
        if not (lo <= len(ev) <= hi):
            return False
        if any(_jaccard(ev, u) > 0.5 for u in used_ev):
            return False
        return True

    short = report.get("shortfall") or {}
    # easy templates from single PRODUCES/INHIBITS
    if short.get("easy", 0) > 0:
        for e in edges:
            if short["easy"] <= 0:
                break
            if e.rel_type not in {"PRODUCES", "INHIBITS", "REQUIRES", "STIMULATES"}:
                continue
            if not e.start_name or not e.end_name:
                continue
            ev = [e.evidence]
            if not ok_new(ev, "easy"):
                continue
            if e.rel_type == "PRODUCES":
                q = f"What does {e.start_name} produce according to the evidence?"
            elif e.rel_type == "INHIBITS":
                q = f"What does {e.start_name} inhibit according to the evidence?"
            elif e.rel_type == "REQUIRES":
                q = f"What does {e.start_name} require according to the evidence?"
            else:
                q = f"What does {e.start_name} stimulate according to the evidence?"
            # mention end entity for grounding
            if e.end_name.lower() not in q.lower():
                q = (
                    f"According to the evidence, how is {e.start_name} related to "
                    f"{e.end_name} via {e.rel_type.lower()}?"
                )
            item = {
                "id": f"q{len(out)}",
                "question": q,
                "evidence": ev,
                "difficulty": "easy",
                "n_evidence": 1,
            }
            out.append(item)
            used_ev.append(ev)
            short["easy"] -= 1

    # medium/hard from chunks
    by_chunk = index["by_chunk"]
    for tier in ("medium", "hard"):
        need = short.get(tier, 0)
        if need <= 0:
            continue
        for cid, elist in sorted(
            by_chunk.items(), key=lambda kv: len(_unique_evidence(kv[1])), reverse=True
        ):
            if need <= 0:
                break
            score, theme = _theme_score(elist)
            if score < 2:
                continue
            ev = _unique_evidence(elist)
            lo, hi = TIER_RANGE[tier]
            if len(ev) > hi:
                ev = ev[:hi]
            if not ok_new(ev, tier):
                continue
            ents = _entities(elist)[:8]
            if len(ents) < 2:
                continue
            if tier == "medium":
                q = (
                    f"Based on the evidence, what relationships connect "
                    f"{ents[0]}, {ents[1]}"
                    + (f", and {ents[2]}" if len(ents) > 2 else "")
                    + f" in the context of {theme}?"
                )
            else:
                named = ", ".join(ents[:5])
                q = (
                    f"Synthesize the evidence about {named}: which producers, "
                    f"products/metabolites, inhibitory or stimulatory effects, "
                    f"and conditions are described?"
                )
            out.append(
                {
                    "id": f"q{len(out)}",
                    "question": q,
                    "evidence": ev,
                    "difficulty": tier,
                    "n_evidence": len(ev),
                }
            )
            used_ev.append(ev)
            need -= 1
        short[tier] = need

    # re-id sequentially
    for i, item in enumerate(out):
        item["id"] = f"q{i}"
    return out


async def async_main(args: argparse.Namespace) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dump = Path(args.dump)
    logger.info("loading dump %s", dump)
    edges = load_edges(dump)
    index = build_index(edges)
    logger.info(
        "edges_kept=%s unique_evidence=%s chunks=%s sources=%s",
        len(edges),
        len(index["evidence_set"]),
        len(index["by_chunk"]),
        len(index["by_source"]),
    )

    # persist light index stats
    stats = {
        "edges_kept": len(edges),
        "unique_evidence": len(index["evidence_set"]),
        "chunks": len(index["by_chunk"]),
        "sources": len(index["by_source"]),
        "rel_types": dict(Counter(e.rel_type for e in edges)),
    }
    (OUT_DIR / "index_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )

    cands = mine_candidates(index, target_pool=args.pool)
    (OUT_DIR / "candidates.jsonl").write_text(
        "\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in cands),
        encoding="utf-8",
    )

    if args.mine_only:
        logger.info("mine-only done: %s candidates", len(cands))
        return

    cands = await generate_questions(
        cands, batch_size=args.batch_size, concurrency=args.concurrency
    )
    (OUT_DIR / "candidates_with_questions.jsonl").write_text(
        "\n".join(json.dumps(asdict(c), ensure_ascii=False) for c in cands),
        encoding="utf-8",
    )

    accepted, report = validate_and_select(cands, index["evidence_set"])
    logger.info("after LLM validate: %s", report)

    if any(v > 0 for v in (report.get("shortfall") or {}).values()):
        accepted = fill_shortfall_template(index, accepted, report)
        # recompute shortfall
        tc = Counter(a["difficulty"] for a in accepted)
        report["tier_counts_after_fill"] = dict(tc)
        report["filled"] = True

    # enforce exact 50 with quotas if possible
    final: list[dict[str, Any]] = []
    by_t: dict[str, list] = defaultdict(list)
    for a in accepted:
        by_t[a["difficulty"]].append(a)
    for tier, n in QUOTAS.items():
        final.extend(by_t[tier][:n])
    for i, item in enumerate(final):
        item["id"] = f"q{i}"

    FINAL_OUT.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    report["final_count"] = len(final)
    report["final_tiers"] = dict(Counter(x["difficulty"] for x in final))
    report["final_n_evidence"] = {
        "min": min((x["n_evidence"] for x in final), default=0),
        "max": max((x["n_evidence"] for x in final), default=0),
        "mean": round(sum(x["n_evidence"] for x in final) / max(len(final), 1), 2),
    }
    (OUT_DIR / "audit_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    # markdown audit
    lines = [
        "# QA evidence-50 audit",
        "",
        f"Final: **{len(final)}** questions",
        f"Tiers: `{report.get('final_tiers')}`",
        f"n_evidence: `{report.get('final_n_evidence')}`",
        "",
        "## Rejects",
        "```json",
        json.dumps(report.get("rejects"), indent=2),
        "```",
        "",
        "## Samples",
    ]
    for tier in ("easy", "medium", "hard"):
        sample = next((x for x in final if x["difficulty"] == tier), None)
        if not sample:
            continue
        lines.append(f"### {tier} — {sample['id']}")
        lines.append(f"**Q:** {sample['question']}")
        lines.append(f"**n_evidence:** {sample['n_evidence']}")
        lines.append("**evidence[0]:** " + sample["evidence"][0][:240])
        lines.append("")
    (OUT_DIR / "audit_report.md").write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote %s (%s items)", FINAL_OUT, len(final))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=str, default=str(DEFAULT_DUMP))
    ap.add_argument("--pool", type=int, default=110)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument(
        "--mine-only",
        action="store_true",
        default=True,
        help="Only mine packs (default). Do not call any remote LLM.",
    )
    ap.add_argument(
        "--with-remote-llm",
        action="store_true",
        help="Optional legacy path: call tool_profile LLM (discouraged).",
    )
    args = ap.parse_args()
    if args.with_remote_llm:
        args.mine_only = False
    else:
        args.mine_only = True
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Cosine: subquestions (sq_cache) × gold evidence via OpenRouter nemotron (v6 embed).

Examples:
  .venv/bin/python scripts/sq_gold_cosine.py \\
    --qa tests/qa_evidence_50.json --qids q0,q1,q2,q3,q4 \\
    --out-stem sq_gold_cosine_q0_q4

  .venv/bin/python scripts/sq_gold_cosine.py \\
    --qa tests/qa_open_20.json --difficulty hard --limit 5 \\
    --out-stem sq_gold_cosine_open_hard5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server.algorithm.embed import _EMBED_BACKEND, _EMBED_MODEL
from server.algorithm.embed_client import get_embeddings_batch

DEFAULT_SQ_CACHE = ROOT / "tests" / "reports" / "v6" / "sq_cache.json"
DEFAULT_OUT_DIR = ROOT / "tests" / "reports" / "v6"


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return float("nan")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return dot / (na * nb)


async def embed_unique(texts: list[str]) -> dict[str, list[float]]:
    uniq: list[str] = []
    seen: set[str] = set()
    for t in texts:
        key = t.strip()
        if key and key not in seen:
            seen.add(key)
            uniq.append(t)
    vectors = await get_embeddings_batch(
        uniq,
        backend=_EMBED_BACKEND,
        model_id=_EMBED_MODEL,
    )
    if len(vectors) != len(uniq):
        raise RuntimeError(f"embed count mismatch: {len(vectors)} != {len(uniq)}")
    out: dict[str, list[float]] = {}
    for t, v in zip(uniq, vectors):
        if not v:
            raise RuntimeError(f"empty embedding for text: {t[:120]!r}")
        out[t.strip()] = v
    return out


def fmt_sim(x: float) -> str:
    if math.isnan(x):
        return "nan"
    return f"{x:+.4f}"


def resolve_qids(
    qa_items: list[dict[str, Any]],
    *,
    qids: list[str] | None,
    difficulty: str | None,
    limit: int | None,
) -> list[str]:
    if qids:
        by_id = {str(it["id"]) for it in qa_items}
        missing = [q for q in qids if q not in by_id]
        if missing:
            raise SystemExit(f"qids not in QA file: {missing}")
        return qids
    selected = qa_items
    if difficulty:
        selected = [it for it in selected if it.get("difficulty") == difficulty]
        if not selected:
            raise SystemExit(f"no items with difficulty={difficulty!r}")
    if limit is not None:
        selected = selected[:limit]
    return [str(it["id"]) for it in selected]


async def run(
    *,
    qa_path: Path,
    sq_cache_path: Path,
    out_md: Path,
    out_json: Path,
    qids: list[str],
    title: str,
) -> None:
    qa_items: list[dict[str, Any]] = json.loads(qa_path.read_text(encoding="utf-8"))
    by_id = {str(it["id"]): it for it in qa_items}
    sq_cache: dict[str, Any] = json.loads(sq_cache_path.read_text(encoding="utf-8"))

    missing_sq = [qid for qid in qids if qid not in sq_cache]
    if missing_sq:
        raise SystemExit(f"missing sq_cache entries: {missing_sq}")

    all_texts: list[str] = []
    for qid in qids:
        item = by_id[qid]
        entry = sq_cache[qid]
        for ev in item["evidence"]:
            all_texts.append(str(ev))
        for sq in entry["subquestions"]:
            all_texts.append(str(sq["text"]))

    emb = await embed_unique(all_texts)

    report: dict[str, Any] = {
        "model": _EMBED_MODEL,
        "backend": str(_EMBED_BACKEND),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "qa_path": str(qa_path.relative_to(ROOT)),
        "qids": qids,
        "questions": [],
    }

    md_lines: list[str] = [
        f"# {title}",
        "",
        f"model: `{_EMBED_MODEL}`  ",
        f"backend: `{_EMBED_BACKEND}`  ",
        f"qa: `{report['qa_path']}`  ",
        f"qids: `{', '.join(qids)}`  ",
        f"generated_at: `{report['generated_at']}`",
        "",
        "Для каждого gold evidence — cosine с каждым подвопросом из `sq_cache.json`.",
        "",
    ]

    for qid in qids:
        item = by_id[qid]
        entry = sq_cache[qid]
        sqs = [
            {"id": str(s["id"]), "text": str(s["text"])}
            for s in entry["subquestions"]
        ]
        evidences = [str(e) for e in item["evidence"]]

        q_block: dict[str, Any] = {
            "id": qid,
            "difficulty": item.get("difficulty"),
            "question": item["question"],
            "subquestions": sqs,
            "gold": [],
        }

        md_lines.append(f"## {qid}")
        md_lines.append("")
        if item.get("difficulty"):
            md_lines.append(f"**difficulty:** {item['difficulty']}")
            md_lines.append("")
        md_lines.append(f"**question:** {item['question']}")
        md_lines.append("")
        md_lines.append("**subquestions:**")
        for sq in sqs:
            md_lines.append(f"- `{sq['id']}`: {sq['text']}")
        md_lines.append("")

        for i, ev in enumerate(evidences):
            ev_vec = emb[ev.strip()]
            sims: dict[str, float] = {}
            for sq in sqs:
                sims[sq["id"]] = cosine(ev_vec, emb[sq["text"].strip()])
            best_id = max(sims, key=lambda k: sims[k])
            q_block["gold"].append(
                {
                    "index": i,
                    "evidence": ev,
                    "similarities": {k: round(v, 6) for k, v in sims.items()},
                    "best_sq": best_id,
                    "best_sim": round(sims[best_id], 6),
                }
            )

            md_lines.append(f"### gold[{i}]")
            md_lines.append("")
            md_lines.append(f"evidence: {ev}")
            md_lines.append("")
            ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
            for sq_id, sim in ranked:
                sq_text = next(s["text"] for s in sqs if s["id"] == sq_id)
                mark = " ← best" if sq_id == best_id else ""
                md_lines.append(
                    f"- `{sq_id}`: {fmt_sim(sim)}{mark}  — {sq_text}"
                )
            md_lines.append("")

        report["questions"].append(q_block)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md_lines).rstrip() + "\n", encoding="utf-8")
    out_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_md}")
    print(f"wrote {out_json}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--qa",
        type=Path,
        default=ROOT / "tests" / "qa_evidence_50.json",
        help="QA JSON path",
    )
    p.add_argument(
        "--sq-cache",
        type=Path,
        default=DEFAULT_SQ_CACHE,
        help="sq_cache JSON path",
    )
    p.add_argument(
        "--qids",
        type=str,
        default="",
        help="Comma-separated question ids (overrides --difficulty/--limit)",
    )
    p.add_argument(
        "--difficulty",
        type=str,
        default="",
        help="Filter by difficulty when --qids not set",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Take first N after filter when --qids not set",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory",
    )
    p.add_argument(
        "--out-stem",
        type=str,
        required=True,
        help="Output filename stem (writes .md and .json)",
    )
    p.add_argument(
        "--title",
        type=str,
        default="",
        help="Markdown H1 title",
    )
    return p.parse_args()


async def main() -> None:
    args = parse_args()
    qa_path = args.qa if args.qa.is_absolute() else ROOT / args.qa
    sq_cache_path = (
        args.sq_cache if args.sq_cache.is_absolute() else ROOT / args.sq_cache
    )
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir

    qa_items: list[dict[str, Any]] = json.loads(qa_path.read_text(encoding="utf-8"))
    qids_arg = [x.strip() for x in args.qids.split(",") if x.strip()] or None
    qids = resolve_qids(
        qa_items,
        qids=qids_arg,
        difficulty=args.difficulty or None,
        limit=args.limit,
    )
    title = args.title or f"SQ × gold evidence cosine ({', '.join(qids)})"
    await run(
        qa_path=qa_path,
        sq_cache_path=sq_cache_path,
        out_md=out_dir / f"{args.out_stem}.md",
        out_json=out_dir / f"{args.out_stem}.json",
        qids=qids,
        title=title,
    )


if __name__ == "__main__":
    asyncio.run(main())

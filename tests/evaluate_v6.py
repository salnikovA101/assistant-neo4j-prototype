"""V6 evaluation harness: mock decompose → run → gold metrics + full MD trace.

Primary metrics match on evidence *text* (unique stripped strings), not edge_keys.

Default dataset: tests/qa_open_20.json. Close pack: tests/qa_evidence_50.json
({id, question, evidence[], ...}). Legacy gold_edge_keys still supported.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import time
from collections.abc import Iterable
from dataclasses import fields
from pathlib import Path
from typing import Any

from server.algorithm.cypher.edges import fetch_evidences_for_edge_keys
from server.algorithm.edge_keys import parse_edge_key
from server.algorithm.graph_cache import (
    bundle_matches,
    load_graph_cache_entry,
    save_graph_cache_entry,
)
from server.algorithm.params import Params, merge_params
from server.algorithm.pipeline import run
from server.core.db import close_driver, init_driver
from server.utils.config import load_config
from tests.algorithm.mock_decompose import mock_decompose

logger = logging.getLogger(__name__)

DEFAULT_DATASET = Path("tests/qa_open_20.json")
DEFAULT_OUT_DIR = Path("tests/reports/v6")
DEFAULT_SQ_CACHE = Path("tests/reports/v6_cache/sq_cache.json")
DEFAULT_GRAPH_CACHE = Path("tests/reports/v6_cache/graphs")
VECTOR_BASELINE_K = 150


def _reset_out_dir(out_dir: Path) -> None:
    """Wipe report dir for a fresh run; never touch v6_cache."""
    resolved = out_dir.resolve()
    if "v6_cache" in resolved.parts:
        raise ValueError(f"Refusing to wipe cache path as out-dir: {out_dir}")
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Reset out-dir %s (cache stays under tests/reports/v6_cache)", out_dir)


def _norm_evidence(text: str | None) -> str:
    return (text or "").strip()


def _parse_param_value(key: str, raw: str) -> Any:
    """Coerce --set VALUE to the Params field type."""
    typ = {f.name: f.type for f in fields(Params)}.get(key)
    if typ is bool or typ == "bool":
        v = raw.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
        raise SystemExit(f"--set {key} expects bool, got {raw!r}")
    if typ is int or typ == "int":
        return int(raw)
    if typ is float or typ == "float":
        return float(raw)
    return raw


def validate_evidence_dataset(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate qa_evidence_50-style packs: question + non-empty evidence[]."""
    errors: list[str] = []
    warnings: list[str] = []
    n_ev = 0
    for i, item in enumerate(items):
        qid = str(item.get("id") or f"q{i}")
        if not str(item.get("question") or "").strip():
            errors.append(f"{qid}: missing question")
        evid = item.get("evidence") or []
        if not isinstance(evid, list) or not evid:
            errors.append(f"{qid}: missing evidence[]")
            continue
        seen: set[str] = set()
        for j, e in enumerate(evid):
            t = _norm_evidence(e if isinstance(e, str) else str(e))
            if not t:
                errors.append(f"{qid}: empty evidence[{j}]")
                continue
            if t in seen:
                warnings.append(f"{qid}: duplicate evidence text")
            seen.add(t)
            n_ev += 1
        n_field = item.get("n_evidence")
        if n_field is not None and int(n_field) != len(evid):
            warnings.append(f"{qid}: n_evidence={n_field} != len(evidence)={len(evid)}")
    return {
        "ok": not errors,
        "n_items": len(items),
        "n_evidences": n_ev,
        "errors": errors,
        "warnings": warnings,
    }


def validate_dataset(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Legacy gold_edge_keys packs (when evidence[] is absent)."""
    errors: list[str] = []
    warnings: list[str] = []
    n_keys = 0
    for i, item in enumerate(items):
        qid = str(item.get("id") or f"q{i}")
        if not str(item.get("question") or "").strip():
            errors.append(f"{qid}: missing question")
        keys = list(item.get("gold_edge_keys") or [])
        if not keys:
            errors.append(f"{qid}: missing gold_edge_keys")
            continue
        seen: set[str] = set()
        for k in keys:
            try:
                parse_edge_key(k)
            except ValueError as e:
                errors.append(f"{qid}: invalid edge_key {k!r}: {e}")
            if k in seen:
                errors.append(f"{qid}: duplicate gold edge_key {k}")
            seen.add(k)
            n_keys += 1
    return {
        "ok": not errors,
        "n_items": len(items),
        "n_keys": n_keys,
        "errors": errors,
        "warnings": warnings,
    }


def _dataset_has_evidence_packs(items: list[dict[str, Any]]) -> bool:
    return any(isinstance(it.get("evidence"), list) and it.get("evidence") for it in items)


def _gold_evidences_from_item(item: dict[str, Any]) -> set[str]:
    """Prefer explicit evidence[] texts; empty if only gold_edge_keys (resolve later)."""
    raw = item.get("evidence")
    if not isinstance(raw, list) or not raw:
        return set()
    return {_norm_evidence(e) for e in raw if _norm_evidence(e)}


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        )
    except Exception:
        return "unknown"


_DIFFICULTY_EFFORT = {
    "easy": "low",
    "medium": "medium",
    "hard": "high",
}


def effort_for_item(item: dict[str, Any], effort_arg: str) -> str:
    """Map item difficulty → effort when --effort auto; else force CLI value."""
    arg = (effort_arg or "auto").strip().lower()
    if arg in ("low", "medium", "high"):
        return arg
    diff = str(item.get("difficulty") or "").strip().lower()
    return _DIFFICULTY_EFFORT.get(diff, "medium")


def mean_metrics(reports: list[dict[str, Any]]) -> dict[str, float | int]:
    n = len(reports)
    if not n:
        return {
            "n": 0,
            "mean_recall_accepted": 0.0,
            "mean_precision_accepted": 0.0,
            "mean_recall_s3_union": 0.0,
            "mean_recall_ann_union": 0.0,
            "mean_recall_rerank_union": 0.0,
            "mean_recall_vector_topk": 0.0,
            "mean_n_accepted": 0.0,
            "mean_n_accepted_all": 0.0,
        }
    return {
        "n": n,
        "mean_recall_accepted": sum(r["recall_accepted"] for r in reports) / n,
        "mean_precision_accepted": sum(r["precision_accepted"] for r in reports) / n,
        "mean_recall_s3_union": sum((r.get("recall_s3") or {}).get("union", 0.0) for r in reports) / n,
        "mean_recall_ann_union": sum(float(r.get("recall_ann_union") or 0.0) for r in reports) / n,
        "mean_recall_rerank_union": sum(float(r.get("recall_rerank_union") or 0.0) for r in reports) / n,
        "mean_recall_vector_topk": sum(float(r.get("recall_vector_topk") or 0.0) for r in reports) / n,
        "mean_n_accepted": sum(int(r.get("n_accepted") or 0) for r in reports) / n,
        "mean_n_accepted_all": sum(int(r.get("n_accepted_all") or r.get("n_accepted") or 0) for r in reports) / n,
    }


def metrics_by_difficulty(
    reports: list[dict[str, Any]],
) -> dict[str, dict[str, float | int]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "easy": [],
        "medium": [],
        "hard": [],
    }
    for r in reports:
        d = str(r.get("difficulty") or "").strip().lower()
        if d in buckets:
            buckets[d].append(r)
    return {d: mean_metrics(rs) for d, rs in buckets.items()}


def load_sq_cache(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("sq-cache load failed %s: %s", path, e)
        return {}
    return data if isinstance(data, dict) else {}


def save_sq_cache(path: Path, cache: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cache_get_sqs(cache: dict[str, Any], *, qid: str, question: str) -> list[dict[str, str]] | None:
    entry = cache.get(qid)
    if not isinstance(entry, dict):
        return None
    raw = entry.get("subquestions")
    if not isinstance(raw, list) or not raw:
        return None
    # Optional sanity: same question text
    cached_q = str(entry.get("question") or "")
    if cached_q and question and cached_q.strip() != question.strip():
        logger.warning(
            "sq-cache %s question mismatch; ignoring cache entry",
            qid,
        )
        return None
    out: list[dict[str, str]] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        sid = str(item.get("id") or f"sq{i + 1}")
        out.append({"id": sid, "text": text})
    return out or None


def cache_put_sqs(
    cache: dict[str, Any],
    *,
    qid: str,
    question: str,
    sqs: list[dict[str, str]],
) -> None:
    cache[qid] = {
        "question": question,
        "subquestions": [
            {"id": str(s.get("id") or f"sq{i + 1}"), "text": str(s.get("text") or "")}
            for i, s in enumerate(sqs)
            if str(s.get("text") or "").strip()
        ],
    }


async def resolve_subquestions(
    question: str,
    *,
    qid: str,
    sq_cache: dict[str, Any] | None,
    sq_cache_path: Path | None,
    sq_cache_read_only: bool,
) -> tuple[list[dict[str, str]], bool]:
    """
    Returns (subquestions, from_cache).
    If sq_cache is enabled: prefer file; on miss optionally call LLM and persist.
    """
    if sq_cache is not None:
        hit = cache_get_sqs(sq_cache, qid=qid, question=question)
        if hit is not None:
            logger.info("%s: subquestions from cache (%s)", qid, len(hit))
            return hit, True
        if sq_cache_read_only:
            raise RuntimeError(
                f"sq-cache miss for {qid!r} (read-only). Populate cache first without --sq-cache-read-only."
            )
        logger.info("%s: sq-cache miss → mock_decompose", qid)

    sqs = await mock_decompose(question)
    if sq_cache is not None and sq_cache_path is not None:
        cache_put_sqs(sq_cache, qid=qid, question=question, sqs=sqs)
        save_sq_cache(sq_cache_path, sq_cache)
        logger.info("%s: wrote %s subquestions → %s", qid, len(sqs), sq_cache_path)
    return sqs, False


def set_recall(pred: set[str], gold: set[str]) -> float:
    if not gold:
        return 0.0
    return len(pred & gold) / len(gold)


def set_precision(pred: set[str], gold: set[str]) -> float:
    if not pred:
        return 0.0
    return len(pred & gold) / len(pred)


def _accepted_edge_keys(result: dict[str, Any]) -> set[str]:
    """Spine + fan edge_keys from every accepted unit."""
    keys: set[str] = set()
    for c in result.get("accepted") or []:
        keys.update(_chain_all_edge_keys(c))
    return keys


def _accepted_evidences(result: dict[str, Any]) -> set[str]:
    """Evidence texts from spine and fans (both count toward final context)."""
    out: set[str] = set()
    for c in result.get("accepted") or []:
        out |= _one_chain_evidences(c)
    return out


def _one_chain_evidences(
    c: dict[str, Any],
    key_to_ev: dict[str, str] | None = None,
) -> set[str]:
    out: set[str] = set()
    for e in c.get("edges") or []:
        ev = _norm_evidence(e.get("evidence"))
        if ev:
            out.add(ev)
    for flist in (c.get("fans") or {}).values():
        for e in flist or []:
            if not isinstance(e, dict):
                continue
            ev = _norm_evidence(e.get("evidence"))
            if ev:
                out.add(ev)
    if key_to_ev:
        for k in _chain_all_edge_keys(c):
            ev = _norm_evidence(key_to_ev.get(k))
            if ev:
                out.add(ev)
    return out


RECALL_AT_KS: tuple[int, ...] = (1, 2, 3, 5, 10, 15, 20)


def metrics_at_k(
    ranked: list[dict[str, Any]],
    gold_ev: set[str],
    key_to_ev: dict[str, str] | None = None,
    ks: tuple[int, ...] = RECALL_AT_KS,
) -> dict[str, dict[str, float | int]]:
    """Cumulative recall/precision on score-sorted prefixes (assistant order)."""
    acc: set[str] = set()
    out: dict[str, dict[str, float | int]] = {}
    want = set(ks)
    for i, c in enumerate(ranked or [], start=1):
        acc |= _one_chain_evidences(c, key_to_ev)
        if i in want:
            out[str(i)] = {
                "recall": set_recall(acc, gold_ev),
                "precision": set_precision(acc, gold_ev),
                "n_paths": i,
                "n_pred": len(acc),
            }
    if ranked:
        last = {
            "recall": set_recall(acc, gold_ev),
            "precision": set_precision(acc, gold_ev),
            "n_paths": len(ranked),
            "n_pred": len(acc),
        }
        for k in ks:
            if str(k) not in out:
                row = dict(last)
                out[str(k)] = row
    return out


def mean_at_k(reports: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for k in RECALL_AT_KS:
        recs: list[float] = []
        precs: list[float] = []
        ns: list[float] = []
        for r in reports:
            row = (r.get("recall_at_k") or {}).get(str(k))
            if not row:
                continue
            recs.append(float(row["recall"]))
            precs.append(float(row["precision"]))
            ns.append(float(row.get("n_paths") or 0))
        if recs:
            out[str(k)] = {
                "mean_recall": sum(recs) / len(recs),
                "mean_precision": sum(precs) / len(precs),
                "mean_n_paths": sum(ns) / len(ns),
            }
    return out


def _evidences_from_key_map(keys: Iterable[str], key_to_ev: dict[str, str]) -> set[str]:
    out: set[str] = set()
    for k in keys:
        ev = _norm_evidence(key_to_ev.get(k))
        if ev:
            out.add(ev)
    return out


def n_gold_in_keys(
    keys: Iterable[str],
    gold_ev: set[str],
    key_to_ev: dict[str, str],
) -> int:
    """How many unique gold evidences appear among edge keys."""
    if not gold_ev:
        return 0
    return len(_evidences_from_key_map(keys, key_to_ev) & gold_ev)


def _gold_stage_coverage(
    *,
    ann_keys: dict[str, list[str]],
    rerank_keys: dict[str, list[str]],
    ann_keys_union: list[str],
    rerank_keys_union: list[str],
    gold_ev: set[str],
    key_to_ev: dict[str, str],
) -> dict[str, Any]:
    """Per-sq + union: pool sizes and gold counts before/after CE."""
    n_total = len(gold_ev)
    per_sq: dict[str, dict[str, int]] = {}
    sq_ids = sorted(set(ann_keys) | set(rerank_keys))
    for sid in sq_ids:
        akeys = list(ann_keys.get(sid) or [])
        rkeys = list(rerank_keys.get(sid) or [])
        per_sq[sid] = {
            "n_ann": len(akeys),
            "n_gold_ann": n_gold_in_keys(akeys, gold_ev, key_to_ev),
            "n_rerank": len(rkeys),
            "n_gold_rerank": n_gold_in_keys(rkeys, gold_ev, key_to_ev),
        }
    return {
        "n_gold_total": n_total,
        "per_sq": per_sq,
        "union": {
            "n_ann": len(ann_keys_union),
            "n_gold_ann": n_gold_in_keys(ann_keys_union, gold_ev, key_to_ev),
            "n_rerank": len(rerank_keys_union),
            "n_gold_rerank": n_gold_in_keys(rerank_keys_union, gold_ev, key_to_ev),
        },
    }


def _s3_evidence_recalls(
    s3_keys: dict[str, list[str]],
    union_keys: list[str],
    gold_ev: set[str],
    key_to_ev: dict[str, str],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for src, keys in (s3_keys or {}).items():
        pred = _evidences_from_key_map(keys, key_to_ev)
        out[src] = set_recall(pred, gold_ev)
    out["union"] = set_recall(_evidences_from_key_map(union_keys or [], key_to_ev), gold_ev)
    return out


def _chain_all_edge_keys(c: dict[str, Any]) -> list[str]:
    """Spine edge_keys + fan edge_keys (walk length after reshape)."""
    keys = list(c.get("edge_keys") or [])
    seen = set(keys)
    for flist in (c.get("fans") or {}).values():
        for e in flist or []:
            if not isinstance(e, dict):
                continue
            ek = e.get("edge_key")
            if ek and ek not in seen:
                keys.append(str(ek))
                seen.add(str(ek))
    return keys


def _chain_gold_counts(
    accepted: list[dict[str, Any]],
    gold_ev: set[str],
    key_to_ev: dict[str, str],
) -> list[dict[str, int]]:
    """Per accepted chain: walk length (spine+fans) and gold evidences inside it."""
    out: list[dict[str, int]] = []
    for c in accepted or []:
        keys = _chain_all_edge_keys(c)
        evs: set[str] = set()
        for e in c.get("edges") or []:
            ev = _norm_evidence(e.get("evidence"))
            if ev:
                evs.add(ev)
        for flist in (c.get("fans") or {}).values():
            for e in flist or []:
                if not isinstance(e, dict):
                    continue
                ev = _norm_evidence(e.get("evidence"))
                if ev:
                    evs.add(ev)
        for k in keys:
            ev = _norm_evidence(key_to_ev.get(k))
            if ev:
                evs.add(ev)
        n_gold = sum(1 for ev in evs if ev in gold_ev)
        out.append(
            {
                "len": len(keys),
                "spine_len": len(c.get("edge_keys") or []),
                "n_gold": n_gold,
            }
        )
    return out


def _vector_baseline_recall(
    ann_edge_sims: dict[str, float],
    k: int,
    gold_ev: set[str],
    key_to_ev: dict[str, str],
) -> float:
    """Flat vector baseline: top-K by cosine from first-iter ANN pool (pre-CE)."""
    if k <= 0 or not gold_ev:
        return 0.0
    ranked = sorted(
        (ann_edge_sims or {}).items(),
        key=lambda kv: kv[1],
        reverse=True,
    )
    top_keys = [ek for ek, _ in ranked[:k]]
    pred = _evidences_from_key_map(top_keys, key_to_ev)
    return set_recall(pred, gold_ev)


def _aggregate_chain_length_stats(
    chain_rows: list[dict[str, int]],
) -> dict[str, Any]:
    by_len: dict[int, list[int]] = {}
    for row in chain_rows:
        by_len.setdefault(int(row["len"]), []).append(int(row["n_gold"]))
    by_length: dict[str, dict[str, float | int]] = {}
    n_paths = 0
    sum_len = 0
    sum_gold = 0
    for L in sorted(by_len):
        golds = by_len[L]
        n = len(golds)
        s = sum(golds)
        by_length[str(L)] = {"n": n, "avg_gold": (s / n) if n else 0.0}
        n_paths += n
        sum_len += L * n
        sum_gold += s
    return {
        "by_length": by_length,
        "n_paths_total": n_paths,
        "avg_len": (sum_len / n_paths) if n_paths else 0.0,
        "avg_gold_per_path": (sum_gold / n_paths) if n_paths else 0.0,
    }


def _chain_brief(c: dict) -> str:
    if c.get("text"):
        return str(c["text"])
    parts = []
    for e in c.get("edges") or []:
        parts.append(f"{e.get('start')}-[{e.get('type')}]->{e.get('end')}: {e.get('evidence') or ''}")
    return " | ".join(parts) if parts else str(c.get("edge_keys"))


def write_trace_md(
    path: Path,
    *,
    question: str,
    qid: str,
    subquestions: list[dict],
    result: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    lines: list[str] = [
        f"# V6 eval trace — {qid}",
        "",
        "## Question",
        question,
        "",
        "## Mock subquestions",
    ]
    for s in subquestions:
        lines.append(f"- `{s.get('id')}`: {s.get('text')}")
    lines.append("")

    tr = result.get("trace") or {}
    if tr:
        lines.append("## S4 / S5")
        if tr.get("stop_reason"):
            lines.append(f"- stop_reason: `{tr.get('stop_reason')}`")
        lines.append(f"- s4_pool: {tr.get('s4_pool')}")
        lines.append("")
        lines.append("### S5 batch")
        for c in tr.get("batch") or []:
            cid = c.get("chain_id")
            lines.append(f"- **{cid}**: {_chain_brief(c)}")
        lines.append("")
        lines.append("### Accepted (pre-emit)")
        for c in tr.get("accepted") or []:
            lines.append(f"- `{c.get('chain_id')}` ({c.get('source_graph')}): {_chain_brief(c)}")
        lines.append("")
        lines.append("### source_graph")
        for c in tr.get("batch") or []:
            lines.append(
                f"- {c.get('chain_id')}: source_graph={c.get('source_graph')} source_graphs={c.get('source_graphs')}"
            )
        lines.append("")

    lines.append("## S3 evidence recall per subgraph + union")
    for k, v in (metrics.get("recall_s3") or {}).items():
        lines.append(f"- `{k}`: {v:.4f}")
    lines.append("")
    lines.append("## ANN / rerank / S3 recall")
    lines.append(f"- `ann_union` (full ANN pool, pre-CE): {float(metrics.get('recall_ann_union') or 0.0):.4f}")
    lines.append(
        f"- `vector_topk` (top-{int(metrics.get('vector_topk_k') or 0)} by sim "
        f"from ANN pool, no CE): "
        f"{float(metrics.get('recall_vector_topk') or 0.0):.4f}"
    )
    lines.append(f"- `rerank_union` (CE keep per sq): {float(metrics.get('recall_rerank_union') or 0.0):.4f}")
    lines.append(f"- `s3_union` (after CE + bridges): {float((metrics.get('recall_s3') or {}).get('union', 0.0)):.4f}")
    lines.append("")
    gc = metrics.get("gold_coverage") or {}
    n_tot = int(gc.get("n_gold_total") or 0)
    lines.append("## Gold coverage (ANN L_raw_max → after CE)")
    lines.append(f"- total gold: {n_tot}")
    for sid, row in (gc.get("per_sq") or {}).items():
        lines.append(
            f"- `{sid}`: ann {row.get('n_ann', 0)} → gold "
            f"{row.get('n_gold_ann', 0)}/{n_tot}; after CE "
            f"{row.get('n_rerank', 0)} → gold {row.get('n_gold_rerank', 0)}/{n_tot}"
        )
    u = gc.get("union") or {}
    lines.append(
        f"- `union`: ann {u.get('n_ann', 0)} → gold "
        f"{u.get('n_gold_ann', 0)}/{n_tot}; after CE "
        f"{u.get('n_rerank', 0)} → gold {u.get('n_gold_rerank', 0)}/{n_tot}"
    )
    lines.append("")
    pool = list(result.get("accepted_all") or [])
    emitted = list(result.get("accepted") or [])
    lines.append("## Score-sorted pool vs emit")
    lines.append(f"- pool (`accepted_all`): {len(pool)}")
    lines.append(f"- emitted (`accepted`): {len(emitted)}")
    if pool:
        lines.append("- pool order (best S4 score first):")
        for i, c in enumerate(pool, start=1):
            lines.append(
                f"  {i}. `{c.get('chain_id')}` score={float(c.get('score') or 0.0):.4f} "
                f"({c.get('source_graph')})"
            )
    lines.append("")
    lines.append("## Recall@k (score-sorted pool, before emit cut)")
    atk = metrics.get("recall_at_k") or {}
    for k in RECALL_AT_KS:
        row = atk.get(str(k)) or {}
        if not row:
            continue
        lines.append(
            f"- @{k}: recall={float(row.get('recall') or 0.0):.4f} "
            f"precision={float(row.get('precision') or 0.0):.4f} "
            f"n_paths={row.get('n_paths')} n_pred={row.get('n_pred')}"
        )
    lines.append("")
    lines.append("## Final accepted (emitted to assistant)")
    for c in emitted:
        lines.append(
            f"- `{c.get('chain_id')}` score={float(c.get('score') or 0.0):.4f} "
            f"({c.get('source_graph')}): {_chain_brief(c)}"
        )
    lines.append("")
    lines.append("## Metrics (primary = evidence text)")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2, ensure_ascii=False))
    lines.append("```")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


async def eval_one(
    driver,
    item: dict[str, Any],
    *,
    effort: str,
    params: Params,
    sq_cache: dict[str, Any] | None = None,
    sq_cache_path: Path | None = None,
    sq_cache_read_only: bool = False,
    graph_cache_dir: Path | None = None,
    graph_cache_read_only: bool = False,
) -> dict[str, Any]:
    qid = str(item.get("id") or item.get("qid") or "q")
    question = str(item.get("question") or item.get("query") or item.get("text") or "")
    gold_keys = set(item.get("gold_edge_keys") or [])
    gold_ev_direct = _gold_evidences_from_item(item)

    t0 = time.perf_counter()
    sqs, sq_from_cache = await resolve_subquestions(
        question,
        qid=qid,
        sq_cache=sq_cache,
        sq_cache_path=sq_cache_path,
        sq_cache_read_only=sq_cache_read_only,
    )

    s3_bundle: dict[str, Any] | None = None
    graph_from_cache = False
    if graph_cache_dir is not None:
        entry = load_graph_cache_entry(graph_cache_dir, qid)
        if entry is not None and bundle_matches(entry, question=question, params=params, sqs=sqs):
            s3_bundle = entry
            graph_from_cache = True
            logger.info("%s: S3 graphs from cache", qid)
        elif graph_cache_read_only:
            raise RuntimeError(
                f"graph-cache miss for {qid!r} (read-only). "
                f"Populate with --graph-cache first (without --graph-cache-read-only)."
            )
        elif entry is not None:
            logger.info("%s: graph-cache fingerprint mismatch → rebuild S1–S3", qid)
        else:
            logger.info("%s: graph-cache miss → run S1–S3", qid)

    result = await run(
        driver,
        subquestions=sqs,
        query=question,
        effort=effort,
        params=params,
        s3_bundle=s3_bundle,
        emit_s3_bundle=bool(graph_cache_dir is not None and not graph_from_cache),
        cache_qid=qid,
    )
    if graph_cache_dir is not None and not graph_from_cache and isinstance(result.get("s3_bundle"), dict):
        path = save_graph_cache_entry(graph_cache_dir, qid, result["s3_bundle"])
        logger.info("%s: wrote S3 graph-cache → %s", qid, path)
    # Keep report JSON lean
    result.pop("s3_bundle", None)
    elapsed = time.perf_counter() - t0

    pred_keys = _accepted_edge_keys(result)
    pred_ev = _accepted_evidences(result)
    ranked_all = list(result.get("accepted_all") or result.get("accepted") or [])

    s3_keys = result.get("s3_keys") or {}
    union_keys = list(result.get("s3_keys_union") or [])
    ann_edge_sims = result.get("ann_edge_sims") or {}
    ann_keys = result.get("ann_keys") or {}
    rerank_keys = result.get("rerank_keys") or {}
    ann_keys_union = list(result.get("ann_keys_union") or [])
    rerank_keys_union = list(result.get("rerank_keys_union") or [])
    need_resolve = (
        set(union_keys) | set(pred_keys) | set(ann_edge_sims.keys()) | set(ann_keys_union) | set(rerank_keys_union)
    )
    for ks in s3_keys.values():
        need_resolve.update(ks)
    for ks in ann_keys.values():
        need_resolve.update(ks)
    for ks in rerank_keys.values():
        need_resolve.update(ks)
    for c in ranked_all:
        need_resolve.update(_chain_all_edge_keys(c))
    if not gold_ev_direct:
        need_resolve |= set(gold_keys)

    key_to_ev = await fetch_evidences_for_edge_keys(driver, need_resolve)
    if gold_ev_direct:
        gold_ev = gold_ev_direct
        missing_gold: list[str] = []
    else:
        gold_ev = _evidences_from_key_map(gold_keys, key_to_ev)
        missing_gold = sorted(k for k in gold_keys if k not in key_to_ev)
        if missing_gold:
            logger.warning(
                "%s: %s/%s gold edge_keys unresolved to evidence",
                qid,
                len(missing_gold),
                len(gold_keys),
            )

    # Fill any accepted edge missing text via Neo4j map
    for k in pred_keys:
        ev = _norm_evidence(key_to_ev.get(k))
        if ev:
            pred_ev.add(ev)

    recall_s3 = _s3_evidence_recalls(s3_keys, union_keys, gold_ev, key_to_ev)
    recall_ann_union = set_recall(_evidences_from_key_map(ann_keys_union, key_to_ev), gold_ev)
    recall_rerank_union = set_recall(_evidences_from_key_map(rerank_keys_union, key_to_ev), gold_ev)
    n_pred_evidences = len(pred_ev)
    vector_topk_k = VECTOR_BASELINE_K
    recall_vector_topk = _vector_baseline_recall(ann_edge_sims, vector_topk_k, gold_ev, key_to_ev)
    chain_gold = _chain_gold_counts(list(result.get("accepted") or []), gold_ev, key_to_ev)
    recall_at_k = metrics_at_k(ranked_all, gold_ev, key_to_ev)
    gold_coverage = _gold_stage_coverage(
        ann_keys=ann_keys,
        rerank_keys=rerank_keys,
        ann_keys_union=ann_keys_union,
        rerank_keys_union=rerank_keys_union,
        gold_ev=gold_ev,
        key_to_ev=key_to_ev,
    )
    metrics = {
        "id": qid,
        "effort": effort,
        "metric": "evidence_text",
        "gold_source": "evidence" if gold_ev_direct else "gold_edge_keys",
        "difficulty": item.get("difficulty"),
        "n_sq": len(sqs),
        "n_accepted": len(result.get("accepted") or []),
        "n_accepted_all": len(ranked_all),
        "sq_from_cache": sq_from_cache,
        "graph_from_cache": bool(result.get("from_graph_cache")),
        "recall_accepted": set_recall(pred_ev, gold_ev),
        "precision_accepted": set_precision(pred_ev, gold_ev),
        "recall_at_k": recall_at_k,
        "recall_vector_topk": recall_vector_topk,
        "vector_topk_k": vector_topk_k,
        "recall_ann_union": recall_ann_union,
        "recall_rerank_union": recall_rerank_union,
        "recall_s3": recall_s3,
        "gold_coverage": gold_coverage,
        "elapsed_sec": round(elapsed, 3),
        "n_gold_evidences": len(gold_ev),
        "n_pred_evidences": n_pred_evidences,
        "pred_evidences": sorted(pred_ev),
        "gold_evidences": sorted(gold_ev),
        "unresolved_gold_keys": missing_gold,
        "chain_gold": chain_gold,
        "subquestions": sqs,
    }
    return {
        "metrics": metrics,
        "result": result,
        "question": question,
        "qid": qid,
        "gold_ev": gold_ev,
        "sqs": sqs,
    }


async def main_async(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    qa_path = Path(args.dataset)
    with open(qa_path, encoding="utf-8") as f:
        dataset = json.load(f)
    if not isinstance(dataset, list):
        raise SystemExit(f"Dataset must be a JSON array: {qa_path}")

    if _dataset_has_evidence_packs(dataset):
        report = validate_evidence_dataset(dataset)
    else:
        report = validate_dataset(dataset)
    if report.get("warnings"):
        logger.warning("Dataset warnings: %s", report["warnings"][:10])
    if not report.get("ok", True):
        logger.error("Dataset validation: %s", json.dumps(report, indent=2)[:2000])
        if not args.allow_invalid:
            raise SystemExit(1)
    items = dataset
    if args.ids:
        want = {x.strip() for x in args.ids.split(",") if x.strip()}
        items = [it for it in items if str(it.get("id") or "") in want]
    if args.difficulty:
        items = [it for it in items if str(it.get("difficulty") or "").lower() == args.difficulty.lower()]
    if args.limit:
        items = items[: int(args.limit)]

    logger.info("Dataset %s → %s items", qa_path, len(items))

    config = load_config()
    driver = init_driver(config.neo4j.uri, config.neo4j.user, config.neo4j.password)
    overrides: dict[str, Any] = {}
    for raw in args.set_param or []:
        if "=" not in raw:
            raise SystemExit(f"--set expects KEY=VALUE, got {raw!r}")
        key, val = raw.split("=", 1)
        key = key.strip()
        allowed = {f.name for f in fields(Params)}
        if key not in allowed:
            raise SystemExit(f"unknown Params field {key!r}")
        overrides[key] = _parse_param_value(key, val.strip())
    base_params = merge_params(overrides or None)
    if args.set_param:
        logged = {}
        for raw in args.set_param:
            k = raw.split("=", 1)[0].strip()
            logged[k] = getattr(base_params, k)
        logger.info("param overrides %s", logged)
    logger.info(
        "effort=%s (auto: easy→low, medium→medium, hard→high)",
        args.effort,
    )

    sq_cache: dict[str, Any] | None = None
    sq_cache_path: Path | None = None
    if args.sq_cache:
        sq_cache_path = Path(args.sq_cache)
        sq_cache = load_sq_cache(sq_cache_path)
        logger.info(
            "sq-cache=%s entries=%s read_only=%s",
            sq_cache_path,
            len(sq_cache),
            args.sq_cache_read_only,
        )

    graph_cache_dir: Path | None = None
    if args.graph_cache:
        graph_cache_dir = Path(args.graph_cache)
        graph_cache_dir.mkdir(parents=True, exist_ok=True)
        n_files = len(list(graph_cache_dir.glob("*.json")))
        logger.info(
            "graph-cache=%s files=%s read_only=%s",
            graph_cache_dir,
            n_files,
            args.graph_cache_read_only,
        )

    reports: list[dict[str, Any]] = []
    all_chain_gold: list[dict[str, int]] = []
    out_dir = Path(args.out_dir)
    _reset_out_dir(out_dir)

    try:
        for item in items:
            has_gold = bool(_gold_evidences_from_item(item) or item.get("gold_edge_keys"))
            if not has_gold:
                logger.warning("Skip %s — no evidence/gold", item.get("id"))
                continue
            item_effort = effort_for_item(item, args.effort)
            item_params = base_params.with_effort(item_effort)
            logger.info(
                "Eval %s difficulty=%s effort=%s",
                item.get("id"),
                item.get("difficulty"),
                item_effort,
            )
            one = await eval_one(
                driver,
                item,
                effort=item_effort,
                params=item_params,
                sq_cache=sq_cache,
                sq_cache_path=sq_cache_path,
                sq_cache_read_only=bool(args.sq_cache_read_only),
                graph_cache_dir=graph_cache_dir,
                graph_cache_read_only=bool(args.graph_cache_read_only),
            )
            metrics = one["metrics"]
            reports.append(metrics)
            all_chain_gold.extend(metrics.get("chain_gold") or [])
            s3_union = float((metrics.get("recall_s3") or {}).get("union", 0.0))
            gu = (metrics.get("gold_coverage") or {}).get("union") or {}
            n_tot = int((metrics.get("gold_coverage") or {}).get("n_gold_total") or 0)
            atk = metrics.get("recall_at_k") or {}
            atk_str = " ".join(
                f"@{k}={float((atk.get(str(k)) or {}).get('recall') or 0.0):.3f}"
                for k in RECALL_AT_KS
            )
            logger.info(
                "%s difficulty=%s effort=%s recall(emitted)=%.3f "
                "precision=%.3f n_emitted=%s n_pool=%s %s "
                "ann_union=%.3f rerank_union=%.3f "
                "s3_union=%.3f vector_topk=%.3f (K=%s) "
                "gold_ann=%s/%s gold_ce=%s/%s",
                one["qid"],
                metrics.get("difficulty"),
                item_effort,
                metrics["recall_accepted"],
                metrics["precision_accepted"],
                metrics.get("n_accepted", 0),
                metrics.get("n_accepted_all", 0),
                atk_str,
                metrics.get("recall_ann_union", 0.0),
                metrics.get("recall_rerank_union", 0.0),
                s3_union,
                metrics.get("recall_vector_topk", 0.0),
                metrics.get("vector_topk_k", 0),
                gu.get("n_gold_ann", 0),
                n_tot,
                gu.get("n_gold_rerank", 0),
                n_tot,
            )
            stem = f"v6_{one['qid']}"
            (out_dir / f"{stem}.json").write_text(
                json.dumps(
                    {
                        "metrics": metrics,
                        "result": {
                            k: one["result"].get(k)
                            for k in (
                                "accepted",
                                "accepted_all",
                                "subquestions",
                                "trace",
                                "s3_keys",
                                "s3_keys_union",
                                "s3_edge_sims",
                                "ann_keys",
                                "ann_keys_union",
                                "ann_edge_sims",
                                "rerank_keys",
                                "rerank_keys_union",
                                "effort",
                            )
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            write_trace_md(
                out_dir / f"{stem}.md",
                question=one["question"],
                qid=one["qid"],
                subquestions=one["sqs"],
                result=one["result"],
                metrics=metrics,
            )
    finally:
        await close_driver()

    overall = mean_metrics(reports)
    by_diff = metrics_by_difficulty(reports)
    for band in ("easy", "medium", "hard"):
        band_reports = [
            r for r in reports if str(r.get("difficulty") or "").strip().lower() == band
        ]
        by_diff[band]["recall_at_k"] = mean_at_k(band_reports)
    chain_length_stats = _aggregate_chain_length_stats(all_chain_gold)
    summary = {
        "git": _git_sha(),
        "dataset": str(qa_path),
        "effort": args.effort,
        "metric": "evidence_text",
        "n": overall["n"],
        "mean_recall_accepted": overall["mean_recall_accepted"],
        "mean_precision_accepted": overall["mean_precision_accepted"],
        "mean_recall_s3_union": overall["mean_recall_s3_union"],
        "mean_recall_ann_union": overall["mean_recall_ann_union"],
        "mean_recall_rerank_union": overall["mean_recall_rerank_union"],
        "mean_recall_vector_topk": overall["mean_recall_vector_topk"],
        "mean_n_accepted": overall["mean_n_accepted"],
        "mean_n_accepted_all": overall["mean_n_accepted_all"],
        "recall_at_k": mean_at_k(reports),
        "vector_topk": {
            "k_policy": "VECTOR_BASELINE_K",
            "pool": "ann_union_pre_rerank",
            "k": VECTOR_BASELINE_K,
            "mean_recall": overall["mean_recall_vector_topk"],
        },
        "chain_length_stats": chain_length_stats,
        "by_difficulty": by_diff,
        "reports": reports,
    }
    # --skip-judge is ignored (judge removed); keep the historical filename
    # when the flag is passed so existing eval commands still find the summary.
    summary_name = "v6_summary_noj.json" if args.skip_judge else "v6_summary.json"
    summary_path = out_dir / summary_name
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    atk = summary["recall_at_k"]
    logger.info(
        "Done n=%s mean_recall(emitted)=%.3f mean_precision=%.3f "
        "avg_n_emitted=%.2f avg_n_pool=%.2f mean_ann_union=%.3f mean_rerank_union=%.3f "
        "mean_s3_union=%.3f mean_recall_vector_topk=%.3f → %s",
        summary["n"],
        summary["mean_recall_accepted"],
        summary["mean_precision_accepted"],
        summary["mean_n_accepted"],
        summary["mean_n_accepted_all"],
        summary["mean_recall_ann_union"],
        summary["mean_recall_rerank_union"],
        summary["mean_recall_s3_union"],
        summary["mean_recall_vector_topk"],
        summary_path,
    )
    logger.info(
        "recall@k (score-sorted pool): %s",
        "  ".join(
            f"@{k} R={((atk.get(str(k)) or {}).get('mean_recall') or 0.0):.3f} "
            f"P={((atk.get(str(k)) or {}).get('mean_precision') or 0.0):.3f} "
            f"n={((atk.get(str(k)) or {}).get('mean_n_paths') or 0.0):.1f}"
            for k in RECALL_AT_KS
        ),
    )
    for band in ("easy", "medium", "hard"):
        m = by_diff[band]
        band_atk = by_diff[band].get("recall_at_k") or {}
        logger.info(
            "%s n=%s mean_recall(emitted)=%.3f mean_precision=%.3f "
            "avg_n_emitted=%.2f avg_n_pool=%.2f mean_ann_union=%.3f mean_rerank_union=%.3f "
            "mean_s3_union=%.3f",
            band,
            m["n"],
            m["mean_recall_accepted"],
            m["mean_precision_accepted"],
            m["mean_n_accepted"],
            m["mean_n_accepted_all"],
            m["mean_recall_ann_union"],
            m["mean_recall_rerank_union"],
            m["mean_recall_s3_union"],
        )
        if m["n"] and band_atk:
            logger.info(
                "%s recall@k: %s",
                band,
                "  ".join(
                    f"@{k} R={((band_atk.get(str(k)) or {}).get('mean_recall') or 0.0):.3f} "
                    f"P={((band_atk.get(str(k)) or {}).get('mean_precision') or 0.0):.3f}"
                    for k in RECALL_AT_KS
                ),
            )
    logger.info(
        "Chain length histogram (all accepted paths, spine+fans): n_paths=%s avg_len=%.2f avg_gold_per_path=%.2f",
        chain_length_stats["n_paths_total"],
        chain_length_stats["avg_len"],
        chain_length_stats["avg_gold_per_path"],
    )
    for L, st in (chain_length_stats.get("by_length") or {}).items():
        logger.info(
            "  len %s: n=%s avg_gold=%.2f",
            L,
            st["n"],
            st["avg_gold"],
        )
    logger.info(
        "Vector top-K baseline (ANN pool pre-CE by sim, K=%s): mean_recall=%.3f",
        VECTOR_BASELINE_K,
        summary["mean_recall_vector_topk"],
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate Algorithm V6 (evidence-text metrics)")
    p.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help=f"JSON dataset (default: {DEFAULT_DATASET}; evidence[] or gold_edge_keys)",
    )
    p.add_argument(
        "--effort",
        default="auto",
        choices=["auto", "low", "medium", "high"],
        help=(
            "Path / emit budget. auto (default): easy→low, medium→medium, hard→high; "
            "low|medium|high forces the same effort for every item"
        ),
    )
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="Deprecated: judge was removed; flag is ignored",
    )
    p.add_argument(
        "--sq-cache",
        nargs="?",
        const=str(DEFAULT_SQ_CACHE),
        default=None,
        help=(
            "Cache mock_decompose subquestions per question id. "
            f"Optional PATH (default: {DEFAULT_SQ_CACHE}). "
            "Hit → read from file; miss → decompose and write (unless --sq-cache-read-only)."
        ),
    )
    p.add_argument(
        "--sq-cache-read-only",
        action="store_true",
        help="With --sq-cache: never call decompose; fail on cache miss",
    )
    p.add_argument(
        "--graph-cache",
        nargs="?",
        const=str(DEFAULT_GRAPH_CACHE),
        default=None,
        help=(
            "Cache S3 CandidateGraphs per question id (skip ANN/CE/bridges on hit). "
            f"Optional DIR (default: {DEFAULT_GRAPH_CACHE}). "
            "Safe to sweep S4/S5/S6 params; rebuilds if L/framing/sq change."
        ),
    )
    p.add_argument(
        "--graph-cache-read-only",
        action="store_true",
        help="With --graph-cache: never rebuild S1–S3; fail on miss/fingerprint mismatch",
    )
    p.add_argument(
        "--ids",
        default="",
        help="Comma-separated question ids to run (e.g. q0,q1,q14)",
    )
    p.add_argument(
        "--difficulty",
        default="",
        choices=["", "easy", "medium", "hard"],
        help="Filter by difficulty field (qa_evidence_50)",
    )
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=(
            f"Report directory (default: {DEFAULT_OUT_DIR}). "
            "Wiped and recreated on every run. "
            "SQ/graph caches live in tests/reports/v6_cache/."
        ),
    )
    p.add_argument(
        "--set",
        dest="set_param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a Params field (repeatable), e.g. --set prize_top=40",
    )
    p.add_argument("--allow-invalid", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

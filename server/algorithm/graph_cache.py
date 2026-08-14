"""S3 CandidateGraph cache: dump after ANN→CE→bridges; replay S4–S5 offline."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from server.algorithm.models import CandidateGraph, EdgeRecord, SubQuestion
from server.algorithm.params import Params
from server.algorithm.stage3_graphs import _finalize_graph

logger = logging.getLogger(__name__)

# Bump when cache edge schema or framing id meaning changes.
GRAPH_CACHE_VERSION = 2

# Hardcoded S2b query template id (framing is not a Params field yet).
RERANK_FRAMING_ID = "claim_v1"

# Params that change the S3 edge pool / transition topology.
S3_FINGERPRINT_KEYS: tuple[str, ...] = (
    "L",
    "L_raw_max",
    "bridge_top",
    "branch_cap",
    "rerank_enabled",
)


def edge_to_cache_dict(e: EdgeRecord) -> dict[str, Any]:
    """Serialize edge for disk."""
    return {
        "edge_key": e.edge_key,
        "element_id": e.element_id,
        "rel_type": e.rel_type,
        "start_id": e.start_id,
        "end_id": e.end_id,
        "start_name": e.start_name,
        "end_name": e.end_name,
        "start_label": e.start_label,
        "end_label": e.end_label,
        "sim": float(e.sim),
        "rerank_score": float(e.rerank_score),
        "chunk_id": e.chunk_id or "",
        "evidence": e.evidence or "",
        "source_file": e.source_file or "",
        "source": e.source or "ann",
        "confidence": float(e.confidence),
    }


def edge_from_cache_dict(d: dict[str, Any]) -> EdgeRecord:
    return EdgeRecord(
        edge_key=str(d.get("edge_key") or ""),
        element_id=str(d.get("element_id") or ""),
        rel_type=str(d.get("rel_type") or d.get("type") or ""),
        start_id=str(d.get("start_id") or ""),
        end_id=str(d.get("end_id") or ""),
        start_name=str(d.get("start_name") or d.get("start") or ""),
        end_name=str(d.get("end_name") or d.get("end") or ""),
        start_label=str(d.get("start_label") or ""),
        end_label=str(d.get("end_label") or ""),
        sim=float(d.get("sim") or 0.0),
        rerank_score=float(d.get("rerank_score") or 0.0),
        chunk_id=str(d.get("chunk_id") or ""),
        evidence=str(d.get("evidence") or ""),
        source_file=str(d.get("source_file") or ""),
        source=str(d.get("source") or "ann"),
        confidence=float(d.get("confidence") or 1.0),
    )


def graph_to_cache_dict(g: CandidateGraph) -> dict[str, Any]:
    return {
        "source_graph": g.source_graph,
        "edges": [edge_to_cache_dict(e) for e in g.edges.values()],
    }


def graph_from_cache_dict(d: dict[str, Any], *, branch_cap: int) -> CandidateGraph:
    source = str(d.get("source_graph") or "sq")
    raw_edges = d.get("edges") or []
    edges: dict[str, EdgeRecord] = {}
    if isinstance(raw_edges, dict):
        items = raw_edges.values()
    else:
        items = raw_edges
    for item in items:
        if not isinstance(item, dict):
            continue
        e = edge_from_cache_dict(item)
        if not e.edge_key:
            continue
        prev = edges.get(e.edge_key)
        if prev is None or e.sim > prev.sim:
            edges[e.edge_key] = e
    return _finalize_graph(source, edges, branch_cap)


def graphs_to_cache_dict(graphs: dict[str, CandidateGraph]) -> dict[str, Any]:
    return {sid: graph_to_cache_dict(g) for sid, g in graphs.items()}


def graphs_from_cache_dict(
    raw: dict[str, Any], *, branch_cap: int
) -> dict[str, CandidateGraph]:
    out: dict[str, CandidateGraph] = {}
    for sid, gd in (raw or {}).items():
        if not isinstance(gd, dict):
            continue
        out[str(sid)] = graph_from_cache_dict(gd, branch_cap=branch_cap)
    return out


def _sq_list(
    sqs: list[SubQuestion] | list[dict[str, Any]],
) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for i, s in enumerate(sqs):
        if isinstance(s, SubQuestion):
            out.append({"id": s.id, "text": s.text})
            continue
        text = str(s.get("text") or "").strip()
        if not text:
            continue
        sid = str(s.get("id") or f"sq{i+1}")
        out.append({"id": sid, "text": text})
    return out


def build_fingerprint(
    params: Params,
    sqs: list[SubQuestion] | list[dict[str, Any]],
    *,
    framing_id: str = RERANK_FRAMING_ID,
) -> dict[str, Any]:
    return {
        "version": GRAPH_CACHE_VERSION,
        "framing": framing_id,
        "params": {k: getattr(params, k) for k in S3_FINGERPRINT_KEYS},
        "subquestions": _sq_list(sqs),
    }


def fingerprint_hash(fp: dict[str, Any]) -> str:
    blob = json.dumps(fp, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def build_s3_bundle(
    *,
    qid: str,
    question: str,
    params: Params,
    sqs: list[SubQuestion] | list[dict[str, Any]],
    graphs: dict[str, CandidateGraph],
    ann_keys: dict[str, list[str]],
    rerank_keys: dict[str, list[str]],
    ann_edge_sims: dict[str, float],
) -> dict[str, Any]:
    fp = build_fingerprint(params, sqs)
    return {
        "qid": qid,
        "question": question,
        "fingerprint": fp,
        "fingerprint_hash": fingerprint_hash(fp),
        "subquestions": _sq_list(sqs),
        "graphs": graphs_to_cache_dict(graphs),
        "ann_keys": {k: list(v) for k, v in ann_keys.items()},
        "rerank_keys": {k: list(v) for k, v in rerank_keys.items()},
        "ann_edge_sims": {k: float(v) for k, v in ann_edge_sims.items()},
    }


def load_s3_bundle_graphs(
    bundle: dict[str, Any], *, branch_cap: int
) -> dict[str, CandidateGraph]:
    return graphs_from_cache_dict(bundle.get("graphs") or {}, branch_cap=branch_cap)


def bundle_matches(
    bundle: dict[str, Any],
    *,
    question: str,
    params: Params,
    sqs: list[SubQuestion] | list[dict[str, Any]],
) -> bool:
    if not isinstance(bundle, dict) or "graphs" not in bundle:
        return False
    cached_q = str(bundle.get("question") or "")
    if cached_q and question and cached_q.strip() != question.strip():
        logger.warning("graph-cache question mismatch; ignoring entry")
        return False
    want = build_fingerprint(params, sqs)
    got = bundle.get("fingerprint")
    if not isinstance(got, dict):
        return False
    if got.get("version") != want["version"]:
        return False
    if got.get("framing") != want["framing"]:
        return False
    if got.get("params") != want["params"]:
        return False
    if got.get("subquestions") != want["subquestions"]:
        return False
    return True


def graph_cache_path(cache_dir: Path, qid: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in qid)
    return cache_dir / f"{safe}.json"


def load_graph_cache_entry(cache_dir: Path, qid: str) -> dict[str, Any] | None:
    path = graph_cache_path(cache_dir, qid)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("graph-cache load failed %s: %s", path, e)
        return None
    return data if isinstance(data, dict) else None


def save_graph_cache_entry(cache_dir: Path, qid: str, bundle: dict[str, Any]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = graph_cache_path(cache_dir, qid)
    path.write_text(
        json.dumps(bundle, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return path

"""V4 graph retrieval pipeline (stages 1–6).

Orchestration previously lived in tests/evaluate_v4.py.
"""

from __future__ import annotations

import json
import logging
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from neo4j import AsyncDriver

from server.algorithm.v4.stage1_chunking import decompose_query
from server.algorithm.v4.stage2_anchoring import find_anchors_for_subqueries
from server.algorithm.v4.stage3_projection import (
    create_dynamic_projection_and_filter,
    drop_projection,
)
from server.algorithm.v4.stage4_coverage import find_coverage_paths
from server.algorithm.v4.stage4_mdst import find_pcst_paths, find_steiner_tree_edges
from server.algorithm.v4.stage4_ppr_chains import find_ppr_modulated_chains
from server.algorithm.v4.stage4_topology import find_topological_paths
from server.algorithm.v4.stage5_reranking import rerank_paths, serialize_path
from server.algorithm.v4.stage6_cascade import rank_and_filter_paths

logger = logging.getLogger(__name__)

# --- HYPERPARAMETERS & CONSTANTS ---
PIPELINE_PARAMS: dict[str, Any] = {
    "STAGE2_K_ANCHORS": 10,
    "STAGE2_L_TOP_PER_SUBQUERY": 200,
    "STAGE3_MIN_COMPRESSION_RATIO": 0.1,
    "STAGE3_MAX_COMPRESSION_RATIO": 0.25,
    "STAGE3_PPR_MASS_TARGET": 0.92,
    "STAGE4_ALPHA": 0.0,
    "STAGE4_BETA": 0.0,
    "STAGE4_K_SHORTEST": 20,
    "STAGE4_ITERATIONS": 1,
    "STAGE4_PENALTY_M": 2.0,
    "STAGE4_PENALTY_C_MULTIPLIER": 1.0,
    "STAGE4_ALGO": "coverage",
    "STAGE4_MAX_PATH_LEN": 4,
    "STAGE4_CANDIDATE_CAP": 300,
    "STAGE4_BUDGET": 50,
    "STAGE4_UNCOVERED_BOOST": 2.0,
    "STAGE4_PPR_BIAS_POW": 0.0,
    "STAGE4_MMR_LAMBDA": 0.1,
    "STAGE4_BEAM_WIDTH": 3,
    "STAGE4_LONG_TAIL_HARVEST": True,
    "STAGE4_MULTI_SUBQUERY": True,
    "STAGE4_CHAINS_PER_SOURCE": 1,
    "STAGE4_MAX_TOTAL_CHAINS": 500,
    "STAGE6_K_LIMIT_FINAL": 50,
    "STAGE6_OVERLAP_THRESHOLD": 0.99,
    "EVAL_K_VALUES": [5, 10, 20, 50, 100, 200, 300, 500],
    "DISABLE_RERANKER": True,
    "GDS_ONLY": False,
}

# Backward-compatible alias for evaluate_v4 / tune scripts
EVAL_PARAMS = PIPELINE_PARAMS


def _path_rel_ids(path: dict) -> set:
    ids = set()
    for hop_rels in path.get("relationships", []):
        for rel in hop_rels:
            rid = rel.get("rel_id")
            if rid:
                ids.add(rid)
    return ids


def compute_recall_curve(expected_ids: set, paths: list, k_values: list[int]) -> dict:
    """Recall vs number of paths (S-Path-RAG style coverage curve)."""
    curve = {}
    retrieved: set = set()
    for k in k_values:
        limit = min(k, len(paths))
        for p in paths[:limit]:
            retrieved |= _path_rel_ids(p)
        hits = len(retrieved.intersection(expected_ids))
        curve[f"RecallCurve@{k}"] = hits / len(expected_ids) if expected_ids else 1.0
    return curve


def compute_path_diversity_metrics(expected_ids: set, paths: list, k: int = 50) -> dict:
    """Per-path marginal gold coverage and hop overlap diversity at top-k."""
    top = paths[:k]
    if not top:
        return {"MarginalGoldEdges@50": 0.0, "HopJaccardDist@50": 0.0}

    covered_gold: set = set()
    marginal_gold_counts: list[int] = []
    hop_sets: list[set] = []

    for p in top:
        rel_ids = _path_rel_ids(p)
        new_gold = rel_ids.intersection(expected_ids) - covered_gold
        marginal_gold_counts.append(len(new_gold))
        covered_gold |= rel_ids.intersection(expected_ids)

        hops = set()
        nodes = p.get("nodes", [])
        for i in range(len(nodes) - 1):
            n1 = nodes[i].get("internal_id")
            n2 = nodes[i + 1].get("internal_id")
            if n1 is not None and n2 is not None:
                hops.add(tuple(sorted([str(n1), str(n2)])))
        hop_sets.append(hops)

    avg_marginal = sum(marginal_gold_counts) / len(marginal_gold_counts)

    jaccard_dists: list[float] = []
    for i in range(1, len(hop_sets)):
        a, b = hop_sets[i - 1], hop_sets[i]
        union = a | b
        if not union:
            continue
        inter = a & b
        jaccard_dists.append(1.0 - len(inter) / len(union))

    avg_jaccard_dist = sum(jaccard_dists) / len(jaccard_dists) if jaccard_dists else 0.0

    return {
        f"MarginalGoldEdges@{k}": avg_marginal,
        f"HopJaccardDist@{k}": avg_jaccard_dist,
    }


def compute_ranking_metrics(expected_ids: set, paths: list, k_values=None):
    if k_values is None:
        k_values = [5, 10, 20, 50]
    metrics = {}
    for k in k_values:
        top_k_paths = paths[:k]

        retrieved_ids = set()
        retrieved_nodes = set()
        first_hit_rank = None

        for rank, p in enumerate(top_k_paths):
            path_rel_ids = set()
            for hop_rels in p.get("relationships", []):
                for rel in hop_rels:
                    rid = rel.get("rel_id")
                    if rid:
                        path_rel_ids.add(rid)
                        retrieved_ids.add(rid)

            for node in p.get("nodes", []):
                nid = node.get("id")
                if nid:
                    retrieved_nodes.add(nid)

            if first_hit_rank is None and not path_rel_ids.isdisjoint(expected_ids):
                first_hit_rank = rank + 1

        hits_count = len(retrieved_ids.intersection(expected_ids))

        metrics[f"Hit@{k}"] = 1.0 if first_hit_rank is not None else 0.0
        metrics[f"MRR@{k}"] = 1.0 / first_hit_rank if first_hit_rank is not None else 0.0
        metrics[f"Recall@{k}"] = hits_count / len(expected_ids) if expected_ids else 1.0

        total_retrieved_edges = len(retrieved_ids)
        metrics[f"Precision@{k}"] = (
            hits_count / total_retrieved_edges if total_retrieved_edges > 0 else 0.0
        )
        metrics[f"UniqueTriplets@{k}"] = total_retrieved_edges
        metrics[f"UniqueNodes@{k}"] = len(retrieved_nodes)

    metrics.update(compute_recall_curve(expected_ids, paths, k_values))
    if 50 in k_values:
        metrics.update(compute_path_diversity_metrics(expected_ids, paths, k=50))

    return metrics


def get_next_attempt_dir(base_path: Path) -> Path:
    base_path.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        attempt_dir = base_path / f"v4_attempt_{attempt}"
        if not attempt_dir.exists():
            attempt_dir.mkdir(parents=True)
            return attempt_dir
        attempt += 1


def _resolve_params(params: Optional[dict] = None) -> dict:
    if params is None:
        return PIPELINE_PARAMS
    merged = deepcopy(PIPELINE_PARAMS)
    merged.update(params)
    return merged


async def evaluate_question(
    driver: AsyncDriver,
    q_data: dict,
    config,
    use_cache: bool = False,
    params: Optional[dict] = None,
) -> dict:
    """Full V4 run with optional gold metrics (used by evaluate_v4)."""
    p = _resolve_params(params)
    start_time = time.time()
    question = q_data["question"]
    target_paths = set(q_data.get("target_paths", []))

    print("\n--- STAGE 1 ---")
    t0 = time.time()

    subqueries = None
    cache_file = Path("tests/cache_subqueries.json")

    if use_cache and cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cache_data = json.load(f)
                if question in cache_data:
                    subqueries = cache_data[question]
                    print("  [Stage 1] Loaded subqueries from cache.")
        except Exception as e:
            logger.warning(f"Failed to read subqueries cache: {e}")

    if not subqueries:
        subqueries = await decompose_query(question, config)
        if use_cache:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                cache_data = {}
                if cache_file.exists():
                    with open(cache_file, "r", encoding="utf-8") as f:
                        cache_data = json.load(f)
                cache_data[question] = subqueries
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache_data, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.warning(f"Failed to write subqueries cache: {e}")

    t1 = time.time()
    print(f"Stage 1 completed in {t1 - t0:.2f}s")

    print("\n--- STAGE 2 ---")
    t2 = time.time()
    anchoring_result = await find_anchors_for_subqueries(
        driver,
        subqueries,
        k=p["STAGE2_K_ANCHORS"],
        l=p["STAGE2_L_TOP_PER_SUBQUERY"],
    )
    t3 = time.time()
    print(f"Stage 2 completed in {t3 - t2:.2f}s")
    anchor_ids = anchoring_result["anchor_ids"]
    subquery_vectors = anchoring_result["vectors"]

    final_paths: list[dict] = []
    raw_paths: list[dict] = []
    gds_time = 0.0
    rerank_time = 0.0
    s4_time = 0.0
    s5_time = 0.0
    s6_time = 0.0
    t5 = t2
    gds_start = t2
    stage3_metrics = {
        "Recall": 0.0,
        "Precision": 0.0,
        "TotalEdges": 0,
        "TotalNodes": 0,
    }

    if anchor_ids and subquery_vectors:
        print("\n--- STAGE 3 ---")
        gds_start = time.time()
        stage3_res = await create_dynamic_projection_and_filter(
            driver,
            anchor_ids,
            subquery_vectors,
            min_compression_ratio=p["STAGE3_MIN_COMPRESSION_RATIO"],
            max_compression_ratio=p["STAGE3_MAX_COMPRESSION_RATIO"],
            base_alpha=p["STAGE4_ALPHA"],
            base_beta=p["STAGE4_BETA"],
            ppr_mass_target=p["STAGE3_PPR_MASS_TARGET"],
        )
        t5 = time.time()
        print(f"Stage 3 completed in {t5 - gds_start:.2f}s")

        allowed_ids = stage3_res.get("allowed_ids", [])
        stage3_metrics = {
            "Recall": 0.0,
            "Precision": 0.0,
            "TotalEdges": 0,
            "TotalNodes": len(allowed_ids) if allowed_ids else 0,
        }

        if allowed_ids:
            async with driver.session() as session:
                res = await session.run(
                    "MATCH (n)-[r]->(m) WHERE elementId(n) IN $allowed "
                    "AND elementId(m) IN $allowed RETURN elementId(r) as rid",
                    allowed=list(allowed_ids),
                )
                subgraph_rids = {record["rid"] async for record in res}

                hits = len(subgraph_rids.intersection(target_paths))
                stage3_metrics["Recall"] = (
                    hits / len(target_paths) if target_paths else 1.0
                )
                stage3_metrics["Precision"] = (
                    hits / len(subgraph_rids) if subgraph_rids else 0.0
                )
                stage3_metrics["TotalEdges"] = len(subgraph_rids)

        if p.get("GDS_ONLY", False):
            gds_time = time.time() - gds_start
            raw_paths = []
        else:
            print("\n--- STAGE 4 ---")
            t6 = time.time()
            if p["STAGE4_ALGO"] == "ppr_chains":
                raw_paths = await find_ppr_modulated_chains(
                    driver,
                    stage3_res["graph_name"],
                    stage3_res["allowed_ids"],
                    beam_width=p["STAGE4_BEAM_WIDTH"],
                    chains_per_source=p["STAGE4_CHAINS_PER_SOURCE"],
                    max_total_chains=p["STAGE4_MAX_TOTAL_CHAINS"],
                )
            elif p["STAGE4_ALGO"] == "coverage":
                raw_paths = await find_coverage_paths(
                    driver,
                    stage3_res["graph_name"],
                    stage3_res["allowed_ids"],
                    anchor_ids=anchor_ids,
                    s2_scores=anchoring_result.get("scores", {}),
                    ppr_scores=stage3_res.get("ppr_scores", {}),
                    subquery_vectors=subquery_vectors,
                    max_path_len=p["STAGE4_MAX_PATH_LEN"],
                    candidate_cap=p["STAGE4_CANDIDATE_CAP"],
                    budget=p["STAGE4_BUDGET"],
                    uncovered_boost=p["STAGE4_UNCOVERED_BOOST"],
                    ppr_bias_pow=p["STAGE4_PPR_BIAS_POW"],
                    mmr_lambda=p["STAGE4_MMR_LAMBDA"],
                    beam_width=p["STAGE4_BEAM_WIDTH"],
                    long_tail_harvest=p["STAGE4_LONG_TAIL_HARVEST"],
                    multi_subquery=p["STAGE4_MULTI_SUBQUERY"],
                )
            elif p["STAGE4_ALGO"] == "pcst":
                raw_paths = await find_pcst_paths(
                    driver,
                    stage3_res["graph_name"],
                    stage3_res["allowed_ids"],
                    anchor_ids,
                    subquery_vectors,
                    anchor_scores=anchoring_result.get("scores", {}),
                    ppr_scores=stage3_res.get("ppr_scores", {}),
                    budget=p["STAGE4_BUDGET"],
                    max_path_len=p["STAGE4_MAX_PATH_LEN"],
                )
            elif p["STAGE4_ALGO"] == "steiner":
                raw_paths = await find_steiner_tree_edges(
                    driver,
                    stage3_res["allowed_ids"],
                    anchor_ids,
                    subquery_vectors,
                    anchor_scores=anchoring_result.get("scores", {}),
                    ppr_scores=stage3_res.get("ppr_scores", {}),
                )
                await drop_projection(driver, stage3_res["graph_name"])
            else:
                raw_paths = await find_topological_paths(
                    driver,
                    stage3_res["graph_name"],
                    stage3_res["allowed_ids"],
                    anchor_ids,
                    s2_scores=anchoring_result.get("scores", {}),
                    ppr_scores=stage3_res.get("ppr_scores", {}),
                    k_limit=p["STAGE4_K_SHORTEST"],
                    iterations=p["STAGE4_ITERATIONS"],
                    penalty_m=p["STAGE4_PENALTY_M"],
                    penalty_c_mult=p["STAGE4_PENALTY_C_MULTIPLIER"],
                )
            t7 = time.time()
            s4_time = t7 - t6
            print(f"Stage 4 completed in {s4_time:.2f}s")
            gds_time = time.time() - gds_start

        if raw_paths:
            print("\n--- STAGE 5 ---")
            rerank_start = time.time()
            if p.get("DISABLE_RERANKER", False):
                reranked_paths = raw_paths
                for path in reranked_paths:
                    path["semantic_score"] = 0.0
                    path["serialized_text"] = serialize_path(path)
            else:
                reranked_paths = await rerank_paths(question, raw_paths)
            t9 = time.time()
            s5_time = t9 - rerank_start
            print(f"Stage 5 completed in {s5_time:.2f}s")
            rerank_time = time.time() - rerank_start

            print("\n--- STAGE 6 ---")
            t10 = time.time()
            if p["STAGE4_ALGO"] in ("coverage", "pcst"):
                if p.get("DISABLE_RERANKER", False):
                    final_paths = sorted(
                        reranked_paths,
                        key=lambda path: float(
                            path.get(
                                "prize_sum",
                                -float(path.get("totalCost", 0.0) or 0.0),
                            )
                            or 0.0
                        ),
                        reverse=True,
                    )[: p["STAGE4_BUDGET"]]
                else:
                    final_paths = sorted(
                        reranked_paths,
                        key=lambda path: path.get("semantic_score", 0.0),
                        reverse=True,
                    )[: p["STAGE4_BUDGET"]]
            else:
                final_paths = rank_and_filter_paths(
                    reranked_paths,
                    k_limit=p["STAGE6_K_LIMIT_FINAL"],
                    overlap_threshold=p["STAGE6_OVERLAP_THRESHOLD"],
                )
            t11 = time.time()
            s6_time = t11 - t10
            print(f"Stage 6 completed in {s6_time:.2f}s")

    metrics = compute_ranking_metrics(
        target_paths, final_paths, k_values=p["EVAL_K_VALUES"]
    )

    paths_details = [
        {
            "gds_cost": (
                f"{path['chain_score']:.6e}"
                if "chain_score" in path
                else (
                    f"prize={path.get('prize_sum', -path.get('totalCost', 0)):.4f}"
                    if p.get("STAGE4_ALGO") in ("coverage", "pcst")
                    else f"{path.get('totalCost', 0):.4f}"
                )
            ),
            "text": path.get("serialized_text", ""),
        }
        for path in final_paths
    ]

    total_time = time.time() - start_time

    return {
        "question": question,
        "metrics": metrics,
        "stage3_metrics": stage3_metrics,
        "timing": {
            "total_time": total_time,
            "s1_time": t1 - t0,
            "s2_time": t3 - t2,
            "s3_time": t5 - gds_start if anchor_ids and subquery_vectors else 0.0,
            "s4_time": s4_time,
            "s5_time": s5_time,
            "s6_time": s6_time,
            "gds_time": gds_time,
            "rerank_time": rerank_time,
        },
        "total_anchors": len(anchor_ids),
        "total_raw_paths": len(raw_paths),
        "total_retrieved_paths": len(final_paths),
        "paths_details": paths_details,
        "final_paths": final_paths,
    }


async def run(
    question: str,
    *,
    driver: Optional[AsyncDriver] = None,
    config=None,
    params: Optional[dict] = None,
    top_k: int = 50,
    use_cache: bool = False,
) -> list[dict]:
    """Agent-facing entry: return top-k ranked path dicts with serialized_text."""
    if config is None:
        from server.utils.config import load_config

        config = load_config()

    if driver is None:
        from server.core.db import get_driver

        driver = get_driver()

    result = await evaluate_question(
        driver,
        {"question": question, "target_paths": []},
        config,
        use_cache=use_cache,
        params=params,
    )
    return result.get("final_paths", [])[:top_k]


class V4Pipeline:
    """Thin wrapper so tools can keep create_default_pipeline(...).run(...)."""

    def __init__(self, llm_profile=None, params: Optional[dict] = None):
        self.llm_profile = llm_profile
        self.params = params

    async def run(
        self,
        question: str,
        top_k: int = 50,
        use_cache: bool = False,
        **_kwargs,
    ) -> list[dict]:
        return await run(
            question,
            params=self.params,
            top_k=top_k,
            use_cache=use_cache,
        )


def create_default_pipeline(llm_profile=None, params: Optional[dict] = None) -> V4Pipeline:
    return V4Pipeline(llm_profile=llm_profile, params=params)

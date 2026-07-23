import asyncio
import logging
import os
import time

import httpx
from neo4j import AsyncDriver

from server.core.db import close_driver, init_driver
from server.utils.config import load_config

logger = logging.getLogger(__name__)

OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
OPENROUTER_EMBED_MODEL = "nvidia/nemotron-3-embed-1b:free"
_MAX_RETRIES = 5
_RETRY_BASE_SEC = 2.0


def _openrouter_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM__PROFILES__OTHER__API_KEY")
    if key:
        return key
    # Fallbacks when running outside docker / without exported env
    from pathlib import Path

    from dotenv import load_dotenv

    root = Path(__file__).resolve().parents[3]
    for name in (".env.ragas-testing", ".env"):
        path = root / name
        if path.exists():
            load_dotenv(path, override=False)
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM__PROFILES__OTHER__API_KEY")
    if not key:
        raise RuntimeError("OpenRouter API key not found (OPENROUTER_API_KEY or LLM__PROFILES__OTHER__API_KEY)")
    return key


async def get_embeddings_batch(
    texts: list[str],
    model_id: str = OPENROUTER_EMBED_MODEL,
    url: str = OPENROUTER_EMBEDDINGS_URL,
) -> list[list[float]]:
    """Fetch embeddings from OpenRouter (Nemotron). Returns [] on hard failure."""
    if not texts:
        return []

    headers = {
        "Authorization": f"Bearer {_openrouter_api_key()}",
        "Content-Type": "application/json",
    }
    payload = {"model": model_id, "input": texts, "encoding_format": "float"}

    async with httpx.AsyncClient(trust_env=False) as client:
        for attempt in range(_MAX_RETRIES):
            try:
                response = await client.post(url, headers=headers, json=payload, timeout=60.0)
                if response.status_code in (429, 500, 502, 503, 504):
                    wait = _RETRY_BASE_SEC * (2**attempt)
                    logger.warning(
                        "OpenRouter embeddings HTTP %s, retry in %.1fs (%s/%s)",
                        response.status_code,
                        wait,
                        attempt + 1,
                        _MAX_RETRIES,
                    )
                    await asyncio.sleep(wait)
                    continue
                response.raise_for_status()
                data = response.json()["data"]
                # OpenAI-compatible: data may be unordered — sort by index
                data_sorted = sorted(data, key=lambda item: item.get("index", 0))
                return [item["embedding"] for item in data_sorted]
            except Exception as e:
                wait = _RETRY_BASE_SEC * (2**attempt)
                logger.error(
                    "Error fetching embeddings (attempt %s/%s): %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    e,
                )
                if attempt + 1 >= _MAX_RETRIES:
                    return []
                await asyncio.sleep(wait)
    return []


async def get_embedding(
    text: str,
    model_id: str = OPENROUTER_EMBED_MODEL,
    url: str = OPENROUTER_EMBEDDINGS_URL,
) -> list[float]:
    """Fetch a single embedding via OpenRouter Nemotron."""
    vectors = await get_embeddings_batch([text], model_id=model_id, url=url)
    return vectors[0] if vectors else []



async def fetch_vector_indexes(driver: AsyncDriver) -> dict[str, list[str]]:
    """
    Dynamically find all node and relationship vector indexes in the database.
    """
    indexes = {"nodes": [], "relationships": []}
    async with driver.session() as session:
        res = await session.run("SHOW VECTOR INDEXES")
        records = [r async for r in res]
        for r in records:
            idx_name = r["name"]
            entity_type = r.get("entityType", "").upper()
            if entity_type == "NODE":
                indexes["nodes"].append(idx_name)
            elif entity_type == "RELATIONSHIP":
                indexes["relationships"].append(idx_name)
            else:
                # Fallback for some Neo4j versions where entityType might be missing
                # If labelsOrTypes exists, we could try to guess, but normally it's NODE by default in older versions
                if "edge" in idx_name.lower():
                    indexes["relationships"].append(idx_name)
                else:
                    indexes["nodes"].append(idx_name)
    return indexes


async def find_anchors_for_subqueries(driver: AsyncDriver, subqueries: list[str], k: int = 20, l: int = 5) -> dict:
    """
    Stage 2: Two-Phase Selective Anchoring
    Returns a dictionary containing:
      - 'anchor_ids': a flat list of unique node elementIds.
      - 'vectors': a dictionary mapping subquery -> embedding vector.
    """
    if not subqueries:
        return {"anchor_ids": [], "vectors": {}}

    # 1. Fetch all vector indices dynamically
    indexes = await fetch_vector_indexes(driver)
    node_indexes = indexes["nodes"]
    rel_indexes = indexes["relationships"]

    logger.info(f"Found Node Indexes: {node_indexes}")
    logger.info(f"Found Relationship Indexes: {rel_indexes}")

    # 2. Embed subqueries
    subquery_vectors = {}
    for sq in subqueries:
        vec = await get_embedding(sq)
        if vec:
            subquery_vectors[sq] = vec

    if not subquery_vectors:
        logger.error("Failed to generate any embeddings.")
        return {"anchor_ids": [], "vectors": {}}

    async def query_node_index(idx_name: str, vec: list[float], sq_index: int):
        query = f"""
        CALL db.index.vector.queryNodes('{idx_name}', $l, $vec) YIELD node, score
        OPTIONAL MATCH (node)-[r]-()
        WITH node, score, count(r) AS degree
        RETURN elementId(node) AS id, [l in labels(node) WHERE l IN ['Metabolite', 'Microbe', 'StarterCulture', 'EnvironmentCondition']][0] AS label, degree, score
        """
        async with driver.session() as session:
            res = await session.run(query, parameters={"l": l, "vec": vec})
            return [
                (r["id"], r["score"], sq_index, r["label"], "node", r["degree"])
                async for r in res
                if r["degree"] < 1000
            ]

    async def query_rel_index(idx_name: str, vec: list[float], sq_index: int):
        query = f"""
        CALL db.index.vector.queryRelationships('{idx_name}', $l, $vec) YIELD relationship, score
        WITH relationship, score, startNode(relationship) AS sn, endNode(relationship) AS en
        OPTIONAL MATCH (sn)-[r1]-()
        WITH relationship, score, sn, en, count(r1) AS start_degree
        OPTIONAL MATCH (en)-[r2]-()
        WITH score, sn, en, start_degree, count(r2) AS end_degree
        RETURN elementId(sn) AS start_id,
               [l in labels(sn) WHERE l IN ['Metabolite', 'Microbe', 'StarterCulture', 'EnvironmentCondition']][0] AS start_label,
               start_degree,
               elementId(en) AS end_id,
               [l in labels(en) WHERE l IN ['Metabolite', 'Microbe', 'StarterCulture', 'EnvironmentCondition']][0] AS end_label,
               end_degree,
               score
        """
        async with driver.session() as session:
            res = await session.run(query, parameters={"l": l, "vec": vec})
            ids = []
            async for r in res:
                if r["end_degree"] < 1000:
                    ids.append(
                        (
                            r["end_id"],
                            r["score"],
                            sq_index,
                            r["end_label"],
                            "rel_end",
                            r["end_degree"],
                        )
                    )
                elif r["start_degree"] < 1000:
                    ids.append(
                        (
                            r["start_id"],
                            r["score"],
                            sq_index,
                            r["start_label"],
                            "rel_start",
                            r["start_degree"],
                        )
                    )
            return ids

    # Run queries concurrently
    tasks = []
    sq_list = list(subquery_vectors.keys())
    for i, sq in enumerate(sq_list):
        vec = subquery_vectors[sq]
        for idx in node_indexes:
            tasks.append(query_node_index(idx, vec, i))
        for idx in rel_indexes:
            tasks.append(query_rel_index(idx, vec, i))

    # Phase 2: Topological Deduplication and Score Merging
    # Dictionary mapping node_id -> (max_score, list_of_subquery_indices)
    global_registry: dict[str, tuple[float, list[int], str, set[str], int]] = {}

    if tasks:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"Error executing vector query: {res}")
            elif isinstance(res, list):
                for node_id, score, sq_idx, label, source, degree in res:
                    if node_id not in global_registry:
                        global_registry[node_id] = (
                            score,
                            [sq_idx],
                            label,
                            {source},
                            degree,
                        )
                    else:
                        (
                            current_max_score,
                            origins,
                            existing_label,
                            sources,
                            existing_degree,
                        ) = global_registry[node_id]
                        new_max_score = max(current_max_score, score)
                        if sq_idx not in origins:
                            origins.append(sq_idx)
                        sources.add(source)
                        global_registry[node_id] = (
                            new_max_score,
                            origins,
                            existing_label,
                            sources,
                            existing_degree,
                        )

    # Phase 3: Two-Phase Filtering
    final_anchors = set()

    # Phase A: Guaranteed Coverage (top-3 per subquery for better coverage)
    phase_a_candidates = []
    seen_in_phase_a = set()
    TOP_PER_SUBQUERY = 1
    for i in range(len(sq_list)):
        candidates = [
            (node_id, score) for node_id, (score, origins, _, _, _) in global_registry.items() if i in origins
        ]
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            for node_id, score in candidates[:TOP_PER_SUBQUERY]:
                if node_id not in seen_in_phase_a:
                    seen_in_phase_a.add(node_id)
                    phase_a_candidates.append((node_id, score))

    # If subqueries > k, we sort the best candidates by score and take top k
    phase_a_candidates.sort(key=lambda x: x[1], reverse=True)
    for node_id, _ in phase_a_candidates[:k]:
        final_anchors.add(node_id)

    # Phase B: Global Greedy Filling
    remaining_budget = k - len(final_anchors)
    if remaining_budget > 0:
        remaining_candidates = [
            (node_id, score) for node_id, (score, _, _, _, _) in global_registry.items() if node_id not in final_anchors
        ]
        remaining_candidates.sort(key=lambda x: x[1], reverse=True)
        for node_id, _ in remaining_candidates[:remaining_budget]:
            final_anchors.add(node_id)

    anchor_ids_list = list(final_anchors)

    # Extract scores for returned anchors
    anchor_scores = {node_id: global_registry[node_id][0] for node_id in anchor_ids_list}
    anchor_info = {
        node_id: (
            global_registry[node_id][2],
            list(global_registry[node_id][3]),
            global_registry[node_id][4],
        )
        for node_id in anchor_ids_list
    }

    logger.info(f"Stage 2 completed. Found {len(anchor_ids_list)} unique anchor nodes.")
    return {
        "anchor_ids": anchor_ids_list,
        "vectors": subquery_vectors,
        "scores": anchor_scores,
        "info": anchor_info,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)

    config = load_config()

    print("Testing Stage 2: Sequential Pipeline 1 -> 2")

    async def main():
        uri = "bolt://127.0.0.1:7687"
        driver = init_driver(uri, config.neo4j.user, config.neo4j.password)
        try:
            query = "Which strains of lactic acid bacteria produce ACE (angiotensin-converting enzyme) inhibitors, and what specific tripeptides are decrypted from casein to lower blood pressure?"
            K = 20
            L = 200
            print(f"\nquery: {query}, K = {K}, L = {L}")
            # Stage 1
            print("\n--- STAGE 1 ---")
            t0 = time.time()
            # subqueries = await decompose_query(query)
            subqueries = [
                "starter culture strains for curd synthesize GABA",
                "starter culture strains for curd synthesize bioavailable forms of GABA",
                "starter culture strains for curd synthesize stable forms of GABA",
                "lactobacilli in curd affect GABA preservation",
                "thermophilic streptococcus in curd affect GABA preservation",
                "lactobacilli in curd affect GABA assimilation by the body",
                "thermophilic streptococcus in curd affect GABA assimilation by the body",
            ]

            t1 = time.time()
            print(f"Stage 1 completed in {t1 - t0:.2f}s")
            print(f"Subqueries: {subqueries}")

            # Stage 2
            print("\n--- STAGE 2 ---")
            t2 = time.time()
            result = await find_anchors_for_subqueries(driver, subqueries, k=K, l=L)
            t3 = time.time()
            print(f"Stage 2 completed in {t3 - t2:.2f}s")
            print(f"Found {len(result['anchor_ids'])} unique anchors.")
            print(f"Vectors mapped: {list(result['vectors'].keys())}")

            # Извлекаем имена для наглядного дебага
            async with driver.session() as session:
                res = await session.run(
                    "MATCH (n) WHERE elementId(n) IN $ids RETURN elementId(n) AS id, coalesce(n.name, n.id, labels(n)[0]) AS name",
                    parameters={"ids": result["anchor_ids"]},
                )
                names = {r["id"]: r["name"] async for r in res}

            anchor_scores = result.get("scores", {})
            anchor_info = result.get("info", {})

            # Сортируем по убыванию score
            sorted_anchor_ids = sorted(
                result["anchor_ids"],
                key=lambda x: anchor_scores.get(x, 0.0),
                reverse=True,
            )

            print("Anchor Names and Scores:")
            for node_id in sorted_anchor_ids:
                name = names.get(node_id, node_id)
                score = anchor_scores.get(node_id, 0.0)
                label, sources, degree = anchor_info.get(node_id, ("Unknown", [], 0))
                sources_str = ", ".join(sources)
                print(f"  - [{label}] {name} (Score: {score:.4f}, Degree: {degree}, Source: {sources_str})")
        finally:
            await close_driver()

    asyncio.run(main())

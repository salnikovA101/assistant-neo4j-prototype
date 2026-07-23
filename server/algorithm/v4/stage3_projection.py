import logging
import math
import time
import uuid
import json
import os
from neo4j import AsyncDriver

logger = logging.getLogger(__name__)

CACHE_FILE = os.path.join(os.path.dirname(__file__), "cache_metadata.json")


async def get_global_metadata(
    driver: AsyncDriver,
    query_vectors: list[list[float]],
    anchor_ids: list[str] | None = None,
    max_hops: int = 2,
) -> tuple[int, float, float]:
    d_max = None

    # 1. Читаем D_max из кэша (если есть)
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r") as f:
                data = json.load(f)
                if "d_max" in data:
                    d_max = data["d_max"]
                    logger.info(f"Loaded D_max from cache: {d_max}")
        except Exception as e:
            logger.warning(f"Failed to read cache file: {e}")

    async with driver.session() as session:
        # 2. Если D_max нет в кэше, считаем и сохраняем
        if d_max is None:
            logger.info("D_max cache miss. Fetching from DB (this may take a while)...")
            d_res = await session.run("MATCH (n) RETURN COUNT { (n)--() } AS deg ORDER BY deg DESC LIMIT 1")
            d_rec = await d_res.single()
            d_max = d_rec["deg"] if d_rec else 1

            try:
                with open(CACHE_FILE, "w") as f:
                    json.dump({"d_max": d_max}, f)
                    logger.info("Saved D_max to cache.")
            except Exception as e:
                logger.warning(f"Failed to write cache file: {e}")

        # 3. ВСЕГДА считаем S_max и S_min для текущего вектора запроса
        s_max = 1.0
        s_min = 0.0
        if anchor_ids:
            s_query = f"""
            MATCH (anchor)
            WHERE elementId(anchor) IN $anchor_ids
            MATCH (anchor)-[*0..{max_hops}]-(n)
            WITH collect(DISTINCT n) AS nodes
            UNWIND nodes AS s
            MATCH (s)-[r]-(t)
            WHERE t IN nodes AND r.evidence_embedding IS NOT NULL
            WITH apoc.coll.max([q_vec IN $query_vectors | vector.similarity.cosine(q_vec, r.evidence_embedding)]) AS max_sim
            RETURN max(max_sim) AS S_max, min(max_sim) AS S_min
            """
            s_params = {"query_vectors": query_vectors, "anchor_ids": anchor_ids}
        else:
            s_query = """
            MATCH (s)-[r]-(t)
            WHERE r.evidence_embedding IS NOT NULL
            WITH apoc.coll.max([q_vec IN $query_vectors | vector.similarity.cosine(q_vec, r.evidence_embedding)]) AS max_sim
            RETURN max(max_sim) AS S_max, min(max_sim) AS S_min
            """
            s_params = {"query_vectors": query_vectors}
        s_res = await session.run(s_query, parameters=s_params)
        s_rec = await s_res.single()
        if s_rec and s_rec["S_max"] is not None:
            s_max = s_rec["S_max"]
            s_min = s_rec["S_min"]

    return d_max, s_max, s_min


async def create_dynamic_projection_and_filter(
    driver: AsyncDriver,
    anchor_ids: list[str],
    subquery_vectors: dict[str, list[float]],
    min_compression_ratio: float = 0.20,
    max_compression_ratio: float = 1.0,
    base_alpha: float = 0.2,
    base_beta: float = 0.1,
    max_hops: int = 3,
    ppr_mass_target: float = 0.92,
) -> dict:
    """
    Unified Pipeline (Stage 3 + 4 combined graph projection):
    1. Finds global S_max, S_min, D_max (S bounds scoped to anchor neighborhood).
    2. Projects the k-hop neighborhood around anchors, calculating sim, topological penalty, and confidence penalty.
    3. Runs PPR using the penalized affinity_weight.
    4. Filters nodes via Top-P and creates a subgraph.
    5. Returns allowed nodes and the filtered graph name.
    """
    if not anchor_ids:
        logger.error("No anchor IDs provided for projection.")
        return {"allowed_ids": [], "ppr_scores": {}, "graph_name": None}

    query_vectors = list(subquery_vectors.values())
    if not query_vectors:
        logger.error("No query vectors provided for projection.")
        return {"allowed_ids": [], "ppr_scores": {}, "graph_name": None}

    logger.info("--- UNIFIED PIPELINE: GRAPH PROJECTION & FILTERING ---")
    t0 = time.time()

    # 1. Fetch Global Metadata (D_max, S_max, S_min)
    d_max, s_max, s_min = await get_global_metadata(driver, query_vectors, anchor_ids, max_hops)

    # Stretch cosine range [s_min, s_max] -> [TARGET_MIN, TARGET_MAX] before penalties
    TARGET_MIN = 0.01
    TARGET_MAX = 1.0
    s_delta = max(0.01, s_max - s_min)
    log_denom_global = math.log((d_max + 1) ** 2) if d_max > 0 else 1.0

    logger.info(
        f"Global Metadata: D_max={d_max}, S_max={s_max:.4f}, S_min={s_min:.4f}, S_delta={s_delta:.4f} "
        f"(projection scoped to {max_hops}-hop neighborhood of {len(anchor_ids)} anchors)"
    )
    logger.info(
        f"Penalty Params: base_alpha={base_alpha:.4f}, base_beta={base_beta:.4f}, "
        f"stretch=[{TARGET_MIN}, {TARGET_MAX}], log_denom={log_denom_global:.4f}"
    )

    t1 = time.time()
    logger.info(f"Metadata fetch took {t1 - t0:.2f}s")

    # 2. Project Graph with fully penalized Affinity Weight
    raw_graph_name = f"raw_graph_{uuid.uuid4().hex[:8]}"

    node_query = f"""
    MATCH (anchor)
    WHERE elementId(anchor) IN $anchor_ids
    MATCH (anchor)-[*0..{max_hops}]-(n)
    RETURN DISTINCT id(n) AS id
    """

    rel_query = f"""
    MATCH (anchor)
    WHERE elementId(anchor) IN $anchor_ids
    MATCH (anchor)-[*0..{max_hops}]-(n)
    WITH collect(DISTINCT n) AS nodes
    UNWIND nodes AS s
    MATCH (s)-[r]-(t)
    WHERE t IN nodes
    WITH s, r, t,
         apoc.coll.max([q_vec IN $query_vectors |
            CASE WHEN r.evidence_embedding IS NOT NULL AND size(r.evidence_embedding) > 0
            THEN vector.similarity.cosine(q_vec, r.evidence_embedding)
            ELSE 0.0 END
         ]) AS max_sim,
         apoc.node.degree(s) AS deg_s,
         apoc.node.degree(t) AS deg_t,
         coalesce(r.confidence, 1.0) AS conf

    WITH s, r, t, coalesce(max_sim, 0.0) AS sim, deg_s, deg_t, conf

    // Stretch local cosine range to contrast range [target_min, target_max]
    WITH s, t, sim, deg_s, deg_t, conf,
         $target_min + ($target_max - $target_min) * (sim - $s_min) / $s_delta AS stretched_sim

    // Penalties (all in ~[0, 1] scale against stretched_sim)
    WITH s, t, stretched_sim,
         $base_alpha * (log((deg_s + 1.0) * (deg_t + 1.0)) / $log_denom_global) AS penalty_topo,
         $base_beta * (1.0 - conf) AS penalty_conf

    WITH s, t, stretched_sim, penalty_topo, penalty_conf,
         stretched_sim - penalty_topo - penalty_conf AS penalized_sim

    // Affinity must be >= 0 for PPR. If it falls below 0 due to penalties, it's virtually disconnected.
    WITH id(s) AS source, id(t) AS target,
         CASE WHEN penalized_sim < 0.0 THEN 0.0 ELSE penalized_sim END AS affinity_weight

    RETURN source, target, affinity_weight, 1.0 - affinity_weight AS distance_weight
    """

    project_query = """
    CALL gds.graph.project.cypher(
        $graph_name,
        $node_query,
        $rel_query,
        {
            parameters: {
                anchor_ids: $anchor_ids,
                query_vectors: $query_vectors,
                s_min: $s_min,
                s_delta: $s_delta,
                target_min: $target_min,
                target_max: $target_max,
                base_alpha: $base_alpha,
                base_beta: $base_beta,
                log_denom_global: $log_denom_global
            },
            validateRelationships: false
        }
    ) YIELD graphName, nodeCount, relationshipCount
    RETURN graphName, nodeCount, relationshipCount
    """

    async with driver.session() as session:
        try:
            res = await session.run(
                project_query,
                parameters={
                    "graph_name": raw_graph_name,
                    "node_query": node_query,
                    "rel_query": rel_query,
                    "anchor_ids": anchor_ids,
                    "query_vectors": query_vectors,
                    "s_min": s_min,
                    "s_delta": s_delta,
                    "target_min": TARGET_MIN,
                    "target_max": TARGET_MAX,
                    "base_alpha": base_alpha,
                    "base_beta": base_beta,
                    "log_denom_global": log_denom_global,
                },
            )
            record = await res.single()
            if record:
                logger.info(
                    f"Projected Unified Graph '{record['graphName']}' "
                    f"with {record['nodeCount']} nodes and {record['relationshipCount']} relationships."
                )
        except Exception as e:
            logger.error(f"GDS Unified Projection failed: {e}")
            return {"allowed_ids": [], "ppr_scores": {}, "graph_name": None}

    t2 = time.time()
    logger.info(f"Graph projection took {t2 - t1:.2f}s")

    # 3. Run PageRank Mutate
    internal_anchors = []
    async with driver.session() as session:
        res = await session.run(
            "MATCH (n) WHERE elementId(n) IN $ids RETURN id(n) AS internal_id", parameters={"ids": anchor_ids}
        )
        internal_anchors = [r["internal_id"] async for r in res]

    ppr_mutate_query = """
    CALL gds.pageRank.mutate(
        $graph_name,
        {
            mutateProperty: 'pprScore',
            relationshipWeightProperty: 'affinity_weight',
            sourceNodes: $source_nodes,
            scaler: 'L1Norm'
        }
    ) YIELD nodePropertiesWritten
    """
    async with driver.session() as session:
        try:
            await session.run(
                ppr_mutate_query, parameters={"graph_name": raw_graph_name, "source_nodes": internal_anchors}
            )
        except Exception as e:
            logger.error(f"GDS PageRank mutate failed: {e}")
            await drop_projection(driver, raw_graph_name)
            return {"allowed_ids": [], "ppr_scores": {}, "graph_name": None}

    # 4. Stream PPR and Top-P Filter
    stream_query = """
    CALL gds.graph.nodeProperty.stream($graph_name, 'pprScore')
    YIELD nodeId, propertyValue
    RETURN nodeId, propertyValue AS score
    ORDER BY score DESC
    """
    allowed_internal_ids = []
    allowed_element_ids = []
    ppr_scores_by_eid = {}
    min_ppr = 0.0

    async with driver.session() as session:
        try:
            res = await session.run(stream_query, parameters={"graph_name": raw_graph_name})
            nodes_with_scores = [(r["nodeId"], r["score"]) async for r in res]

            if not nodes_with_scores:
                logger.warning("No nodes found in PageRank stream.")
                await drop_projection(driver, raw_graph_name)
                return {"allowed_ids": [], "ppr_scores": {}, "graph_name": None}

            if len(nodes_with_scores) < 2:
                allowed_internal_ids = [n[0] for n in nodes_with_scores]
                min_ppr = nodes_with_scores[-1][1] if nodes_with_scores else 0.0
                min_k = len(nodes_with_scores)
            else:
                N = len(nodes_with_scores)
                min_k = int(N * min_compression_ratio)
                min_k = max(1, min_k)

                max_k = int(N * max_compression_ratio)
                max_k = min(N, max_k)
                max_k = max(min_k, max_k)

                # 1. Считаем кумулятивную сумму
                cum_sums = []
                current_sum = 0.0
                for nid, score in nodes_with_scores:
                    current_sum += score
                    cum_sums.append(current_sum)

                if min_k == max_k or cum_sums[min_k - 1] == cum_sums[max_k - 1]:
                    best_k = max_k
                else:
                    max_dist = -float("inf")
                    best_k = max_k

                    c_min = cum_sums[min_k - 1]
                    c_max = cum_sums[max_k - 1]

                    # 2. Ищем "колено" локально между min_k и max_k
                    for i in range(min_k - 1, max_k):
                        x_norm = (i - (min_k - 1)) / (max_k - 1 - (min_k - 1))
                        y_norm = (cum_sums[i] - c_min) / (c_max - c_min)

                        # На вогнутой кривой дуга ВЫШЕ прямой y = x
                        dist = y_norm - x_norm

                        if dist > max_dist:
                            max_dist = dist
                            best_k = i + 1

                # Adaptive: expand knee if too little PPR mass retained
                total_mass = cum_sums[-1]
                if total_mass > 0 and ppr_mass_target < 1.0:
                    mass_at_k = cum_sums[best_k - 1] / total_mass
                    if mass_at_k < ppr_mass_target and best_k < max_k:
                        for i in range(best_k, max_k):
                            if cum_sums[i] / total_mass >= ppr_mass_target:
                                best_k = i + 1
                                break
                        else:
                            best_k = max_k

                allowed_internal_ids = [n[0] for n in nodes_with_scores[:best_k]]
                min_ppr = nodes_with_scores[best_k - 1][1]

            logger.info(
                f"Cumulative Kneedle filter retained {len(allowed_internal_ids)} / {len(nodes_with_scores)} nodes (min_k={min_k}, max_k={max_k if 'max_k' in locals() else 'N/A'}, min_ppr={min_ppr:.6f})."
            )

        except Exception as e:
            logger.error(f"Failed to stream and filter PPR: {e}")
            await drop_projection(driver, raw_graph_name)
            return {"allowed_ids": [], "ppr_scores": {}, "graph_name": None}

    # 5. Create Filtered Subgraph
    filtered_graph_name = f"filtered_graph_{uuid.uuid4().hex[:8]}"
    subgraph_query = f"""
    CALL gds.beta.graph.project.subgraph(
        $filtered_graph_name,
        $raw_graph_name,
        'n.pprScore >= {min_ppr}',
        '*'
    ) YIELD graphName, nodeCount, relationshipCount
    RETURN graphName, nodeCount, relationshipCount
    """
    async with driver.session() as session:
        try:
            res = await session.run(
                subgraph_query,
                parameters={"filtered_graph_name": filtered_graph_name, "raw_graph_name": raw_graph_name},
            )
            rec = await res.single()
            logger.info(
                f"Created filtered subgraph '{filtered_graph_name}' with {rec['nodeCount']} nodes and {rec['relationshipCount']} relationships."
            )

            # Map allowed internal IDs to elementIds
            res_map = await session.run(
                "MATCH (n) WHERE id(n) IN $ids RETURN id(n) AS iid, elementId(n) AS eid",
                parameters={"ids": allowed_internal_ids},
            )
            eid_map = {r["iid"]: r["eid"] async for r in res_map}

            # Populate outputs
            for nid, score in nodes_with_scores:
                if score >= min_ppr:
                    eid = eid_map.get(nid)
                    if eid:
                        allowed_element_ids.append(eid)
                        ppr_scores_by_eid[eid] = score

            # Percentile-rank normalize: lowest rank -> 1/N, highest rank -> 1.0
            if ppr_scores_by_eid:
                sorted_eids = sorted(ppr_scores_by_eid, key=lambda e: ppr_scores_by_eid[e])
                n = len(sorted_eids)
                ppr_scores_by_eid = {eid: (rank + 1) / n for rank, eid in enumerate(sorted_eids)}

        except Exception as e:
            logger.error(f"Failed to create filtered subgraph: {e}")
            await drop_projection(driver, raw_graph_name)
            return {"allowed_ids": [], "ppr_scores": {}, "graph_name": None}

    # Drop the raw graph to save memory, keeping only the filtered one for Stage 4
    await drop_projection(driver, raw_graph_name)

    t3 = time.time()
    logger.info(f"Filtering and subgraph creation took {t3 - t2:.2f}s")

    return {"allowed_ids": allowed_element_ids, "ppr_scores": ppr_scores_by_eid, "graph_name": filtered_graph_name}


async def drop_projection(driver: AsyncDriver, graph_name: str) -> bool:
    if not graph_name:
        return False
    async with driver.session() as session:
        try:
            res = await session.run(
                "CALL gds.graph.drop($graph_name, false) YIELD graphName", parameters={"graph_name": graph_name}
            )
            record = await res.single()
            if record:
                logger.info(f"Successfully dropped graph '{graph_name}'.")
                return True
        except Exception as e:
            logger.error(f"Failed to drop graph '{graph_name}': {e}")
    return False

import logging
import math
import time

from neo4j import AsyncDriver

from server.algorithm.v4.stage3_projection import drop_projection

logger = logging.getLogger(__name__)


async def find_ppr_modulated_chains(
    driver: AsyncDriver,
    graph_name: str,
    allowed_ids: list[str],
    beam_width: int = 15,
    chains_per_source: int = 5,
    max_total_chains: int = 500,
) -> list[dict]:
    """
    Stage 4 (PPR Max-Product Chains):
    Builds the PPR-modulated transition matrix M_ij = W_ij * sqrt(P_i * P_j) over the
    Stage 3 projection (W_ij = affinity_weight, P_i/P_j = pprScore) and searches for the
    strongest 3-hop chains s -> k1 -> k2 -> t via a beam-limited Max-Product expansion.
    Both s and t range over ANY node of the projection (not restricted to anchors), since
    the anchor-scoping already happened when Stage 3 built the 3-hop projection.
    This is equivalent to computing bounded rows/entries of M^{\\otimes 3} where
    (A ⊗ B)_ij = max_k(A_ik * B_kj).
    """
    if not graph_name:
        logger.error("No graph name provided to Stage 4 (PPR Max-Product Chains).")
        return []

    logger.info("--- UNIFIED PIPELINE: PATH SEARCH (PPR MAX-PRODUCT CHAINS) ---")
    logger.info(f"Allowed node pool size (from Stage 3): {len(allowed_ids)}")
    t0 = time.time()

    # 1. Stream pprScore -> P_i
    node_ppr: dict[int, float] = {}
    ppr_stream_query = """
    CALL gds.graph.nodeProperty.stream($graph_name, 'pprScore')
    YIELD nodeId, propertyValue
    RETURN nodeId, propertyValue AS score
    """
    async with driver.session() as session:
        try:
            res = await session.run(ppr_stream_query, parameters={"graph_name": graph_name})
            async for r in res:
                node_ppr[r["nodeId"]] = r["score"]
        except Exception as e:
            logger.error(f"Failed to stream pprScore: {e}")
            await drop_projection(driver, graph_name)
            return []

    if not node_ppr:
        logger.warning("No nodes found while streaming pprScore.")
        await drop_projection(driver, graph_name)
        return []

    t1 = time.time()
    logger.info(f"Streamed pprScore for {len(node_ppr)} nodes in {t1 - t0:.3f}s")

    # 2. Stream affinity_weight -> W_ij, build M_ij = W_ij * sqrt(P_i * P_j)
    rel_stream_query = """
    CALL gds.graph.relationshipProperty.stream($graph_name, 'affinity_weight')
    YIELD sourceNodeId, targetNodeId, propertyValue AS weight
    RETURN sourceNodeId, targetNodeId, weight
    """
    adj: dict[int, list[tuple[int, float]]] = {}
    edge_count = 0
    async with driver.session() as session:
        try:
            res = await session.run(rel_stream_query, parameters={"graph_name": graph_name})
            async for r in res:
                u, v, w = r["sourceNodeId"], r["targetNodeId"], r["weight"]
                if u == v:
                    continue
                p_u = max(node_ppr.get(u, 0.0), 0.0)
                p_v = max(node_ppr.get(v, 0.0), 0.0)
                m_uv = w * math.sqrt(p_u * p_v)
                adj.setdefault(u, []).append((v, m_uv))
                edge_count += 1
        except Exception as e:
            logger.error(f"Failed to stream affinity_weight: {e}")
            await drop_projection(driver, graph_name)
            return []

    if not adj:
        logger.warning("No edges found while streaming affinity_weight.")
        await drop_projection(driver, graph_name)
        return []

    # Sort each adjacency list descending by M_ij once, so beam slicing keeps the strongest neighbors.
    for node in adj:
        adj[node].sort(key=lambda x: x[1], reverse=True)

    t2 = time.time()
    logger.info(f"Built M matrix: {len(adj)} source nodes, {edge_count} directed edges in {t2 - t1:.3f}s")

    # 3. Beam-limited 3-hop Max-Product expansion from every source node (s -> k1 -> k2 -> t)
    raw_chains = []
    for s, s_neighbors in adj.items():
        # best (score, k1, k2) seen so far per distinct target t: this is the max_k reduction
        # that matrix exponentiation performs, restricted to the beam-visited (k1, k2) pairs.
        best_per_target: dict[int, tuple[float, int, int]] = {}

        for k1, m_sk1 in s_neighbors[:beam_width]:
            k1_neighbors = adj.get(k1)
            if not k1_neighbors:
                continue
            for k2, m_k1k2 in k1_neighbors[:beam_width]:
                if k2 == s:
                    continue
                k2_neighbors = adj.get(k2)
                if not k2_neighbors:
                    continue
                partial_score = m_sk1 * m_k1k2
                for t, m_k2t in k2_neighbors[:beam_width]:
                    if t == s or t == k1:
                        continue
                    score = partial_score * m_k2t
                    existing = best_per_target.get(t)
                    if existing is None or score > existing[0]:
                        best_per_target[t] = (score, k1, k2)

        if not best_per_target:
            continue

        top_targets = sorted(best_per_target.items(), key=lambda x: x[1][0], reverse=True)[:chains_per_source]
        for t, (score, k1, k2) in top_targets:
            raw_chains.append({"source": s, "k1": k1, "k2": k2, "target": t, "score": score})

    t3 = time.time()
    logger.info(f"Max-Product expansion produced {len(raw_chains)} candidate chains in {t3 - t2:.3f}s")

    if not raw_chains:
        await drop_projection(driver, graph_name)
        return []

    # 4. Merge & truncate globally
    raw_chains.sort(key=lambda x: x["score"], reverse=True)
    raw_chains = raw_chains[:max_total_chains]
    for idx, chain in enumerate(raw_chains):
        chain["path_index"] = idx

    logger.info(f"Retained top {len(raw_chains)} chains after global truncation (max_total_chains={max_total_chains}).")

    # We only need the real Neo4j graph for reconstruction from here on.
    await drop_projection(driver, graph_name)

    # 5. Reconstruct real DB nodes/relationships for each chain [s, k1, k2, t]
    reconstruct_query = """
    UNWIND $chains AS chain
    WITH chain, [chain.source, chain.k1, chain.k2, chain.target] AS nodeIds

    // 1. Fetch ordered nodes
    MATCH (n) WHERE id(n) IN nodeIds
    WITH chain, nodeIds, collect({internal_id: id(n), id: elementId(n), labels: labels(n), name: n.name}) AS unordered_nodes
    WITH chain, nodeIds, [nid IN nodeIds |
        [x IN unordered_nodes WHERE x.internal_id = nid][0]
    ] AS ordered_nodes

    // 2. Fetch relationships between adjacent nodes
    UNWIND range(0, size(nodeIds)-2) AS i
    MATCH (a)-[r]-(b)
    WHERE id(a) = nodeIds[i] AND id(b) = nodeIds[i+1]

    // Aggregate ALL parallel relationships for this hop
    WITH chain, ordered_nodes, i, collect(r) AS hop_rels
    ORDER BY chain.path_index, i

    // Package them nicely
    WITH chain, ordered_nodes, collect([rel IN hop_rels | {
        type: type(rel),
        rel_id: elementId(rel),
        evidence: coalesce(rel.evidence, '')
    }]) AS ordered_rels

    RETURN chain.score AS score, ordered_nodes AS nodes, ordered_rels AS relationships
    """

    final_paths = []
    BATCH_SIZE = 100
    for i in range(0, len(raw_chains), BATCH_SIZE):
        batch = raw_chains[i : i + BATCH_SIZE]
        async with driver.session() as session:
            try:
                res = await session.run(reconstruct_query, parameters={"chains": batch})
                async for r in res:
                    score = r["score"]
                    final_paths.append(
                        {
                            "totalCost": -score,
                            "chain_score": score,
                            "nodes": r["nodes"],
                            "relationships": r["relationships"],
                        }
                    )
            except Exception as e:
                logger.error(f"Error reconstructing chain batch: {e}")

    t4 = time.time()
    logger.info(f"Chain reconstruction took {t4 - t3:.3f}s")
    logger.info(f"Successfully reconstructed {len(final_paths)} PPR max-product chains.")
    return final_paths

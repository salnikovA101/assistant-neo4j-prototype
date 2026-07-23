import logging
import math
import time
import networkx as nx
from neo4j import AsyncDriver

logger = logging.getLogger(__name__)

async def find_topological_paths(
    driver: AsyncDriver,
    graph_name: str,
    allowed_ids: list[str],
    anchor_ids: list[str],
    s2_scores: dict[str, float],
    ppr_scores: dict[str, float],
    k_limit: int = 20,
    iterations: int = 3,
    penalty_m: float = 2.0,
    penalty_c_mult: float = 1.0,
    max_path_vertices: int = 20,
) -> list[dict]:
    """
    Unified Stage 4: Runs Edge Penalization SSSP (NetworkX) from top anchors.
    """
    if not graph_name:
        logger.error("No graph name provided to Stage 4.")
        return []

    if not anchor_ids or len(anchor_ids) < 2:
        logger.warning("Not enough anchors for path search.")
        await drop_projection(driver, graph_name)
        return []

    logger.info("--- UNIFIED PIPELINE: PATH SEARCH (NETWORKX EDGE PENALIZATION) ---")
    t0 = time.time()

    t1 = time.time()

    # Convert anchor elementIds to internal IDs
    internal_ids_map = {}
    async with driver.session() as session:
        res = await session.run(
            "MATCH (n) WHERE elementId(n) IN $ids RETURN elementId(n) AS eid, id(n) AS internal_id",
            parameters={"ids": allowed_ids},
        )
        async for r in res:
            internal_ids_map[r["eid"]] = r["internal_id"]

    ppr_by_internal = {
        internal_ids_map[eid]: score for eid, score in ppr_scores.items() if eid in internal_ids_map
    }

    # Calculate Combined Scores and select Top K anchors
    anchor_scores = []
    for a_id in anchor_ids:
        if a_id in internal_ids_map:
            int_id = internal_ids_map[a_id]
            s2 = s2_scores.get(a_id, 0.0)
            ppr = ppr_scores.get(a_id, 0.0)
            combined = s2 * ppr
            anchor_scores.append((int_id, combined))

    # Sort descending by Combined Score
    anchor_scores.sort(key=lambda x: x[1], reverse=True)
    target_anchors = [a[0] for a in anchor_scores[:k_limit]]

    # 1. Stream Graph from GDS to NetworkX
    logger.info(f"Streaming graph '{graph_name}' from GDS to NetworkX...")
    stream_query = """
    CALL gds.graph.relationshipProperty.stream($graph_name, 'affinity_weight')
    YIELD sourceNodeId, targetNodeId, propertyValue AS affinity
    RETURN sourceNodeId, targetNodeId, affinity
    """

    G = nx.DiGraph()
    total_weight = 0.0
    edge_count = 0
    EPS = 1e-9

    async with driver.session() as session:
        res = await session.run(stream_query, parameters={"graph_name": graph_name})
        async for r in res:
            u, v, affinity = r["sourceNodeId"], r["targetNodeId"], r["affinity"]
            combined = max(
                affinity * math.sqrt(ppr_by_internal.get(u, EPS) * ppr_by_internal.get(v, EPS)),
                EPS,
            )
            w = -math.log(combined)
            # Add or update edge keeping minimum weight for parallel edges
            if G.has_edge(u, v):
                G[u][v]["weight"] = min(G[u][v]["weight"], w)
                G[u][v]["original_weight"] = min(G[u][v]["original_weight"], w)
            else:
                G.add_edge(u, v, weight=w, original_weight=w)
            total_weight += w
            edge_count += 1

    if edge_count == 0:
        w_avg = 0.5
    else:
        w_avg = total_weight / edge_count

    C = w_avg * penalty_c_mult
    logger.info(
        f"Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} unique edges. W_avg={w_avg:.4f}, Penalty C={C:.4f}, M={penalty_m}"
    )

    raw_paths = []
    path_idx = 0

    # 2. Iterative Edge Penalization Loop
    for iter_idx in range(iterations):
        logger.info(f"Iteration {iter_idx + 1}/{iterations}...")
        for i in range(len(target_anchors)):
            source_anchor = target_anchors[i]
            target_list = target_anchors[i + 1 :]

            if not target_list:
                break

            if source_anchor not in G:
                continue

            # SSSP for this source to all targets
            try:
                # Get paths to all nodes that are reachable
                paths = nx.single_source_dijkstra_path(G, source_anchor, weight="weight")

                for target_anchor in target_list:
                    if target_anchor in paths:
                        path_nodes = paths[target_anchor]
                        if len(path_nodes) < 2:
                            continue
                        if len(path_nodes) > max_path_vertices:
                            continue

                        path_cost = sum(
                            G[path_nodes[j]][path_nodes[j + 1]]["original_weight"]
                            for j in range(len(path_nodes) - 1)
                        )

                        raw_paths.append(
                            {
                                "path_index": path_idx,
                                "source": source_anchor,
                                "target": target_anchor,
                                "totalCost": path_cost,
                                "nodeIds": path_nodes,
                            }
                        )
                        path_idx += 1

                        # Penalize edges
                        for j in range(len(path_nodes) - 1):
                            u, v = path_nodes[j], path_nodes[j + 1]
                            old_w = G[u][v]["weight"]
                            new_w = old_w * penalty_m + C
                            G[u][v]["weight"] = new_w

                            # In an undirected knowledge graph, penalize the reverse direction too
                            if G.has_edge(v, u):
                                G[v][u]["weight"] = new_w
            except Exception as e:
                logger.error(f"Error in NetworkX Dijkstra: {e}")

    t2 = time.time()
    logger.info(f"NetworkX iterations took {t2 - t1:.3f}s. Found {len(raw_paths)} raw paths.")

    # Drop the graph from memory
    await drop_projection(driver, graph_name)

    if not raw_paths:
        return []

    # 3. Reconstruct paths with real DB nodes and relationships
    reconstruct_query = """
    UNWIND $paths AS path_obj
    WITH path_obj, path_obj.nodeIds AS nodeIds

    // 1. Fetch ordered nodes
    MATCH (n) WHERE id(n) IN nodeIds
    WITH path_obj, nodeIds, collect({internal_id: id(n), id: elementId(n), labels: labels(n), name: n.name}) AS unordered_nodes
    WITH path_obj, nodeIds, [nid IN nodeIds |
        [x IN unordered_nodes WHERE x.internal_id = nid][0]
    ] AS ordered_nodes

    // 2. Fetch relationships between adjacent nodes
    UNWIND range(0, size(nodeIds)-2) AS i
    MATCH (a)-[r]-(b)
    WHERE id(a) = nodeIds[i] AND id(b) = nodeIds[i+1]

    // Aggregate ALL parallel relationships for this hop
    WITH path_obj, ordered_nodes, i, collect(r) AS hop_rels
    ORDER BY path_obj.path_index, i

    // Package them nicely
    WITH path_obj, ordered_nodes, collect([rel IN hop_rels | {
        type: type(rel),
        rel_id: elementId(rel),
        evidence: coalesce(rel.evidence, '')
    }]) AS ordered_rels

    RETURN path_obj.totalCost AS totalCost, ordered_nodes AS nodes, ordered_rels AS relationships
    """

    final_paths = []
    BATCH_SIZE = 100
    for i in range(0, len(raw_paths), BATCH_SIZE):
        batch = raw_paths[i : i + BATCH_SIZE]
        async with driver.session() as session:
            try:
                res = await session.run(reconstruct_query, parameters={"paths": batch})
                async for r in res:
                    final_paths.append(
                        {"totalCost": r["totalCost"], "nodes": r["nodes"], "relationships": r["relationships"]}
                    )
            except Exception as e:
                logger.error(f"Error reconstructing path batch: {e}")

    t3 = time.time()
    logger.info(f"Path Reconstruction took {t3 - t2:.3f}s")
    logger.info(f"Successfully reconstructed {len(final_paths)} paths.")
    return final_paths


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

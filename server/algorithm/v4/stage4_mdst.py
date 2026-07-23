import asyncio
import logging
import uuid
import math
import time
from neo4j import AsyncDriver

from server.core.db import init_driver, close_driver
from server.utils.config import load_config
from server.algorithm.v4.stage3_projection import create_dynamic_projection_and_filter, drop_projection

logger = logging.getLogger(__name__)

async def find_steiner_tree_edges(
    driver: AsyncDriver, 
    allowed_node_ids: list[str], 
    anchor_ids: list[str], 
    subquery_vectors: dict[str, list[float]],
    anchor_scores: dict = None,
    ppr_scores: dict = None,
    alpha: float = 0.05, 
    beta: float = 0.35, 
) -> list[dict]:
    """
    Stage 4: Distance Calibration & MDST Search
    Finds the Minimum Spanning Steiner Tree that connects all anchors minimizing distance weights.
    """
    if not anchor_ids or len(anchor_ids) < 2:
        logger.warning("Not enough anchors for PCST search.")
        return []
    
    if anchor_scores is None:
        anchor_scores = {}
    if ppr_scores is None:
        ppr_scores = {}
        
    query_vectors = list(subquery_vectors.values())
        
    # Convert elementIds to internal IDs for GDS
    internal_ids_map = {}
    async with driver.session() as session:
        res = await session.run("MATCH (n) WHERE elementId(n) IN $ids RETURN elementId(n) AS eid, id(n) AS internal_id", parameters={"ids": allowed_node_ids})
        async for r in res:
            internal_ids_map[r["eid"]] = r["internal_id"]
            
    internal_allowed_ids = list(internal_ids_map.values())
    internal_anchors = [internal_ids_map.get(a) for a in anchor_ids if a in internal_ids_map]
    
    if len(internal_anchors) < 2:
        logger.warning("Could not map enough anchors to internal IDs.")
        return []

    t0 = time.time()
    # Dynamic Calibration Phase
    logger.info("Harvesting local graph metadata for dynamic calibration...")
    async with driver.session() as session:
        d_max_query = "MATCH (n) WHERE id(n) IN $internal_allowed_ids RETURN COUNT { (n)--() } AS d ORDER BY d DESC LIMIT 1"
        res_d = await session.run(d_max_query, parameters={"internal_allowed_ids": internal_allowed_ids})
        rec_d = await res_d.single()
        d_max_local = rec_d["d"] if rec_d else 1
        
        s_query = """
        MATCH (s)-[r]-(t)
        WHERE id(s) IN $internal_allowed_ids AND id(t) IN $internal_allowed_ids
        WITH apoc.coll.max([q_vec IN $query_vectors | 
            CASE WHEN r.evidence_embedding IS NOT NULL AND size(r.evidence_embedding) > 0 
            THEN vector.similarity.cosine(q_vec, r.evidence_embedding) 
            ELSE 0.0 END
        ]) AS max_sim
        RETURN max(max_sim) AS s_max, min(max_sim) AS s_min
        """
        res_s = await session.run(s_query, parameters={"internal_allowed_ids": internal_allowed_ids, "query_vectors": query_vectors})
        rec_s = await res_s.single()
        s_max_local = rec_s["s_max"] if rec_s and rec_s["s_max"] is not None else 0.0
        s_min_local = rec_s["s_min"] if rec_s and rec_s["s_min"] is not None else 0.0

    delta_local = s_max_local - s_min_local
    if delta_local < 0.01:
        delta_local = 0.01
        
    dyn_alpha = alpha * delta_local
    dyn_beta = beta * delta_local
    
    # We use 2 * ln(D_max_local + 1) because degree(A)*degree(B) max is D^2, and ln(D^2) = 2*ln(D)
    log_denom_local = 2.0 * math.log(d_max_local + 1.0)
    if log_denom_local <= 0.0:
        log_denom_local = 1.0

    logger.info(f"Local Metadata: D_max={d_max_local}, S_max={s_max_local:.4f}, S_min={s_min_local:.4f}, Delta={delta_local:.4f}")
    logger.info(f"Local Params: dyn_alpha={dyn_alpha:.4f}, dyn_beta={dyn_beta:.4f}, log_denom={log_denom_local:.4f}")

    # Select sourceNode and targetNodes
    # We will score anchors by S2 * PPR and pick the highest as sourceNode
    # The rest will be targetNodes.
    max_s2 = max(anchor_scores.values()) if anchor_scores else 1.0
    max_ppr = max(ppr_scores.values()) if ppr_scores else 1.0

    anchor_combined_scores = {}
    for eid in anchor_ids:
        s2_score = anchor_scores.get(eid, 0.0)
        ppr = ppr_scores.get(eid, 0.0)
        norm_s2 = s2_score / max_s2 if max_s2 > 0 else 0.0
        norm_ppr = ppr / max_ppr if max_ppr > 0 else 0.0
        internal_id = internal_ids_map.get(eid)
        if internal_id is not None:
            anchor_combined_scores[internal_id] = float(norm_s2 * norm_ppr)

    if not anchor_combined_scores:
        logger.warning("No valid anchor scores mapped. Falling back to arbitrary source.")
        source_node_id = internal_anchors[0]
        target_node_ids = internal_anchors[1:]
    else:
        # Sort internal anchors by combined score descending
        sorted_anchors = sorted(anchor_combined_scores.keys(), key=lambda k: anchor_combined_scores[k], reverse=True)
        source_node_id = sorted_anchors[0]
        target_node_ids = sorted_anchors[1:]
        # Include any anchors that didn't have scores (just in case)
        for ia in internal_anchors:
            if ia != source_node_id and ia not in target_node_ids:
                target_node_ids.append(ia)

    t1 = time.time()
    logger.info(f"MDST Step 1 (Calibration & Source Selection) took {t1 - t0:.3f}s")
    logger.info(f"MDST Source Node: {source_node_id}, Targets count: {len(target_node_ids)}")
    
    # Create Metric Subgraph (clean_subgraph)
    clean_graph_name = f"mdst_subgraph_{uuid.uuid4().hex[:8]}"
    node_query = """
    UNWIND $internal_allowed_ids AS id 
    RETURN id
    """
    rel_query = f"""
    MATCH (s)-[r]-(t)
    WHERE id(s) IN $internal_allowed_ids AND id(t) IN $internal_allowed_ids

    // 1. Group parallel edges between s and t
    WITH s, t, collect(r) AS rels,
         COUNT {{ (s)--() }} AS deg_s,
         COUNT {{ (t)--() }} AS deg_t

    // 2. Calculate array of semantic similarities for all parallel edges
    WITH s, t, rels, deg_s, deg_t,
         [r IN rels | 
             REDUCE(max_s = -1.0, q_vec IN $query_vectors | 
               CASE WHEN r.evidence_embedding IS NOT NULL AND size(r.evidence_embedding) > 0 
               THEN CASE WHEN vector.similarity.cosine(q_vec, r.evidence_embedding) > max_s
                    THEN vector.similarity.cosine(q_vec, r.evidence_embedding)
                    ELSE max_s END
               ELSE max_s END
             )
         ] AS sims

    // 3. Calculate Dijkstra distance array for these parallel edges
    WITH s, t, rels, sims, deg_s, deg_t,
         [i IN range(0, size(sims)-1) |
             (1.0 - coalesce(sims[i], 0.0)) 
             + $dyn_alpha * (log((deg_s + 1.0) * (deg_t + 1.0)) / $log_denom_local) 
             + $dyn_beta * (1.0 - coalesce(rels[i].confidence, 1.0))
         ] AS distances

    // 4. Collapse edges, returning strictly ONE edge with minimum distance
    RETURN id(s) AS source, id(t) AS target,
           apoc.coll.min(distances) AS distance_weight,
           "REL" AS type
    """

    project_query = """
    CALL gds.graph.project.cypher(
        $clean_graph_name,
        $node_query,
        $rel_query,
        {
            parameters: {
                internal_allowed_ids: $internal_allowed_ids,
                query_vectors: $query_vectors,
                dyn_alpha: $dyn_alpha,
                dyn_beta: $dyn_beta,
                log_denom_local: $log_denom_local
            },
            validateRelationships: false
        }
    ) YIELD graphName, nodeCount, relationshipCount
    RETURN graphName, nodeCount, relationshipCount
    """
    
    async with driver.session() as session:
        try:
            res = await session.run(project_query, parameters={
                "clean_graph_name": clean_graph_name,
                "node_query": node_query,
                "rel_query": rel_query,
                "internal_allowed_ids": internal_allowed_ids,
                "query_vectors": query_vectors,
                "dyn_alpha": dyn_alpha,
                "dyn_beta": dyn_beta,
                "log_denom_local": log_denom_local
            })
            rec = await res.single()
            if rec:
                logger.info(f"Created MDST subgraph '{rec['graphName']}' with {rec['nodeCount']} nodes and {rec['relationshipCount']} rels.")
        except Exception as e:
            logger.error(f"Failed to create MDST subgraph: {e}")
            return []

    t2 = time.time()
    logger.info(f"MDST Step 2 (Projection) took {t2 - t1:.3f}s")
    
    # Mutate to undirected graph
    undirected_query = """
    CALL gds.graph.relationships.toUndirected($clean_graph_name, {
        relationshipType: 'REL',
        mutateRelationshipType: 'REL_UNDIR'
    }) YIELD relationshipsWritten
    RETURN relationshipsWritten
    """
    async with driver.session() as session:
        try:
            await session.run(undirected_query, parameters={"clean_graph_name": clean_graph_name})
            logger.info("Successfully mutated graph to undirected.")
        except Exception as e:
            logger.error(f"Failed to mutate to undirected: {e}")
            await drop_projection(driver, clean_graph_name)
            return []
    
    t3 = time.time()
    logger.info(f"MDST Step 3 (toUndirected) took {t3 - t2:.3f}s")

    # Run MDST (Steiner Tree) algorithm
    mdst_query = """
    CALL gds.steinerTree.stream($clean_graph_name, {
        relationshipWeightProperty: 'distance_weight',
        sourceNode: $source_node_id,
        targetNodes: $target_node_ids,
        relationshipTypes: ['REL_UNDIR']
    })
    YIELD nodeId, parentId, weight
    WHERE nodeId <> parentId
    RETURN nodeId, parentId, weight
    """
    
    mdst_edges = []
    async with driver.session() as session:
        try:
            res = await session.run(mdst_query, parameters={
                "clean_graph_name": clean_graph_name,
                "source_node_id": source_node_id,
                "target_node_ids": target_node_ids
            })
            async for r in res:
                mdst_edges.append({
                    "child": r["nodeId"],
                    "parent": r["parentId"],
                    "weight": r["weight"]
                })
        except Exception as e:
            logger.error(f"Failed to execute MDST algorithm: {e}")
            await drop_projection(driver, clean_graph_name)
            return []
            
    t4 = time.time()
    logger.info(f"MDST Step 4 (Algorithm Execution) took {t4 - t3:.3f}s")
            
    # Drop virtual mask
    await drop_projection(driver, clean_graph_name)

    if not mdst_edges:
        logger.warning("MDST returned no edges.")
        return []

    logger.info(f"MDST yielded {len(mdst_edges)} tree edges. Reconstructing in DB...")
    
    # Reconstruct paths in DB to fetch node IDs and relationships
    reconstruct_query = """
    UNWIND $tree_edges AS edge
    MATCH (c)-[r]-(p)
    WHERE id(c) = edge.child AND id(p) = edge.parent
      AND id(c) IN $internal_allowed_ids AND id(p) IN $internal_allowed_ids
    
    // Pick the best relationship if there are parallel ones (optional, but good practice)
    WITH edge, c, p, collect(r) AS rels
    WITH edge, c, p, rels[0] AS best_rel
    
    RETURN edge.weight AS totalCost,
           [
             {internal_id: id(c), id: elementId(c), labels: labels(c), name: c.name},
             {internal_id: id(p), id: elementId(p), labels: labels(p), name: p.name}
           ] AS nodes,
           [
             [
               {type: type(best_rel), rel_id: elementId(best_rel), evidence: coalesce(best_rel.evidence, '')}
             ]
           ] AS relationships
    """
    
    final_paths = []
    
    # Batch the execution to avoid blowing up memory if tree is large
    batch_size = 50
    for i in range(0, len(mdst_edges), batch_size):
        batch = mdst_edges[i:i+batch_size]
        async with driver.session() as session:
            try:
                res = await session.run(reconstruct_query, parameters={
                    "tree_edges": batch,
                    "internal_allowed_ids": internal_allowed_ids
                })
                async for r in res:
                    final_paths.append({
                        "totalCost": r["totalCost"],
                        "nodes": r["nodes"],
                        "relationships": r["relationships"]
                    })
            except Exception as e:
                logger.error(f"Error reconstructing path batch: {e}")

    t5 = time.time()
    logger.info(f"MDST Step 5 (Reconstruct) took {t5 - t4:.3f}s")
    logger.info(f"Successfully reconstructed {len(final_paths)} paths (tree edges).")
    return final_paths


def _greedy_pcst_edges(
    node_list: list[int],
    edge_triples: list[tuple[int, int, float]],
    terminal_indices: set[int],
    edge_cost: float = 1.0,
) -> list[tuple[int, int]]:
    """Greedy PCST fallback when pcst_fast is unavailable."""
    index_of = {n: i for i, n in enumerate(node_list)}
    parent = list(range(len(node_list)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    terminal_idx = {index_of[t] for t in terminal_indices if t in index_of}
    scored_edges: list[tuple[float, int, int]] = []
    for u, v, prize in edge_triples:
        if u not in index_of or v not in index_of:
            continue
        ui, vi = index_of[u], index_of[v]
        scored_edges.append((prize - edge_cost, ui, vi))
    scored_edges.sort(reverse=True)

    selected: list[tuple[int, int]] = []
    for _, ui, vi in scored_edges:
        if find(ui) != find(vi):
            union(ui, vi)
            selected.append((node_list[ui], node_list[vi]))
        elif len({find(i) for i in terminal_idx}) > 1:
            # still disconnected terminals — allow high-prize edge
            selected.append((node_list[ui], node_list[vi]))
            union(ui, vi)

    # Ensure terminal connectivity: add cheapest connecting edges if needed
    if terminal_idx:
        components = {find(i) for i in terminal_idx}
        if len(components) > 1:
            for _, ui, vi in reversed(scored_edges):
                if find(ui) != find(vi):
                    selected.append((node_list[ui], node_list[vi]))
                    union(ui, vi)
                    if len({find(i) for i in terminal_idx}) == 1:
                        break

    return selected


def _pcst_selected_edges(
    node_list: list[int],
    edge_triples: list[tuple[int, int, float]],
    node_prizes: dict[int, float],
    terminal_ids: set[int],
    edge_cost: float = 1.0,
) -> list[tuple[int, int]]:
    try:
        import numpy as np
        import pcst_fast  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("pcst_fast not installed; using greedy PCST fallback.")
        return _greedy_pcst_edges(node_list, edge_triples, terminal_ids, edge_cost)

    index_of = {n: i for i, n in enumerate(node_list)}
    prizes = np.array([node_prizes.get(n, 0.0) for n in node_list], dtype=np.float64)

    edges_arr = []
    costs_arr = []
    for u, v, hop_prize in edge_triples:
        if u not in index_of or v not in index_of:
            continue
        edges_arr.append([index_of[u], index_of[v]])
        costs_arr.append(max(edge_cost - hop_prize * 0.1, 0.01))

    if not edges_arr:
        return []

    edges_np = np.array(edges_arr, dtype=np.int64)
    costs_np = np.array(costs_arr, dtype=np.float64)

    root = -1
    if terminal_ids:
        best_terminal = max(terminal_ids, key=lambda t: node_prizes.get(t, 0.0))
        root = index_of.get(best_terminal, -1)

    _, sel_edge_idx = pcst_fast.pcst_fast(
        edges_np,
        prizes,
        costs_np,
        root,
        1,
        "gw",
        0,
    )
    selected: list[tuple[int, int]] = []
    for ei in sel_edge_idx:
        u_idx, v_idx = edges_np[ei]
        selected.append((node_list[int(u_idx)], node_list[int(v_idx)]))
    return selected


def _decompose_tree_to_paths(
    tree_edges: list[tuple[int, int]],
    edge_prizes: dict[frozenset[int], float],
    root: int,
    budget: int,
    max_path_len: int,
) -> list[tuple[list[int], float]]:
    """Extract root-to-leaf paths from a PCST tree, ranked by hop prize sum."""
    adj: dict[int, list[int]] = {}
    for u, v in tree_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)

    def hop_prize(a: int, b: int) -> float:
        return edge_prizes.get(frozenset({a, b}), 0.0)

    paths: list[tuple[list[int], float]] = []

    def dfs(node: int, parent: int, path: list[int], acc: float) -> None:
        if len(path) - 1 >= max_path_len:
            paths.append((list(path), acc))
            return
        children = [n for n in adj.get(node, []) if n != parent]
        if not children:
            if len(path) >= 2:
                paths.append((list(path), acc))
            return
        for child in children:
            p = hop_prize(node, child)
            path.append(child)
            dfs(child, node, path, acc + p)
            path.pop()

    if root in adj:
        dfs(root, -1, [root], 0.0)

    # dedupe by node sequence
    seen: set[tuple[int, ...]] = set()
    unique: list[tuple[list[int], float]] = []
    for nodes, ps in sorted(paths, key=lambda x: x[1], reverse=True):
        key = tuple(nodes)
        if key not in seen and len(nodes) >= 2:
            seen.add(key)
            unique.append((nodes, ps))

    return unique[:budget]


async def find_pcst_paths(
    driver: AsyncDriver,
    graph_name: str,
    allowed_node_ids: list[str],
    anchor_ids: list[str],
    subquery_vectors: dict[str, list[float]],
    anchor_scores: dict | None = None,
    ppr_scores: dict | None = None,
    budget: int = 50,
    max_path_len: int = 5,
    edge_cost: float = 1.0,
) -> list[dict]:
    """
    Stage 4 baseline: Prize-Collecting Steiner Tree (PCST) + tree path decomposition.
    Unlike min-Steiner, PCST can include high-prize branches when prize > cost.
    """
    _ = subquery_vectors, anchor_scores
    if not graph_name or not anchor_ids:
        return []

    logger.info("--- UNIFIED PIPELINE: PATH SEARCH (PCST BASELINE) ---")
    t0 = time.time()

    internal_ids_map: dict[str, int] = {}
    async with driver.session() as session:
        res = await session.run(
            "MATCH (n) WHERE elementId(n) IN $ids RETURN elementId(n) AS eid, id(n) AS internal_id",
            parameters={"ids": allowed_node_ids},
        )
        async for r in res:
            internal_ids_map[r["eid"]] = r["internal_id"]

    internal_anchors = [internal_ids_map[a] for a in anchor_ids if a in internal_ids_map]
    if len(internal_anchors) < 1:
        await drop_projection(driver, graph_name)
        return []

    ppr_by_internal = {
        internal_ids_map[eid]: score for eid, score in (ppr_scores or {}).items() if eid in internal_ids_map
    }

    rel_stream = """
    CALL gds.graph.relationshipProperty.stream($graph_name, 'affinity_weight')
    YIELD sourceNodeId, targetNodeId, propertyValue AS affinity
    RETURN sourceNodeId, targetNodeId, affinity
    """
    ppr_stream = """
    CALL gds.graph.nodeProperty.stream($graph_name, 'pprScore')
    YIELD nodeId, propertyValue
    RETURN nodeId, propertyValue AS score
    """
    directed: list[tuple[int, int, float]] = []
    async with driver.session() as session:
        res = await session.run(ppr_stream, parameters={"graph_name": graph_name})
        async for r in res:
            ppr_by_internal[r["nodeId"]] = r["score"]
        res = await session.run(rel_stream, parameters={"graph_name": graph_name})
        async for r in res:
            directed.append((r["sourceNodeId"], r["targetNodeId"], r["affinity"]))

    if not directed:
        await drop_projection(driver, graph_name)
        return []

    edge_prizes: dict[frozenset[int], float] = {}
    edge_triples: list[tuple[int, int, float]] = []
    nodes: set[int] = set()
    for u, v, aff in directed:
        nodes.add(u)
        nodes.add(v)
        pu = max(ppr_by_internal.get(u, 0.0), 1e-9)
        pv = max(ppr_by_internal.get(v, 0.0), 1e-9)
        hop_prize = max(aff, 0.0) * math.sqrt(pu * pv)
        hk = frozenset({u, v})
        if hop_prize > edge_prizes.get(hk, 0.0):
            edge_prizes[hk] = hop_prize
        edge_triples.append((u, v, hop_prize))

    node_list = sorted(nodes)
    node_prizes = {n: ppr_by_internal.get(n, 0.0) for n in node_list}

    tree_edges = _pcst_selected_edges(
        node_list,
        edge_triples,
        node_prizes,
        set(internal_anchors),
        edge_cost=edge_cost,
    )
    t1 = time.time()
    logger.info(f"PCST selected {len(tree_edges)} tree edges in {t1 - t0:.3f}s")

    root = max(internal_anchors, key=lambda n: ppr_by_internal.get(n, 0.0))
    path_specs = _decompose_tree_to_paths(tree_edges, edge_prizes, root, budget, max_path_len)

    await drop_projection(driver, graph_name)

    if not path_specs:
        return []

    from server.algorithm.v4.stage4_coverage import _reconstruct_paths

    raw_paths = [
        {"path_index": i, "totalCost": -ps, "nodeIds": nodes, "prize_sum": ps}
        for i, (nodes, ps) in enumerate(path_specs)
    ]
    final_paths = await _reconstruct_paths(driver, raw_paths)
    logger.info(f"PCST reconstructed {len(final_paths)} paths.")
    return final_paths

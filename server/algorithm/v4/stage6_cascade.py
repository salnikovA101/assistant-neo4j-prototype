import logging

logger = logging.getLogger(__name__)

def dense_rank(scores: list[float], reverse: bool = False) -> list[int]:
    """
    Computes dense ranks for a list of scores.
    If reverse=True, higher scores get better (lower) rank (e.g. semantic score).
    If reverse=False, lower scores get better (lower) rank (e.g. topological cost).
    """
    sorted_unique_scores = sorted(list(set(scores)), reverse=reverse)
    rank_dict = {score: rank + 1 for rank, score in enumerate(sorted_unique_scores)}
    return [rank_dict[score] for score in scores]

def extract_edge_tuples(path: dict) -> set[tuple]:
    """
    Extracts a set of hop signatures from a path for overlap calculation.
    Since we aggregate all parallel edges between nodes, a hop is uniquely defined by its nodes.
    Signature: (node1_id, node2_id) sorted to treat A->B and B->A consistently.
    """
    edges = set()
    nodes = path.get("nodes", [])
    for i in range(len(nodes) - 1):
        n1 = nodes[i].get("internal_id")
        n2 = nodes[i+1].get("internal_id")
        
        if n1 is not None and n2 is not None:
            hop = tuple(sorted([str(n1), str(n2)]))
            edges.add(hop)
            
    return edges

def rank_and_filter_paths(paths: list[dict], k_limit: int = 50, overlap_threshold: float = 0.75) -> list[dict]:
    """
    Stage 6: Hybrid Cascade & Greedy Set Cover.
    Ranks paths using RRF (GDS Total Cost + Reranker Score).
    Deduplicates using Greedy Set Cover (overlap > 0.75 drops the path).
    Returns Top-K paths.
    """
    if not paths:
        return []

    # 1. Extract scores
    gds_costs = [p.get("totalCost", float('inf')) for p in paths]
    semantic_scores = [p.get("semantic_score", 0.0) for p in paths]
    
    # 2. Dense Ranking
    rank_gds = dense_rank(gds_costs, reverse=False)       # Lower cost is better
    rank_semantic = dense_rank(semantic_scores, reverse=True) # Higher score is better
    
    # 3. Calculate RRF Score
    # We use a standard RRF formula: 1 / (k + rank)
    # k is typically set to 60 in IR literature
    k_rrf = 60
    
    for i, p in enumerate(paths):
        p["rank_topo"] = rank_gds[i]
        p["rank_semantic"] = rank_semantic[i]
        
        r_topo = p.get("rank_topo", len(paths))
        r_sem = p.get("rank_semantic", len(paths))
        
        # Hybrid formula: RRF of Topology Rank + RRF of Semantic Rank
        p["rrf_score"] = (1.0 / (k_rrf + r_topo)) + (1.0 / (k_rrf + r_sem))  
        
    # 4. Sort by Score descending
    paths.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    # 5. Greedy Set Cover Deduplication
    selected_paths = []
    covered_edges = set()
    
    for p in paths:
        path_edges = extract_edge_tuples(p)
        if not path_edges:
            continue
            
        # Calculate overlap with ALL currently selected edges
        intersection = path_edges.intersection(covered_edges)
        overlap_ratio = len(intersection) / len(path_edges)
        
        if overlap_ratio <= overlap_threshold:
            p["overlap_ratio"] = overlap_ratio
            selected_paths.append(p)
            covered_edges.update(path_edges)
            
            if len(selected_paths) >= k_limit:
                break
        else:
            logger.debug(f"Dropped path due to overlap {overlap_ratio:.2f} > {overlap_threshold}")
            
    logger.info(f"Stage 6 completed. Reduced {len(paths)} paths to {len(selected_paths)} unique paths.")
    return selected_paths

"""
Stage 4: Prize-Coverage Path Harvesting (PCPH).

Harvests connected paths by greedily covering high-prize hops (node pairs),
without anchor-pair shortest paths or Steiner trees.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from neo4j import AsyncDriver

from server.algorithm.v4.stage3_projection import drop_projection

logger = logging.getLogger(__name__)

EPS = 1e-9


def hop_key(u: int, v: int) -> frozenset[int]:
    return frozenset({u, v})


def directed_hop(u: int, v: int) -> tuple[int, int]:
    return (u, v)


def path_hops(node_ids: list[int]) -> frozenset[frozenset[int]]:
    return frozenset(hop_key(node_ids[i], node_ids[i + 1]) for i in range(len(node_ids) - 1))


def path_directed_hops(node_ids: list[int]) -> frozenset[tuple[int, int]]:
    return frozenset(directed_hop(node_ids[i], node_ids[i + 1]) for i in range(len(node_ids) - 1))


@dataclass
class PathCandidate:
    node_ids: list[int]
    hops: frozenset[frozenset[int]] = field(default_factory=frozenset)
    directed_hops: frozenset[tuple[int, int]] = field(default_factory=frozenset)
    prize_sum: float = 0.0

    def __post_init__(self) -> None:
        if len(self.node_ids) >= 2:
            if not self.hops:
                self.hops = path_hops(self.node_ids)
            if not self.directed_hops:
                self.directed_hops = path_directed_hops(self.node_ids)


@dataclass
class PrizeGraph:
    """Hop graph with directed edge prizes and undirected adjacency for growth."""

    prize: dict[frozenset[int], float]
    directed_prize: dict[tuple[int, int], float]
    adj: dict[int, list[tuple[int, float]]]
    ppr: dict[int, float]

    def get_prize(self, u: int, v: int) -> float:
        d = self.directed_prize.get(directed_hop(u, v), 0.0)
        if d <= 0:
            d = self.directed_prize.get(directed_hop(v, u), 0.0)
        if d <= 0:
            d = self.prize.get(hop_key(u, v), 0.0)
        return d

    def sorted_hops(self) -> list[tuple[int, int, float]]:
        seen: set[frozenset[int]] = set()
        hops: list[tuple[int, int, float]] = []
        for hk, p in self.prize.items():
            if hk in seen or len(hk) != 2:
                continue
            seen.add(hk)
            nodes = list(hk)
            hops.append((nodes[0], nodes[1], p))
        hops.sort(key=lambda x: x[2], reverse=True)
        return hops


def build_prize_graph(
    directed_edges: list[tuple[int, int, float]],
    ppr_by_internal: dict[int, float] | None = None,
) -> PrizeGraph:
    """Build prize graph; directed edges keep separate scores, adj uses max on link."""
    prize: dict[frozenset[int], float] = {}
    directed_prize: dict[tuple[int, int], float] = {}
    adj: dict[int, list[tuple[int, float]]] = {}

    for u, v, affinity in directed_edges:
        if u == v:
            continue
        hop_prize = max(affinity, EPS)
        hk = hop_key(u, v)
        if hop_prize > prize.get(hk, 0.0):
            prize[hk] = hop_prize
        prev = directed_prize.get(directed_hop(u, v), 0.0)
        if hop_prize > prev:
            directed_prize[directed_hop(u, v)] = hop_prize

    for hk, hop_prize in prize.items():
        if len(hk) != 2:
            continue
        nodes = list(hk)
        a, b = nodes[0], nodes[1]
        adj.setdefault(a, []).append((b, hop_prize))
        adj.setdefault(b, []).append((a, hop_prize))

    for node in adj:
        adj[node].sort(key=lambda x: x[1], reverse=True)

    return PrizeGraph(
        prize=prize,
        directed_prize=directed_prize,
        adj=adj,
        ppr=dict(ppr_by_internal or {}),
    )


def build_combined_multi_subquery_graph(
    channel_graphs: list[PrizeGraph],
) -> PrizeGraph:
    """Combine per-subquery graphs with min-prize across channels (all intents must match)."""
    if not channel_graphs:
        return PrizeGraph(prize={}, directed_prize={}, adj={}, ppr={})
    if len(channel_graphs) == 1:
        return channel_graphs[0]

    all_directed: set[tuple[int, int]] = set()
    for g in channel_graphs:
        all_directed.update(g.directed_prize.keys())

    directed_prize: dict[tuple[int, int], float] = {}
    prize: dict[frozenset[int], float] = {}
    for edge in all_directed:
        scores = [g.directed_prize.get(edge, 0.0) for g in channel_graphs]
        # Edge must be present with positive prize in every channel.
        if any(s <= 0.0 for s in scores):
            continue
        p = min(scores)
        directed_prize[edge] = p
        u, v = edge
        hk = hop_key(u, v)
        if p > prize.get(hk, 0.0):
            prize[hk] = p

    adj: dict[int, list[tuple[int, float]]] = {}
    for hk, hop_prize in prize.items():
        if len(hk) != 2:
            continue
        nodes = list(hk)
        a, b = nodes[0], nodes[1]
        adj.setdefault(a, []).append((b, hop_prize))
        adj.setdefault(b, []).append((a, hop_prize))
    for node in adj:
        adj[node].sort(key=lambda x: x[1], reverse=True)

    merged_ppr: dict[int, float] = {}
    for g in channel_graphs:
        for nid, score in g.ppr.items():
            if score > merged_ppr.get(nid, 0.0):
                merged_ppr[nid] = score

    return PrizeGraph(
        prize=prize, directed_prize=directed_prize, adj=adj, ppr=merged_ppr
    )


def growth_score(
    graph: PrizeGraph,
    x: int,
    y: int,
    covered: set[frozenset[int]],
    uncovered_boost: float,
    ppr_bias_pow: float,
    anchor_nodes: set[int] | None = None,
    anchor_boost: float = 1.15,
) -> float:
    hk = hop_key(x, y)
    boost = uncovered_boost if hk not in covered else 1.0
    score = graph.get_prize(x, y) * boost
    if ppr_bias_pow != 0.0 and graph.ppr:
        score *= max(graph.ppr.get(y, 0.0), EPS) ** ppr_bias_pow
    if anchor_nodes and anchor_boost > 1.0 and (x in anchor_nodes or y in anchor_nodes):
        score *= anchor_boost
    return score


def path_prize_sum(graph: PrizeGraph, node_ids: list[int]) -> float:
    total = 0.0
    for i in range(len(node_ids) - 1):
        total += graph.get_prize(node_ids[i], node_ids[i + 1])
    return total


def top_neighbors(
    graph: PrizeGraph,
    x: int,
    exclude: set[int],
    covered: set[frozenset[int]],
    uncovered_boost: float,
    ppr_bias_pow: float,
    k: int,
    anchor_nodes: set[int] | None = None,
) -> list[tuple[int, float]]:
    scored: list[tuple[int, float]] = []
    for neighbor, _ in graph.adj.get(x, []):
        if neighbor in exclude:
            continue
        score = growth_score(
            graph, x, neighbor, covered, uncovered_boost, ppr_bias_pow, anchor_nodes
        )
        if score > 0:
            scored.append((neighbor, score))
    scored.sort(key=lambda t: t[1], reverse=True)
    return scored[:k]


def best_neighbor(
    graph: PrizeGraph,
    x: int,
    exclude: set[int],
    covered: set[frozenset[int]],
    uncovered_boost: float,
    ppr_bias_pow: float,
    anchor_nodes: set[int] | None = None,
) -> tuple[int, float] | None:
    top = top_neighbors(
        graph, x, exclude, covered, uncovered_boost, ppr_bias_pow, 1, anchor_nodes
    )
    return top[0] if top else None


def grow_path(
    graph: PrizeGraph,
    a: int,
    b: int,
    covered: set[frozenset[int]],
    max_path_len: int,
    uncovered_boost: float,
    ppr_bias_pow: float,
    beam_width: int = 1,
    anchor_nodes: set[int] | None = None,
) -> list[int]:
    """Bidirectional growth from seed hop; uses beam search when beam_width > 1."""
    if beam_width <= 1:
        return _grow_path_greedy(
            graph, a, b, covered, max_path_len, uncovered_boost, ppr_bias_pow, anchor_nodes
        )
    paths = grow_path_beam(
        graph, a, b, covered, max_path_len, uncovered_boost, ppr_bias_pow, beam_width, anchor_nodes
    )
    return paths[0] if paths else [a, b]


def _grow_path_greedy(
    graph: PrizeGraph,
    a: int,
    b: int,
    covered: set[frozenset[int]],
    max_path_len: int,
    uncovered_boost: float,
    ppr_bias_pow: float,
    anchor_nodes: set[int] | None = None,
) -> list[int]:
    seq = [a, b]
    max_hops = max(1, max_path_len - 1)

    while len(seq) - 1 < max_hops:
        exclude = set(seq)
        left = best_neighbor(
            graph, seq[0], exclude, covered, uncovered_boost, ppr_bias_pow, anchor_nodes
        )
        right = best_neighbor(
            graph, seq[-1], exclude, covered, uncovered_boost, ppr_bias_pow, anchor_nodes
        )

        if left is None and right is None:
            break

        if left is not None and (right is None or left[1] >= right[1]):
            seq.insert(0, left[0])
        elif right is not None:
            seq.append(right[0])
        else:
            break

    return seq


def grow_path_beam(
    graph: PrizeGraph,
    a: int,
    b: int,
    covered: set[frozenset[int]],
    max_path_len: int,
    uncovered_boost: float,
    ppr_bias_pow: float,
    beam_width: int,
    anchor_nodes: set[int] | None = None,
) -> list[list[int]]:
    """Beam search bidirectional growth; returns up to beam_width distinct paths."""
    max_hops = max(1, max_path_len - 1)
    beams: list[list[int]] = [[a, b]]

    for _ in range(max_hops - 1):
        candidates: list[tuple[list[int], float]] = []
        any_extended = False

        for seq in beams:
            if len(seq) - 1 >= max_hops:
                candidates.append((seq, path_prize_sum(graph, seq)))
                continue

            exclude = set(seq)
            left_opts = top_neighbors(
                graph, seq[0], exclude, covered, uncovered_boost, ppr_bias_pow, beam_width, anchor_nodes
            )
            right_opts = top_neighbors(
                graph, seq[-1], exclude, covered, uncovered_boost, ppr_bias_pow, beam_width, anchor_nodes
            )

            expanded = False
            for n, _ in left_opts:
                new_seq = [n] + seq
                candidates.append((new_seq, path_prize_sum(graph, new_seq)))
                expanded = True
            for n, _ in right_opts:
                new_seq = seq + [n]
                candidates.append((new_seq, path_prize_sum(graph, new_seq)))
                expanded = True

            if not expanded:
                candidates.append((seq, path_prize_sum(graph, seq)))

            any_extended = any_extended or expanded

        if not candidates:
            break

        candidates.sort(key=lambda x: x[1], reverse=True)
        next_beams: list[list[int]] = []
        seen_keys: set[tuple[int, ...]] = set()
        for seq, _ in candidates:
            key = tuple(seq)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            next_beams.append(seq)
            if len(next_beams) >= beam_width:
                break

        if not next_beams:
            break
        beams = next_beams

        if not any_extended:
            break

    if not beams:
        return [[a, b]]
    beams.sort(key=lambda s: path_prize_sum(graph, s), reverse=True)
    return beams[:beam_width]


def _make_candidate(graph: PrizeGraph, node_ids: list[int]) -> PathCandidate:
    return PathCandidate(
        node_ids=node_ids,
        prize_sum=path_prize_sum(graph, node_ids),
    )


def _harvest_from_hops(
    graph: PrizeGraph,
    seed_hops: list[tuple[int, int, float]],
    candidate_cap: int,
    max_path_len: int,
    uncovered_boost: float,
    ppr_bias_pow: float,
    beam_width: int,
    growth_covered: set[frozenset[int]],
    seen_paths: set[tuple[int, ...]],
    candidates: list[PathCandidate],
    paths_per_seed: int = 1,
    anchor_nodes: set[int] | None = None,
) -> None:
    used_seeds: set[frozenset[int]] = set()

    for a, b, _ in seed_hops:
        if len(candidates) >= candidate_cap:
            break
        hk = hop_key(a, b)
        # Diversify: only seed from hops not already covered by earlier paths.
        if hk in used_seeds or hk in growth_covered:
            continue
        used_seeds.add(hk)

        if beam_width > 1 and paths_per_seed > 1:
            path_list = grow_path_beam(
                graph, a, b, growth_covered, max_path_len, uncovered_boost, ppr_bias_pow, beam_width, anchor_nodes
            )
        else:
            path_list = [
                grow_path(
                    graph, a, b, growth_covered, max_path_len, uncovered_boost, ppr_bias_pow, beam_width, anchor_nodes
                )
            ]

        for node_ids in path_list[:paths_per_seed]:
            path_key = tuple(node_ids)
            if path_key in seen_paths or len(node_ids) < 2:
                continue
            seen_paths.add(path_key)
            cand = _make_candidate(graph, node_ids)
            candidates.append(cand)
            growth_covered |= cand.hops


def harvest_candidates(
    graph: PrizeGraph,
    candidate_cap: int,
    max_path_len: int,
    uncovered_boost: float,
    ppr_bias_pow: float,
    beam_width: int = 1,
    long_tail_harvest: bool = False,
    paths_per_seed: int = 1,
    anchor_nodes: set[int] | None = None,
) -> list[PathCandidate]:
    """Generate path candidates from high-prize hops, optionally with long-tail pass."""
    growth_covered: set[frozenset[int]] = set()
    candidates: list[PathCandidate] = []
    seen_paths: set[tuple[int, ...]] = set()

    all_hops = graph.sorted_hops()
    primary_cap = candidate_cap if not long_tail_harvest else int(candidate_cap * 0.75)

    _harvest_from_hops(
        graph,
        all_hops,
        primary_cap,
        max_path_len,
        uncovered_boost,
        ppr_bias_pow,
        beam_width,
        growth_covered,
        seen_paths,
        candidates,
        paths_per_seed=paths_per_seed,
        anchor_nodes=anchor_nodes,
    )

    if long_tail_harvest and len(candidates) < candidate_cap and len(all_hops) >= 4:
        n = len(all_hops)
        tail_start = n // 2
        tail_end = min(n, int(n * 0.95))
        tail_hops = all_hops[tail_start:tail_end]
        _harvest_from_hops(
            graph,
            tail_hops,
            candidate_cap,
            max_path_len,
            uncovered_boost,
            ppr_bias_pow,
            max(1, beam_width - 1),
            growth_covered,
            seen_paths,
            candidates,
            paths_per_seed=1,
            anchor_nodes=anchor_nodes,
        )

    return candidates


def mmr_select(
    candidates: list[PathCandidate],
    graph: PrizeGraph,
    budget: int,
    mmr_lambda: float,
) -> list[PathCandidate]:
    """Greedy max-coverage with MMR overlap penalty; fill by marginal uncovered hops."""
    remaining = list(candidates)
    selected: list[PathCandidate] = []
    covered: set[frozenset[int]] = set()

    def marginal_gain(cand: PathCandidate) -> tuple[float, float, int]:
        new_prize = 0.0
        overlap_prize = 0.0
        new_hops = 0
        for h in cand.hops:
            p = graph.prize.get(h, 0.0)
            if h in covered:
                overlap_prize += p
            else:
                new_prize += p
                new_hops += 1
        gain = new_prize - mmr_lambda * overlap_prize
        return gain, new_prize, new_hops

    while remaining and len(selected) < budget:
        best_idx = -1
        best_gain = -float("inf")
        best_new = -float("inf")
        best_hops = -1

        for i, cand in enumerate(remaining):
            gain, new_prize, new_hops = marginal_gain(cand)
            if (
                gain > best_gain
                or (gain == best_gain and new_hops > best_hops)
                or (gain == best_gain and new_hops == best_hops and new_prize > best_new)
            ):
                best_gain = gain
                best_new = new_prize
                best_hops = new_hops
                best_idx = i

        if best_idx < 0:
            break

        gain, _, _ = marginal_gain(remaining[best_idx])
        if gain <= 0:
            break

        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        covered |= chosen.hops

    if remaining and len(selected) < budget:
        def fill_key(cand: PathCandidate) -> tuple[int, float, float]:
            new_hops = sum(1 for h in cand.hops if h not in covered)
            new_prize = sum(graph.prize.get(h, 0.0) for h in cand.hops if h not in covered)
            return (new_hops, new_prize, cand.prize_sum)

        remaining.sort(key=fill_key, reverse=True)
        for cand in remaining:
            if len(selected) >= budget:
                break
            path_key = tuple(cand.node_ids)
            if any(tuple(s.node_ids) == path_key for s in selected):
                continue
            new_hops = sum(1 for h in cand.hops if h not in covered)
            if new_hops == 0 and len(selected) >= budget // 2:
                continue
            selected.append(cand)
            covered |= cand.hops

    return selected


def reorder_paths_by_density(paths: list[PathCandidate]) -> list[PathCandidate]:
    """Rank paths by semantic prize density (helps MRR@5)."""
    def density(c: PathCandidate) -> float:
        n_hops = max(1, len(c.node_ids) - 1)
        return c.prize_sum / n_hops

    return sorted(paths, key=density, reverse=True)


RECONSTRUCT_QUERY = """
UNWIND $paths AS path_obj
WITH path_obj, path_obj.nodeIds AS nodeIds

MATCH (n) WHERE id(n) IN nodeIds
WITH path_obj, nodeIds, collect({internal_id: id(n), id: elementId(n), labels: labels(n), name: n.name}) AS unordered_nodes
WITH path_obj, nodeIds, [nid IN nodeIds |
    [x IN unordered_nodes WHERE x.internal_id = nid][0]
] AS ordered_nodes

UNWIND range(0, size(nodeIds)-2) AS i
MATCH (a)-[r]-(b)
WHERE id(a) = nodeIds[i] AND id(b) = nodeIds[i+1]

WITH path_obj, ordered_nodes, i, collect(r) AS hop_rels
ORDER BY path_obj.path_index, i

WITH path_obj, ordered_nodes, collect([rel IN hop_rels | {
    type: type(rel),
    rel_id: elementId(rel),
    evidence: coalesce(rel.evidence, '')
}]) AS ordered_rels

RETURN path_obj.totalCost AS totalCost, ordered_nodes AS nodes, ordered_rels AS relationships
"""


async def _stream_subquery_edge_sims(
    driver: AsyncDriver,
    allowed_ids: list[str],
    query_vectors: list[list[float]],
) -> list[tuple[int, int, list[float]]]:
    """Per directed edge, cosine sim for each subquery vector."""
    if not query_vectors or not allowed_ids:
        return []

    query = """
    MATCH (s)-[r]-(t)
    WHERE elementId(s) IN $allowed AND elementId(t) IN $allowed
      AND r.evidence_embedding IS NOT NULL AND size(r.evidence_embedding) > 0
    WITH id(s) AS src, id(t) AS tgt,
         [q IN $query_vectors |
            vector.similarity.cosine(q, r.evidence_embedding)
         ] AS sims
    RETURN src, tgt, sims
    """
    edges: list[tuple[int, int, list[float]]] = []
    async with driver.session() as session:
        res = await session.run(
            query,
            parameters={"allowed": allowed_ids, "query_vectors": query_vectors},
        )
        async for r in res:
            sims = [max(0.0, s) for s in r["sims"]]
            edges.append((r["src"], r["tgt"], sims))
    return edges


def _graphs_from_subquery_sims(
    edge_sims: list[tuple[int, int, list[float]]],
    ppr_by_internal: dict[int, float] | None = None,
) -> list[PrizeGraph]:
    if not edge_sims:
        return []
    n_channels = len(edge_sims[0][2])
    channel_edges: list[list[tuple[int, int, float]]] = [[] for _ in range(n_channels)]

    for src, tgt, sims in edge_sims:
        for k, sim in enumerate(sims):
            if sim > 0:
                channel_edges[k].append((src, tgt, max(sim, EPS)))
                channel_edges[k].append((tgt, src, max(sim, EPS)))

    return [
        build_prize_graph(edges, ppr_by_internal)
        for edges in channel_edges
        if edges
    ]


async def _reconstruct_paths(driver: AsyncDriver, raw_paths: list[dict]) -> list[dict]:
    final_paths: list[dict] = []
    batch_size = 100
    for i in range(0, len(raw_paths), batch_size):
        batch = raw_paths[i : i + batch_size]
        async with driver.session() as session:
            try:
                res = await session.run(RECONSTRUCT_QUERY, parameters={"paths": batch})
                async for r in res:
                    final_paths.append(
                        {
                            "totalCost": r["totalCost"],
                            "nodes": r["nodes"],
                            "relationships": r["relationships"],
                            "prize_sum": -r["totalCost"],
                        }
                    )
            except Exception as e:
                logger.error(f"Error reconstructing path batch: {e}")
    return final_paths


async def find_coverage_paths(
    driver: AsyncDriver,
    graph_name: str,
    allowed_ids: list[str],
    anchor_ids: list[str] | None = None,
    s2_scores: dict[str, float] | None = None,
    ppr_scores: dict[str, float] | None = None,
    subquery_vectors: dict[str, list[float]] | None = None,
    max_path_len: int = 5,
    candidate_cap: int = 200,
    budget: int = 50,
    uncovered_boost: float = 2.0,
    ppr_bias_pow: float = 0.5,
    mmr_lambda: float = 0.3,
    beam_width: int = 1,
    long_tail_harvest: bool = False,
    multi_subquery: bool = True,
) -> list[dict]:
    """
    Stage 4 PCPH: prize-coverage path harvesting over the Stage 3 filtered subgraph.
    Prize = semantic affinity (directed edges); optional per-subquery harvest.
    """
    _ = s2_scores

    if not graph_name:
        logger.error("No graph name provided to Stage 4 (PCPH).")
        return []

    logger.info("--- UNIFIED PIPELINE: PATH SEARCH (PRIZE-COVERAGE PATH HARVESTING) ---")
    t0 = time.time()

    anchor_internal: set[int] = set()
    if anchor_ids:
        async with driver.session() as session:
            res = await session.run(
                "MATCH (n) WHERE elementId(n) IN $ids RETURN id(n) AS internal_id",
                parameters={"ids": anchor_ids},
            )
            async for r in res:
                anchor_internal.add(r["internal_id"])

    rel_stream_query = """
    CALL gds.graph.relationshipProperty.stream($graph_name, 'affinity_weight')
    YIELD sourceNodeId, targetNodeId, propertyValue AS affinity
    RETURN sourceNodeId, targetNodeId, affinity
    """
    ppr_stream_query = """
    CALL gds.graph.nodeProperty.stream($graph_name, 'pprScore')
    YIELD nodeId, propertyValue
    RETURN nodeId, propertyValue AS score
    """
    directed_edges: list[tuple[int, int, float]] = []
    ppr_by_internal: dict[int, float] = {}
    async with driver.session() as session:
        try:
            res = await session.run(rel_stream_query, parameters={"graph_name": graph_name})
            async for r in res:
                directed_edges.append((r["sourceNodeId"], r["targetNodeId"], r["affinity"]))
            try:
                res = await session.run(ppr_stream_query, parameters={"graph_name": graph_name})
                async for r in res:
                    ppr_by_internal[r["nodeId"]] = float(r["score"])
            except Exception as e:
                logger.warning(f"Could not stream pprScore for PCPH bias: {e}")
        except Exception as e:
            logger.error(f"Failed to stream affinity_weight: {e}")
            await drop_projection(driver, graph_name)
            return []

    # Fallback: map elementId PPR scores if GDS stream was empty.
    if not ppr_by_internal and ppr_scores and allowed_ids:
        async with driver.session() as session:
            res = await session.run(
                "MATCH (n) WHERE elementId(n) IN $ids "
                "RETURN elementId(n) AS eid, id(n) AS internal_id",
                parameters={"ids": list(allowed_ids)},
            )
            async for r in res:
                score = ppr_scores.get(r["eid"])
                if score is not None:
                    ppr_by_internal[r["internal_id"]] = float(score)

    if not directed_edges:
        logger.warning("No edges in projection for PCPH.")
        await drop_projection(driver, graph_name)
        return []

    base_graph = build_prize_graph(directed_edges, ppr_by_internal)
    t1 = time.time()
    logger.info(
        f"Loaded prize graph in {t1 - t0:.3f}s "
        f"(ppr_nodes={len(ppr_by_internal)}, ppr_bias_pow={ppr_bias_pow})"
    )

    query_vectors = list(subquery_vectors.values()) if subquery_vectors else []
    channel_graphs: list[PrizeGraph] = []
    if multi_subquery and len(query_vectors) > 1:
        edge_sims = await _stream_subquery_edge_sims(driver, allowed_ids, query_vectors)
        channel_graphs = _graphs_from_subquery_sims(edge_sims, ppr_by_internal)
        logger.info(
            f"Multi-subquery: {len(channel_graphs)} channel graphs from {len(edge_sims)} edges"
        )

    all_candidates: list[PathCandidate] = []
    seen_paths: set[tuple[int, ...]] = set()

    def _add_candidates(cands: list[PathCandidate]) -> None:
        for c in cands:
            key = tuple(c.node_ids)
            if key not in seen_paths:
                seen_paths.add(key)
                all_candidates.append(c)

    main_cands = harvest_candidates(
        base_graph,
        candidate_cap=candidate_cap,
        max_path_len=max_path_len,
        uncovered_boost=uncovered_boost,
        ppr_bias_pow=ppr_bias_pow,
        beam_width=beam_width,
        long_tail_harvest=long_tail_harvest,
        paths_per_seed=min(2, beam_width),
        anchor_nodes=anchor_internal or None,
    )
    _add_candidates(main_cands)

    if channel_graphs:
        if len(channel_graphs) > 1:
            combined = build_combined_multi_subquery_graph(channel_graphs)
            if combined.prize:
                combined_cap = max(40, candidate_cap // (len(channel_graphs) + 2))
                _add_candidates(
                    harvest_candidates(
                        combined,
                        candidate_cap=combined_cap,
                        max_path_len=max_path_len,
                        uncovered_boost=uncovered_boost,
                        ppr_bias_pow=ppr_bias_pow,
                        beam_width=beam_width,
                        long_tail_harvest=False,
                        paths_per_seed=min(2, beam_width),
                        anchor_nodes=anchor_internal or None,
                    )
                )

        per_channel_cap = max(50, candidate_cap // (len(channel_graphs) + 1))
        for ch_graph in channel_graphs:
            _add_candidates(
                harvest_candidates(
                    ch_graph,
                    candidate_cap=per_channel_cap,
                    max_path_len=max_path_len,
                    uncovered_boost=uncovered_boost,
                    ppr_bias_pow=ppr_bias_pow,
                    beam_width=beam_width,
                    long_tail_harvest=False,
                    paths_per_seed=min(2, beam_width),
                    anchor_nodes=anchor_internal or None,
                )
            )

    if not all_candidates:
        all_candidates = main_cands

    t2 = time.time()
    logger.info(
        f"Prize graph: {len(base_graph.prize)} hops, {len(base_graph.adj)} nodes; "
        f"harvested {len(all_candidates)} candidates in {t2 - t1:.3f}s"
    )

    selected = mmr_select(all_candidates, base_graph, budget=budget, mmr_lambda=mmr_lambda)
    selected = reorder_paths_by_density(selected)
    t3 = time.time()
    logger.info(f"MMR selected {len(selected)} paths in {t3 - t2:.3f}s")

    await drop_projection(driver, graph_name)

    if not selected:
        return []

    raw_paths = [
        {
            "path_index": idx,
            "totalCost": -c.prize_sum,
            "nodeIds": c.node_ids,
            "prize_sum": c.prize_sum,
        }
        for idx, c in enumerate(selected)
    ]

    final_paths = await _reconstruct_paths(driver, raw_paths)
    t4 = time.time()
    logger.info(f"Reconstructed {len(final_paths)} paths in {t4 - t3:.3f}s")
    return final_paths

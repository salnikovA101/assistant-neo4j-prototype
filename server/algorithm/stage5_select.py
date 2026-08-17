"""S5: dedup S4 pool by spine evidence seq; per-graph k quota; hydrate."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

from neo4j import AsyncDriver

from server.algorithm.cypher.edges import fetch_edge_evidence
from server.algorithm.models import Chain
from server.algorithm.params import Params

logger = logging.getLogger(__name__)


def dedup_s4_pool(pool: list[Chain]) -> list[Chain]:
    """
    One unit per spine_evidence_seq; prefer higher G, then longer spine,
    then sq_* over global. Secondary: identical spine edge_key sets merge graphs.
    """
    best: dict[tuple[str, ...], Chain] = {}
    for c in pool:
        key = c.spine_evidence_seq()
        prev = best.get(key)
        if prev is None:
            best[key] = c
            continue
        graphs = list(
            dict.fromkeys((prev.source_graphs or [prev.source_graph]) + (c.source_graphs or [c.source_graph]))
        )
        prefer_c = False
        if (
            c.score > prev.score
            or abs(c.score - prev.score) < 1e-12
            and len(c.edge_keys) > len(prev.edge_keys)
            or (
                abs(c.score - prev.score) < 1e-12
                and len(c.edge_keys) == len(prev.edge_keys)
                and (prev.source_graph == "global" and c.source_graph != "global")
            )
        ):
            prefer_c = True
        if prefer_c:
            c.source_graphs = graphs
            best[key] = c
        else:
            prev.source_graphs = graphs
    return list(best.values())


def _score_key(c: Chain) -> tuple[float, int]:
    return (c.score, len(c.edge_keys))


def _is_eligible(
    c: Chain,
    *,
    batch_seqs: set[tuple[str, ...]],
) -> bool:
    if not c.edge_keys and not c.fans:
        return False
    seq = c.spine_evidence_seq()
    if seq in batch_seqs:
        return False
    return True


def select_budget_batch(
    unique_pool: list[Chain],
    *,
    k: int,
    graph_ids: Sequence[str] | None = None,
) -> list[Chain]:
    """
    Fill up to k units with even per-graph quota, then leftover by global score.

    After S4 each graph may contribute several tours; the per-graph round
    takes floor(k/n) of them, then fill uses leftover slots.

    For n active graphs: take floor(k/n) best novel units from each graph
    (by score desc). If slots remain (remainder or thin graphs), fill with the
    best remaining units across all graphs.
    """
    if k <= 0:
        return []

    ids = list(graph_ids) if graph_ids else sorted({c.source_graph for c in unique_pool if c.source_graph})
    if not ids:
        ids = ["_"]

    by_graph: dict[str, list[Chain]] = defaultdict(list)
    for c in unique_pool:
        g = c.source_graph or "_"
        by_graph[g].append(c)
    for g in by_graph:
        by_graph[g].sort(key=_score_key, reverse=True)

    n = max(1, len(ids))
    per = k // n
    batch: list[Chain] = []
    batch_seqs: set[tuple[str, ...]] = set()
    picked: set[int] = set()  # id(c)

    def _try_add(c: Chain) -> bool:
        if len(batch) >= k:
            return False
        if id(c) in picked:
            return False
        if not _is_eligible(c, batch_seqs=batch_seqs):
            return False
        batch.append(c)
        batch_seqs.add(c.spine_evidence_seq())
        picked.add(id(c))
        return True

    # Round 1: even quota per active graph
    if per > 0:
        for gid in ids:
            taken = 0
            for c in by_graph.get(gid, []):
                if taken >= per:
                    break
                if _try_add(c):
                    taken += 1

    # Round 2: fill remainder with best across all graphs
    if len(batch) < k:
        leftovers = sorted(unique_pool, key=_score_key, reverse=True)
        for c in leftovers:
            if len(batch) >= k:
                break
            _try_add(c)

    # Score ascending; emit re-ranks by score descending.
    batch.sort(key=_score_key)

    out: list[Chain] = []
    for i, c in enumerate(batch):
        out.append(
            Chain(
                chain_id=f"c{i + 1}",
                edge_keys=list(c.edge_keys),
                score=c.score,
                source_graph=c.source_graph,
                source_graphs=list(c.source_graphs or [c.source_graph]),
                edges=list(c.edges),
                fans={h: list(fl) for h, fl in c.fans.items()},
                fan_hub_names=dict(c.fan_hub_names),
                walk=list(c.walk),
            )
        )
    return out


async def hydrate_chains(driver: AsyncDriver, chains: Sequence[Chain]) -> None:
    need: list[str] = []
    for c in chains:
        for e in c.all_edges():
            if e.element_id and not e.evidence:
                need.append(e.element_id)
    if need:
        rows = await fetch_edge_evidence(driver, need)
        for c in chains:
            for e in c.all_edges():
                row = rows.get(e.element_id)
                if not row:
                    continue
                e.evidence = row.get("evidence") or e.evidence
                e.chunk_id = row.get("chunk_id") or e.chunk_id
                e.source_file = row.get("source_file") or e.source_file
                if row.get("confidence") is not None:
                    e.confidence = float(row.get("confidence") or e.confidence)
    for c in chains:
        c.text = c.format_unit(c.chain_id)


def prepare_s5_batch(
    s4_pool: list[Chain],
    *,
    params: Params,
    graph_ids: Sequence[str] | None = None,
    k: int | None = None,
) -> list[Chain]:
    unique = dedup_s4_pool(s4_pool)
    k_use = int(params.effort_max_paths() if k is None else k)
    batch = select_budget_batch(
        unique,
        k=k_use,
        graph_ids=graph_ids,
    )
    n_g = len(graph_ids) if graph_ids else len({c.source_graph for c in unique})
    logger.info(
        "V6 S5 unique=%s batch=%s k=%s graphs=%s per~%s",
        len(unique),
        len(batch),
        k_use,
        n_g,
        (k_use // n_g) if n_g else k_use,
    )
    return batch

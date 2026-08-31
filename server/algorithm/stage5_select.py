"""S5: dedup S4 pool by spine evidence seq; hydrate."""

from __future__ import annotations

import logging
from collections.abc import Sequence

from neo4j import AsyncDriver

from server.algorithm.cypher.edges import fetch_edge_evidence
from server.algorithm.models import Chain, parse_confidence

logger = logging.getLogger(__name__)


def dedup_s4_pool(pool: list[Chain]) -> list[Chain]:
    """Keep the first unit per spine_evidence_seq (carousel order)."""
    seen: dict[tuple[str, ...], Chain] = {}
    out: list[Chain] = []
    for c in pool:
        key = c.spine_evidence_seq()
        prev = seen.get(key)
        if prev is None:
            seen[key] = c
            out.append(c)
            continue
        graphs = list(
            dict.fromkeys(
                (prev.source_graphs or [prev.source_graph])
                + (c.source_graphs or [c.source_graph])
            )
        )
        prev.source_graphs = graphs
    return out


def _copy_labeled(c: Chain, chain_id: str) -> Chain:
    return Chain(
        chain_id=chain_id,
        edge_keys=list(c.edge_keys),
        score=c.score,
        source_graph=c.source_graph,
        source_graphs=list(c.source_graphs or [c.source_graph]),
        edges=list(c.edges),
        fans={h: list(fl) for h, fl in c.fans.items()},
        fan_hub_names=dict(c.fan_hub_names),
        walk=list(c.walk),
    )


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
                    e.confidence = parse_confidence(row.get("confidence"))
    for c in chains:
        c.text = c.format_unit(c.chain_id)


def prepare_s5_batch(s4_pool: list[Chain]) -> list[Chain]:
    unique = dedup_s4_pool(s4_pool)
    batch = [_copy_labeled(c, f"c{i + 1}") for i, c in enumerate(unique)]
    logger.info("S5 unique=%s from pool=%s", len(batch), len(s4_pool))
    return batch

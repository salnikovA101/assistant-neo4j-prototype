"""Deterministic metrics for frozen multi-turn GraphRAG trajectories."""

from __future__ import annotations

import random
from typing import Any


def unit_signature(unit: dict[str, Any]) -> tuple[str, ...]:
    raw = unit.get("spine_evidence_seq") or unit.get("edge_keys") or []
    return tuple(str(value) for value in raw)


def trajectory_metrics(batches: list[list[dict[str, Any]]]) -> dict[str, Any]:
    seen: set[tuple[str, ...]] = set()
    previous: set[tuple[str, ...]] = set()
    iterations: list[dict[str, Any]] = []
    for index, batch in enumerate(batches, start=1):
        current = {unit_signature(unit) for unit in batch if unit_signature(unit)}
        unique = current - seen
        union = current | previous
        iterations.append({
            "iteration": index,
            "n": len(current),
            "uniqueEvidenceGain": len(unique),
            "duplicateRate": 1.0 - (len(unique) / len(current)) if current else 0.0,
            "jaccardPrevious": (len(current & previous) / len(union)) if union else 0.0,
        })
        seen.update(current)
        previous = current
    return {"iterations": iterations, "totalUniqueUnits": len(seen)}


def prefix_consistent(stepwise: list[dict[str, Any]], one_shot: list[dict[str, Any]]) -> bool:
    left = [unit_signature(unit) for unit in stepwise]
    right = [unit_signature(unit) for unit in one_shot[: len(left)]]
    return left == right


def paired_bootstrap_ci(
    rows: list[tuple[float, float]],
    *,
    samples: int = 5000,
    seed: int = 17,
) -> dict[str, float]:
    """95% CI for paired mean delta (candidate - baseline)."""
    if not rows:
        return {"delta": 0.0, "low": 0.0, "high": 0.0}
    deltas = [candidate - baseline for baseline, candidate in rows]
    rng = random.Random(seed)
    means = sorted(
        sum(rng.choice(deltas) for _ in deltas) / len(deltas)
        for _ in range(max(100, samples))
    )
    return {
        "delta": sum(deltas) / len(deltas),
        "low": means[int(0.025 * (len(means) - 1))],
        "high": means[int(0.975 * (len(means) - 1))],
    }

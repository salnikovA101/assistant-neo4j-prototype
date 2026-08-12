"""In-memory accepted-chain snapshots for on-demand graph visualization."""

from __future__ import annotations

import copy
import time
import uuid
from contextvars import ContextVar, Token
from typing import Any

_current_graph_collector: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "current_graph_collector",
    default=None,
)


def new_graph_collector() -> Token:
    """Start collecting accepted chains for the current assistant turn."""
    return _current_graph_collector.set([])


def reset_graph_collector(token: Token) -> None:
    _current_graph_collector.reset(token)


def current_graph_collector() -> list[dict[str, Any]] | None:
    return _current_graph_collector.get()


def record_accepted_chains(chains: list[dict[str, Any]] | None) -> None:
    """Append accepted chain dicts to the active turn collector, if any."""
    if not chains:
        return
    collector = _current_graph_collector.get()
    if collector is None:
        return
    collector.extend(copy.deepcopy(chains))


class GraphRunStore:
    """Small TTL store; graph payloads are rebuilt from Neo4j on demand."""

    def __init__(self, ttl_seconds: int = 1800, max_entries: int = 100) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._runs: dict[str, tuple[float, list[dict[str, Any]]]] = {}

    def _prune(self) -> None:
        now = time.monotonic()
        expired = [
            run_id
            for run_id, (created_at, _) in self._runs.items()
            if now - created_at > self.ttl_seconds
        ]
        for run_id in expired:
            self._runs.pop(run_id, None)

        while len(self._runs) >= self.max_entries:
            oldest = min(self._runs.items(), key=lambda item: item[1][0])[0]
            self._runs.pop(oldest, None)

    def put(self, chains: list[dict[str, Any]]) -> str:
        snapshot = [c for c in chains if isinstance(c, dict)]
        if not snapshot:
            return ""
        self._prune()
        run_id = f"gr_{uuid.uuid4().hex[:12]}"
        self._runs[run_id] = (time.monotonic(), copy.deepcopy(snapshot))
        return run_id

    def get(self, run_id: str) -> list[dict[str, Any]] | None:
        if not run_id:
            return None
        self._prune()
        item = self._runs.get(run_id)
        if item is None:
            return None
        return copy.deepcopy(item[1])


graph_run_store = GraphRunStore()

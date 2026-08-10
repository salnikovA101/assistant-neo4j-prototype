"""S7 pheromone: evaporate then deposit on edge_key."""

from __future__ import annotations

from typing import Iterable, Mapping

from server.algorithm.params import Params


def get_tau(store: Mapping[str, float], edge_key: str) -> float:
    return float(store.get(edge_key, 1.0))


def evaporate_all(store: dict[str, float], params: Params) -> None:
    rho = float(params.tau_rho)
    dead: list[str] = []
    for k, v in list(store.items()):
        nv = 1.0 + rho * (float(v) - 1.0)
        if abs(nv - 1.0) < 1e-6:
            dead.append(k)
        else:
            store[k] = nv
    for k in dead:
        store.pop(k, None)


def deposit_accepted(
    store: dict[str, float],
    edge_keys: Iterable[str],
    params: Params,
) -> None:
    delta = float(params.tau_delta)
    tmax = float(params.tau_max)
    for ek in edge_keys:
        if not ek:
            continue
        cur = get_tau(store, ek)
        store[ek] = min(tmax, cur + delta)


def apply_s7(
    store: dict[str, float],
    accepted_edge_keys: Iterable[str],
    params: Params,
) -> dict[str, float]:
    """Evaporate entire store, then deposit accepted edges. Returns store."""
    evaporate_all(store, params)
    deposit_accepted(store, accepted_edge_keys, params)
    return store

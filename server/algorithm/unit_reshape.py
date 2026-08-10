"""Reshape a linear star walk into SPINE + FANS by tracking the transition hub.

Walk edges are ordered; consecutive edges share exactly one node. While the
shared node between edge i-1 and i stays the same, edges beyond the entry ray
are FANS of that hub. When the shared node changes, the edge goes to SPINE and
the hub advances.

Segment at hub H:  entry(→H) [spine], ray1..rayN [fans@H], exit(H→) [spine].
Pure through walks have no fans.
"""

from __future__ import annotations

from server.algorithm.models import EdgeRecord, format_node_ref


def _shared_id(a: EdgeRecord, b: EdgeRecord) -> str:
    shared = {a.start_id, a.end_id} & {b.start_id, b.end_id}
    return next(iter(shared)) if len(shared) == 1 else ""


def _hub_name(hub_id: str, edges: list[EdgeRecord]) -> str:
    for e in edges:
        if e.start_id == hub_id and e.start_name:
            return format_node_ref(e.start_label, e.start_name, hub_id)
        if e.end_id == hub_id and e.end_name:
            return format_node_ref(e.end_label, e.end_name, hub_id)
    return hub_id


def reshape_star_walk(
    path_edges: list[EdgeRecord],
) -> tuple[list[EdgeRecord], dict[str, list[EdgeRecord]], dict[str, str]]:
    """
    Split ordered walk into SPINE + FANS at the current transition hub.

    Rule: keep the shared node between consecutive edges as `current_hub`.
    While it does not change, extra rays are fans of that hub; only entry and
    exit of a hub segment stay on the spine. Pure through walks stay all-spine.
    """
    if not path_edges:
        return [], {}, {}

    n = len(path_edges)
    spine: list[EdgeRecord] = [path_edges[0]]
    fans: dict[str, list[EdgeRecord]] = {}
    hub_names: dict[str, str] = {}

    i = 0
    while i < n - 1:
        hub = _shared_id(path_edges[i], path_edges[i + 1])
        if not hub:
            # Broken stitch: treat next edge as new spine start
            spine.append(path_edges[i + 1])
            i += 1
            continue

        # Collect the run of consecutive edges incident to this hub
        j = i + 1
        while j + 1 < n and _shared_id(path_edges[j], path_edges[j + 1]) == hub:
            j += 1
        run_len = j - i + 1  # edges[i..j]

        if run_len >= 3:
            # entry stays (already in spine), middle rays → fans, exit → spine
            for idx in range(i + 1, j):
                fans.setdefault(hub, []).append(path_edges[idx])
            hub_names[hub] = _hub_name(hub, path_edges)
            spine.append(path_edges[j])
        else:
            # Pure through (or pair): both edges on spine
            spine.append(path_edges[j])
        i = j

    return spine, fans, hub_names

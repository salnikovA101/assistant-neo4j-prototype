"""Reshape a linear star walk into SPINE + FANS by tracking the transition hub.

Walk edges are ordered; consecutive edges share exactly one node. While the
shared node between edge i-1 and i stays the same, edges beyond the entry ray
are FANS of that hub. When the shared node changes, the edge goes to SPINE and
the hub advances.

Segment at hub H:  entry(→H) [spine], ray1..rayN [fans@H], exit(H→) [spine].
Pure through walks have no fans.

Print tags (`linger_hubs`): same segments, but exit stays tagged @H so the
tour reads "still at H" until the shared vertex changes.
"""

from __future__ import annotations

from server.algorithm.models import EdgeRecord, format_node_ref


def _shared_id(a: EdgeRecord, b: EdgeRecord) -> str:
    shared = {a.start_id, a.end_id} & {b.start_id, b.end_id}
    return next(iter(shared)) if len(shared) == 1 else ""


def _incident(e: EdgeRecord, hub_id: str) -> bool:
    return bool(hub_id) and hub_id in {e.start_id, e.end_id}


def _hub_name(hub_id: str, edges: list[EdgeRecord]) -> str:
    for e in edges:
        if e.start_id == hub_id and e.start_name:
            return format_node_ref(e.start_label, e.start_name, hub_id)
        if e.end_id == hub_id and e.end_name:
            return format_node_ref(e.end_label, e.end_name, hub_id)
    return hub_id


def hub_display_name(
    hub_id: str,
    edges: list[EdgeRecord],
    names: dict[str, str] | None = None,
) -> str:
    """Prefer reshape `fan_hub_names`, else `Label: name` from an incident edge."""
    if names:
        got = (names.get(hub_id) or "").strip()
        if got:
            return got
    return _hub_name(hub_id, edges)


def linger_hubs(path_edges: list[EdgeRecord]) -> list[str]:
    """Hub id to prefix `@Hub` on each walk edge, or `""`.

    Star segment (≥3 edges sharing H): entry unmarked; rays **and** exit tagged
    H. Through pairs stay unmarked.
    """
    n = len(path_edges)
    tags = [""] * n
    if n < 3:
        return tags
    i = 0
    while i < n - 1:
        hub = _shared_id(path_edges[i], path_edges[i + 1])
        if not hub:
            i += 1
            continue
        j = i + 1
        while j + 1 < n and _shared_id(path_edges[j], path_edges[j + 1]) == hub:
            j += 1
        if (j - i + 1) >= 3:
            for idx in range(i + 1, j + 1):
                tags[idx] = hub
        i = j
    return tags


def reconstruct_walk(
    spine: list[EdgeRecord],
    fans: dict[str, list[EdgeRecord]],
) -> list[EdgeRecord]:
    """Inverse of reshape: insert each hub's rays between spine entry and exit."""
    if not spine:
        out: list[EdgeRecord] = []
        for flist in fans.values():
            out.extend(flist)
        return out

    placed: set[str] = set()
    out = [spine[0]]
    for i in range(len(spine) - 1):
        hub = _shared_id(spine[i], spine[i + 1])
        if hub and hub in fans and hub not in placed:
            out.extend(fans[hub])
            placed.add(hub)
        out.append(spine[i + 1])

    for hub, flist in fans.items():
        if not flist or hub in placed:
            continue
        insert_at: int | None = None
        for idx, e in enumerate(out):
            if _incident(e, hub):
                insert_at = idx + 1
                break
        if insert_at is None:
            out.extend(flist)
        else:
            out[insert_at:insert_at] = flist
        placed.add(hub)
    return out


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
            # Discontinuous walk: treat the next edge as a new spine start
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

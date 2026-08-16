"""Session-scoped source_file ↔ id mapping and user-facing citation render."""

from __future__ import annotations

import re
from typing import Iterable


class SourceRegistry:
    """Maps source_file strings to stable session ids starting at 1."""

    def __init__(self) -> None:
        self._file_to_id: dict[str, int] = {}
        self._id_to_file: dict[int, str] = {}
        self._next_id = 1

    def clear(self) -> None:
        self._file_to_id.clear()
        self._id_to_file.clear()
        self._next_id = 1

    def register(self, source_file: str) -> int | None:
        key = (source_file or "").strip()
        if not key:
            return None
        existing = self._file_to_id.get(key)
        if existing is not None:
            return existing
        sid = self._next_id
        self._next_id += 1
        self._file_to_id[key] = sid
        self._id_to_file[sid] = key
        return sid

    def resolve(self, source_id: int) -> str | None:
        return self._id_to_file.get(int(source_id))

    def known_files(self) -> list[str]:
        return list(self._file_to_id.keys())


# (source:1), (source:1; source:3), (source 1), mixed whitespace
_SOURCE_GROUP_RE = re.compile(
    r"\(\s*source\s*:?\s*\d+(?:\s*;\s*source\s*:?\s*\d+)*\s*\)",
    flags=re.IGNORECASE,
)
_SOURCE_ID_RE = re.compile(r"source\s*:?\s*(\d+)", flags=re.IGNORECASE)

_ISTOCHNIKI_SECTION_RE = re.compile(
    r"(?ms)^[ \t]*###[ \t]*Источники[ \t]*\n.*?(?=^[ \t]*###[ \t]+\S|\Z)"
)


def session_source_ids_in_text(text: str) -> list[int]:
    """Unique session source ids found in text, sorted."""
    found = {int(x) for x in _SOURCE_ID_RE.findall(text or "")}
    return sorted(found)


def format_source_id_list(ids: Iterable[int]) -> str:
    """Compact id list: 1, 2, 5-9 (pairs stay comma-separated)."""
    nums = sorted({int(x) for x in ids})
    if not nums:
        return ""
    ranges: list[tuple[int, int]] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append((start, prev))
        start = prev = n
    ranges.append((start, prev))
    parts: list[str] = []
    for a, b in ranges:
        if a == b:
            parts.append(str(a))
        elif b == a + 1:
            parts.append(f"{a}, {b}")
        else:
            parts.append(f"{a}-{b}")
    return ", ".join(parts)


def tool_history_stub(result: str, *, ok: bool = True) -> str:
    """
    Compact tool.content for chat history. Not an empty-search marker.
    Full UNIT stays in the live UI event only.
    """
    if not ok:
        msg = (result or "unknown").strip()
        if msg.lower().startswith("error:"):
            msg = msg[6:].strip()
        if len(msg) > 200:
            msg = msg[:199] + "…"
        return f"Tool error: {msg}"
    ids = session_source_ids_in_text(result)
    if ids:
        listed = format_source_id_list(ids)
        return (
            f"Retrieved data. Session sources {listed} (stable ids). "
            "Facts are in the following assistant message. Not an empty result."
        )
    return (
        "Retrieved data. No source:N in this result. Not an empty result."
    )


def collect_source_files(accepted: Iterable[dict]) -> list[str]:
    """Unique source_file values from accepted chain dicts (edges + fans)."""
    seen: set[str] = set()
    out: list[str] = []
    for item in accepted:
        for sf in _iter_chain_source_files(item):
            if sf not in seen:
                seen.add(sf)
                out.append(sf)
    return out


def _iter_chain_source_files(chain: dict) -> Iterable[str]:
    for edge in chain.get("edges") or []:
        if not isinstance(edge, dict):
            continue
        sf = (edge.get("source_file") or "").strip()
        if sf:
            yield sf
    fans = chain.get("fans") or {}
    for flist in fans.values():
        for edge in flist or []:
            if not isinstance(edge, dict):
                continue
            sf = (edge.get("source_file") or "").strip()
            if sf:
                yield sf


def extract_cited_source_files(text: str, registry: SourceRegistry) -> list[str]:
    """
    Unique source_file values cited as (source:N) in the raw answer,
    in first-seen order. Unknown session ids are ignored.
    """
    seen: set[str] = set()
    out: list[str] = []
    for raw_id in _SOURCE_ID_RE.findall(text or ""):
        fname = registry.resolve(int(raw_id))
        if not fname or fname in seen:
            continue
        seen.add(fname)
        out.append(fname)
    return out


def filter_chains_by_source_files(
    chains: Iterable[dict],
    cited_files: Iterable[str],
) -> list[dict]:
    """Keep chains that share at least one source_file with cited_files."""
    wanted = {(f or "").strip() for f in cited_files if (f or "").strip()}
    if not wanted:
        return []
    out: list[dict] = []
    for chain in chains:
        if not isinstance(chain, dict):
            continue
        if any(sf in wanted for sf in _iter_chain_source_files(chain)):
            out.append(chain)
    return out


def remap_filenames_to_source_ids(text: str, registry: SourceRegistry) -> str:
    """
    Replace registered filenames in UNIT text with source:N.
    Longest filenames first to avoid partial overlaps.
    """
    files = sorted(registry.known_files(), key=len, reverse=True)
    out = text
    for fname in files:
        sid = registry.register(fname)
        if sid is None:
            continue
        out = out.replace(fname, f"source:{sid}")
    return out


def render_citations(text: str, registry: SourceRegistry) -> str:
    """
    Convert model citations (source:N) → dense [1]..[k] for the user (per answer)
    and append ### Источники. Session ids stay in history; only the display layer
    renumbers by first-citation order in this answer.
    """
    raw = text or ""
    # Drop model-written bibliography if present
    cleaned = _ISTOCHNIKI_SECTION_RE.sub("", raw).rstrip()

    cited_order: list[int] = []  # session ids, first-seen in this answer
    cited_set: set[int] = set()
    session_to_display: dict[int, int] = {}

    def _display_id(session_id: int) -> int | None:
        if registry.resolve(session_id) is None:
            return None
        if session_id not in session_to_display:
            session_to_display[session_id] = len(session_to_display) + 1
            if session_id not in cited_set:
                cited_set.add(session_id)
                cited_order.append(session_id)
        return session_to_display[session_id]

    def _repl_group(match: re.Match[str]) -> str:
        ids = [int(x) for x in _SOURCE_ID_RE.findall(match.group(0))]
        markers: list[str] = []
        for sid in ids:
            did = _display_id(sid)
            if did is None:
                continue
            markers.append(f"[{did}]")
        return "".join(markers)

    rendered = _SOURCE_GROUP_RE.sub(_repl_group, cleaned)

    def _repl_bare(match: re.Match[str]) -> str:
        sid = int(match.group(1))
        did = _display_id(sid)
        if did is None:
            return ""
        return f"[{did}]"

    rendered = re.sub(
        r"(?<!\w)source\s*:?\s*(\d+)(?!\w)",
        _repl_bare,
        rendered,
        flags=re.IGNORECASE,
    )

    if not cited_order:
        return rendered.rstrip() + "\n"

    lines = [rendered.rstrip(), "", "### Источники"]
    for sid in cited_order:
        did = session_to_display[sid]
        name = registry.resolve(sid) or ""
        lines.append(f"[{did}] {name}")
    return "\n".join(lines) + "\n"

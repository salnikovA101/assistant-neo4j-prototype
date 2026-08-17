"""Typed models for Algorithm V6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Schema labels for unit display; Cypher whitelist in cypher/edges.py imports this.
PRIMARY_NODE_LABELS: tuple[str, ...] = (
    "Metabolite",
    "Microbe",
    "StarterCulture",
    "EnvironmentCondition",
)


def pick_primary_label(labels: list[str] | tuple[str, ...] | None) -> str:
    if not labels:
        return ""
    for primary in PRIMARY_NODE_LABELS:
        if primary in labels:
            return primary
    return ""


def format_node_ref(label: str, name: str, fallback_id: str = "") -> str:
    """Display `Label: name` when a primary label is known; else bare name/id."""
    nm = (name or "").strip() or (fallback_id or "").strip()
    lbl = (label or "").strip()
    if lbl and nm:
        return f"{lbl}: {nm}"
    return nm or "?"


def _edge_ev_key(evidence: str, chunk_id: str = "") -> str:
    ev = (evidence or "").strip()
    if ev:
        return ev
    return (chunk_id or "").strip()


def _quote_ev(evidence: str) -> str:
    return (evidence or "").replace('"', "'")


def _edge_meta_suffix(source_file: str = "", confidence: float | None = None) -> str:
    """Trailing meta: always both keys. Filename remaps to source:N later."""
    sf = (source_file or "").strip()
    src = sf if sf else "source:None"
    conf = f"{float(confidence):.2f}" if confidence is not None else "None"
    return f"  ({src}; conf={conf})"


def _directed_edge_line(e: EdgeRecord, hub_display: str = "") -> str:
    """Triple line: optional `@Hub  ` then Label: start —REL→ Label: end."""
    triple = f"{e.start_ref()} —{e.rel_type}→ {e.end_ref()}"
    hub = (hub_display or "").strip()
    if hub:
        return f"@{hub}  {triple}"
    return triple


def _edge_card_lines(e: EdgeRecord, hub_display: str = "") -> list[str]:
    """One edge as triple + indented quote with (source; conf) on the quote line."""
    return [
        _directed_edge_line(e, hub_display=hub_display),
        f'  "{_quote_ev(e.evidence)}"{_edge_meta_suffix(e.source_file, e.confidence)}',
    ]


@dataclass
class SubQuestion:
    id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text}


@dataclass
class EdgeRecord:
    edge_key: str
    element_id: str
    rel_type: str
    start_id: str
    end_id: str
    start_name: str
    end_name: str
    start_label: str = ""
    end_label: str = ""
    sim: float = 0.0
    rerank_score: float | None = None
    embedding: list[float] = field(default_factory=list)
    chunk_id: str = ""
    evidence: str = ""
    source_file: str = ""
    source: str = "ann"  # graph: ann|bridge; after S4 on chains: prize|bridge
    confidence: float | None = 1.0

    def start_ref(self) -> str:
        return format_node_ref(self.start_label, self.start_name, self.start_id)

    def end_ref(self) -> str:
        return format_node_ref(self.end_label, self.end_name, self.end_id)

    def to_dict_edge(self) -> dict[str, Any]:
        return {
            "edge_key": self.edge_key,
            "element_id": self.element_id,
            "type": self.rel_type,
            "start": self.start_name,
            "end": self.end_name,
            "start_label": self.start_label,
            "end_label": self.end_label,
            "start_id": self.start_id,
            "end_id": self.end_id,
            "evidence": self.evidence,
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "sim": round(self.sim, 4),
            "confidence": (
                round(self.confidence, 4) if self.confidence is not None else None
            ),
            "source": self.source,
        }


@dataclass
class CandidateGraph:
    source_graph: str
    edges: dict[str, EdgeRecord] = field(default_factory=dict)
    node_to_edges: dict[str, list[str]] = field(default_factory=dict)
    transition_adj: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Chain:
    """Evidence unit: walk-ordered tour; spine+fans kept for viz/S5."""

    chain_id: str
    edge_keys: list[str]
    score: float
    source_graph: str = ""
    source_graphs: list[str] = field(default_factory=list)
    edges: list[EdgeRecord] = field(default_factory=list)
    # hub_id -> fan edges walked at that hub
    fans: dict[str, list[EdgeRecord]] = field(default_factory=dict)
    # hub_id -> display name
    fan_hub_names: dict[str, str] = field(default_factory=dict)
    # hop-DP order; empty → reconstruct from spine+fans at format time
    walk: list[EdgeRecord] = field(default_factory=list)
    text: str = ""

    def all_edges(self) -> list[EdgeRecord]:
        if self.walk:
            return list(self.walk)
        out = list(self.edges)
        for flist in self.fans.values():
            out.extend(flist)
        return out

    def all_edge_keys(self) -> list[str]:
        if self.walk:
            return list(dict.fromkeys(e.edge_key for e in self.walk))
        keys = list(self.edge_keys)
        for flist in self.fans.values():
            for e in flist:
                if e.edge_key not in keys:
                    keys.append(e.edge_key)
        return keys

    def spine_evidence_seq(self) -> tuple[str, ...]:
        """Ordered spine evidence keys; fallback to edge_keys if all empty."""
        seq: list[str] = []
        for e in self.edges:
            k = _edge_ev_key(e.evidence, e.chunk_id)
            if k:
                seq.append(k)
        if seq:
            return tuple(seq)
        return tuple(self.edge_keys)

    def format_unit(self, uid: str | None = None) -> str:
        from server.algorithm.unit_reshape import (
            hub_display_name,
            linger_hubs,
            reconstruct_walk,
        )

        label = uid or self.chain_id
        lines = [f"UNIT {label}"]
        walk = list(self.walk) if self.walk else reconstruct_walk(self.edges, self.fans)
        if not walk:
            for k in self.edge_keys:
                lines.append(str(k))
            return "\n".join(lines)
        tags = linger_hubs(walk)
        for e, hub_id in zip(walk, tags, strict=True):
            display = (
                hub_display_name(hub_id, walk, self.fan_hub_names) if hub_id else ""
            )
            lines.extend(_edge_card_lines(e, hub_display=display))
        return "\n".join(lines)

    def brief(self) -> str:
        if self.text:
            return self.text
        return self.format_unit(self.chain_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chain_id": self.chain_id,
            "score": round(self.score, 6),
            "edge_keys": list(self.edge_keys),
            "source_graph": self.source_graph,
            "source_graphs": list(self.source_graphs),
            "text": self.brief(),
            "edges": [e.to_dict_edge() for e in self.edges],
            "fans": {hub: [e.to_dict_edge() for e in flist] for hub, flist in self.fans.items()},
            "fan_hub_names": dict(self.fan_hub_names),
            "walk": [e.to_dict_edge() for e in self.walk],
            "spine_evidence_seq": list(self.spine_evidence_seq()),
        }


@dataclass
class SessionState:
    """Per-question run: subquestions, accepted units, embed cache."""

    subquestions: list[SubQuestion] = field(default_factory=list)
    accepted: list[Chain] = field(default_factory=list)
    embed_cache: dict[str, list[float]] = field(default_factory=dict)

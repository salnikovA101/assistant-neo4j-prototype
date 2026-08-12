"""Typed models for Algorithm V6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Schema node labels only (nodes often carry an extra non-schema label too).
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
    """Trailing meta: (file.pdf; conf=0.87) — only present fields."""
    parts: list[str] = []
    sf = (source_file or "").strip()
    if sf:
        parts.append(sf)
    if confidence is not None:
        parts.append(f"conf={float(confidence):.2f}")
    if not parts:
        return ""
    return f"  ({'; '.join(parts)})"


def _directed_edge_line(e: EdgeRecord) -> str:
    """Neo4j direction: Label: start -[REL: ev]-> Label: end  (source; conf)."""
    ev = _quote_ev(e.evidence)
    return (
        f'{e.start_ref()} -[{e.rel_type}: "{ev}"]-> {e.end_ref()}'
        f"{_edge_meta_suffix(e.source_file, e.confidence)}"
    )


def _fan_leaf_line_out(
    rel_type: str,
    evidence: str,
    leaf: str,
    source_file: str = "",
    confidence: float | None = None,
) -> str:
    ev = _quote_ev(evidence)
    return (
        f'-[{rel_type}: "{ev}"]-> {leaf}'
        f"{_edge_meta_suffix(source_file, confidence)}"
    )


def _fan_leaf_line_in(
    rel_type: str,
    evidence: str,
    leaf: str,
    source_file: str = "",
    confidence: float | None = None,
) -> str:
    ev = _quote_ev(evidence)
    return (
        f'<-[{rel_type}: "{ev}"]- {leaf}'
        f"{_edge_meta_suffix(source_file, confidence)}"
    )


@dataclass
class SubQuestion:
    id: str
    text: str
    closed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "text": self.text, "closed": self.closed}


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
    rerank_score: float = 0.0
    embedding: list[float] = field(default_factory=list)
    chunk_id: str = ""
    evidence: str = ""
    source_file: str = ""
    source: str = "ann"  # graph: ann|bridge; after S4 on chains: prize|bridge
    confidence: float = 1.0

    def start_ref(self) -> str:
        return format_node_ref(self.start_label, self.start_name, self.start_id)

    def end_ref(self) -> str:
        return format_node_ref(self.end_label, self.end_name, self.end_id)

    def to_brief_part(self) -> str:
        ev = self.evidence or ""
        return f"{self.start_ref()}-[{self.rel_type}]->{self.end_ref()}: {ev}"

    def to_spine_line(self) -> str:
        return _directed_edge_line(self)

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
            "confidence": round(self.confidence, 4),
            "source": self.source,
        }


@dataclass
class CandidateGraph:
    source_graph: str
    edges: dict[str, EdgeRecord] = field(default_factory=dict)
    node_to_edges: dict[str, list[str]] = field(default_factory=dict)
    transition_adj: dict[str, list[str]] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.edges)


def _fan_line(hub_id: str, e: EdgeRecord) -> str:
    """Hub-centric fan line with Neo4j direction: out -> Leaf, in <- Leaf."""
    if e.start_id == hub_id:
        return _fan_leaf_line_out(
            e.rel_type,
            e.evidence,
            e.end_ref(),
            e.source_file,
            e.confidence,
        )
    if e.end_id == hub_id:
        return _fan_leaf_line_in(
            e.rel_type,
            e.evidence,
            e.start_ref(),
            e.source_file,
            e.confidence,
        )
    return _directed_edge_line(e)


def format_spine_lines(edges: list[EdgeRecord]) -> list[str]:
    """
    Directed spine lines in Neo4j start→end order.

    SPINE/FANS membership is decided upstream by reshape (transition-hub rule);
    arrows always reflect the stored relationship direction (A -> B, C -> B).
    """
    return [_directed_edge_line(e) for e in edges]


@dataclass
class Chain:
    """Evidence unit: SPINE (ordered) + optional FANS at hubs."""

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
    text: str = ""

    def all_edges(self) -> list[EdgeRecord]:
        out = list(self.edges)
        for flist in self.fans.values():
            out.extend(flist)
        return out

    def all_edge_keys(self) -> list[str]:
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
        label = uid or self.chain_id
        lines = [f"UNIT {label}  (score={self.score:.4f})", "SPINE:"]
        if self.edges:
            for line in format_spine_lines(self.edges):
                lines.append(f"  {line}")
        else:
            for k in self.edge_keys:
                lines.append(f"  {k}")
        for hub_id, flist in self.fans.items():
            if not flist:
                continue
            hub_name = self.fan_hub_names.get(hub_id) or hub_id
            lines.append(f"FANS @{hub_name}:")
            for e in flist:
                lines.append(f"  {_fan_line(hub_id, e)}")
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
            "spine_evidence_seq": list(self.spine_evidence_seq()),
        }


UNIT_RULES = """\
RULES:
- SPINE: one line = one directed edge
  Label: A -[REL: "evidence"]-> Label: B  (source_file.pdf; conf=0.87),
  top to bottom. Labels are Microbe / Metabolite / StarterCulture /
  EnvironmentCondition. Adjacent lines may share an endpoint (walk order);
  arrow is Neo4j direction. Trailing (...) lists source_file and/or conf when present.
- FANS @Hub: hub-centric leaves only — out-star -[REL]-> Leaf  (source; conf),
  in-star <-[REL]- Leaf  (source; conf). Sibling fans are NOT linked to each other.
- Do not invent facts outside the listed units.
- User-facing bibliography uses [n] → ### Источники mapped from these source_file
  values; do not cite UNIT indices as sources. Edge conf is graph provenance, not
  an answer-level confidence score.
""".strip()


@dataclass
class SessionState:
    """Per tool-call session (p, accepted, open sq)."""

    subquestions: list[SubQuestion] = field(default_factory=list)
    accepted: list[Chain] = field(default_factory=list)
    p_edges: dict[str, float] = field(default_factory=dict)
    embed_cache: dict[str, list[float]] = field(default_factory=dict)
    used_edges: set[str] = field(default_factory=set)
    # Exact spine evidence sequences already shown (accepted + rejected)
    seen_spine_seqs: set[tuple[str, ...]] = field(default_factory=set)

    def open_sqs(self) -> list[SubQuestion]:
        return [s for s in self.subquestions if not s.closed]

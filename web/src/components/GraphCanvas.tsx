import { useEffect, useMemo, useRef, useState } from "react";
import { DataSet, Network } from "vis-network/standalone";
import "vis-network/styles/vis-network.min.css";
import type { GraphEdge, GraphNode, GraphPayload } from "../types";
import { tripletCaption } from "../format";

type Selected =
  | { kind: "node"; node: GraphNode }
  | { kind: "edge"; edge: GraphEdge }
  | null;

const NETWORK_OPTIONS = {
  autoResize: false,
  physics: {
    enabled: true,
    solver: "forceAtlas2Based" as const,
    forceAtlas2Based: {
      gravitationalConstant: -90,
      centralGravity: 0.01,
      springLength: 180,
      springConstant: 0.08,
      damping: 0.4,
      avoidOverlap: 0.6,
    },
    stabilization: { enabled: true, iterations: 160, fit: true },
  },
  interaction: {
    hover: true,
    tooltipDelay: 180,
    zoomView: true,
    dragView: true,
  },
  layout: { hierarchical: { enabled: false } },
  nodes: {
    shape: "dot",
    size: 18,
    font: { size: 13, face: "Inter, sans-serif", color: "#ececec" },
    borderWidth: 0,
  },
  edges: {
    arrows: { to: { enabled: true, scaleFactor: 0.55 } },
    color: { color: "rgba(255,255,255,0.28)", highlight: "#4d9fff" },
    font: { size: 10, color: "#9a9a9a", strokeWidth: 0 },
    smooth: { enabled: true, type: "cubicBezier", roundness: 0.35 },
  },
};

function captionOf(node: GraphNode): string {
  return (node.caption || "").trim() || node.group || node.id;
}

async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) {
    await navigator.clipboard.writeText(text);
    return;
  }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.cssText = "position:fixed;opacity:0";
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  ta.remove();
}

function hasSize(el: HTMLElement): boolean {
  return el.clientWidth >= 16 && el.clientHeight >= 16;
}

export function GraphCanvas({
  payload,
  viewId,
  emptyHint,
  hideSearch = false,
  focusEdgeId = "",
  onExpandNode,
}: {
  payload: GraphPayload | null;
  viewId: string | "all";
  emptyHint: string;
  hideSearch?: boolean;
  focusEdgeId?: string;
  onExpandNode?: (nodeId: string) => void;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const netRef = useRef<Network | null>(null);
  const [selected, setSelected] = useState<Selected>(null);
  const [query, setQuery] = useState("");
  const [pinned, setPinned] = useState<string[]>([]);

  const graph = useMemo(() => {
    if (!payload) return { nodes: [] as GraphNode[], edges: [] as GraphEdge[] };
    if (viewId === "all") return payload.all;
    return payload.views.find((item) => item.id === viewId) || payload.all;
  }, [payload, viewId]);

  useEffect(() => {
    setPinned([]);
    setSelected(null);
    setQuery("");
  }, [payload, viewId]);

  useEffect(() => {
    if (!focusEdgeId) return;
    const edge = graph.edges.find((item) => item.id === focusEdgeId);
    if (!edge) return;
    setSelected({ kind: "edge", edge });
    setPinned([edge.id]);
  }, [focusEdgeId, graph]);

  const q = query.trim().toLowerCase();
  const hits = useMemo(() => {
    const edges = graph.edges.filter((e) => {
      if (!q) return true;
      const hay = [
        e.label,
        e.from_name,
        e.to_name,
        e.hub_name,
        String(e.properties?.evidence || ""),
        String(e.properties?.source_file || ""),
      ]
        .join(" ")
        .toLowerCase();
      return hay.includes(q);
    });
    return edges;
  }, [graph, q]);

  const filtered = useMemo(() => {
    const pinnedSet = new Set(pinned);
    const edges = pinnedSet.size
      ? graph.edges.filter((e) => pinnedSet.has(e.id))
      : q
        ? hits
        : graph.edges;
    const ids = new Set<string>();
    for (const e of edges) {
      ids.add(e.from);
      ids.add(e.to);
    }
    return {
      nodes: graph.nodes.filter((n) => ids.has(n.id)),
      edges,
    };
  }, [graph, hits, pinned, q]);

  const legend = useMemo(() => {
    const seen = new Map<string, string>();
    for (const node of graph.nodes) {
      if (!seen.has(node.group)) seen.set(node.group, node.color);
    }
    return [...seen.entries()];
  }, [graph]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host || !filtered.nodes.length) return;
    let cancelled = false;
    let network: Network | null = null;
    let ro: ResizeObserver | null = null;

    const nodes = new DataSet(
      filtered.nodes.map((n) => ({
        id: n.id,
        label: captionOf(n),
        color: {
          background: n.color || "#a5abb6",
          border: n.color || "#a5abb6",
          highlight: { background: n.color || "#a5abb6", border: "#ececec" },
        },
        title: `${n.group}: ${captionOf(n)}`,
      }))
    );
    const edges = new DataSet(
      filtered.edges.map((e) => ({
        id: e.id,
        from: e.from,
        to: e.to,
        label: e.label,
        title: `${e.from_name || e.from} —${e.label}→ ${e.to_name || e.to}`,
      }))
    );

    const applySize = (fit: boolean) => {
      if (!network || cancelled) return;
      const w = host.clientWidth;
      const h = host.clientHeight;
      if (w < 16 || h < 16) return;
      network.setSize(`${w}px`, `${h}px`);
      network.redraw();
      if (fit) network.fit({ animation: false });
    };

    const start = () => {
      if (cancelled || network) return;
      netRef.current?.destroy();
      network = new Network(host, { nodes, edges }, NETWORK_OPTIONS);
      netRef.current = network;
      network.on("selectNode", (ev: { nodes: string[] }) => {
        const node = graph.nodes.find((item) => item.id === ev.nodes[0]);
        setSelected(node ? { kind: "node", node } : null);
        if (node && onExpandNode) onExpandNode(node.id);
      });
      network.on("selectEdge", (ev: { edges: string[]; nodes: string[] }) => {
        if (ev.nodes.length) return;
        const edge = graph.edges.find((item) => item.id === ev.edges[0]);
        setSelected(edge ? { kind: "edge", edge } : null);
      });
      network.on("deselectNode", () => setSelected(null));
      network.on("deselectEdge", () => setSelected(null));
      network.once("stabilizationIterationsDone", () => applySize(true));
      applySize(true);
      ro?.disconnect();
      ro = new ResizeObserver(() => applySize(false));
      ro.observe(host);
    };

    if (hasSize(host)) {
      start();
    } else {
      ro = new ResizeObserver(() => {
        if (hasSize(host)) start();
      });
      ro.observe(host);
      requestAnimationFrame(() => {
        if (hasSize(host)) start();
      });
    }

    return () => {
      cancelled = true;
      ro?.disconnect();
      network?.destroy();
      if (netRef.current === network) netRef.current = null;
    };
  }, [filtered, graph, onExpandNode]);

  if (!payload) {
    return <div className="graph-empty">{emptyHint}</div>;
  }

  const pickTriplet = (edge: GraphEdge) => {
    setSelected({ kind: "edge", edge });
    setPinned((prev) => (prev.includes(edge.id) ? prev : [...prev, edge.id]));
    const net = netRef.current;
    if (!net) return;
    net.selectEdges([edge.id]);
    net.selectNodes([edge.from, edge.to]);
    net.focus(edge.from, { animation: false, scale: 1.1 });
  };

  return (
    <div className="graph-stage">
      {!hideSearch && (
        <div className="graph-local-search">
          <input
            className="graph-search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Найти ребро в этом графе"
          />
          {q && hits.length > 0 && (
            <div className="edge-suggestions" role="listbox" aria-label="Рёбра этого графа">
              {hits.slice(0, 12).map((edge) => (
                <button key={edge.id} type="button" onClick={() => { pickTriplet(edge); setQuery(""); }}>
                  <strong>{tripletCaption(edge)}</strong>
                  {Boolean(edge.properties?.evidence) && <span>{String(edge.properties.evidence)}</span>}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      {!filtered.nodes.length ? (
        <div className="graph-empty">{emptyHint || "Ничего не найдено"}</div>
      ) : (
        <div className="graph-body">
          <div className="graph-canvas-wrap">
            <div ref={hostRef} className="graph-canvas" />
            <div className="graph-legend">
              {legend.map(([group, color]) => (
                <span key={group} className="graph-legend-item">
                  <i style={{ background: color }} />
                  {group}
                </span>
              ))}
            </div>
          </div>
          <aside className="graph-inspector">
            {!selected && <p className="muted">Выберите ребро на графе или через поиск</p>}
            {selected?.kind === "node" && (
              <div>
                <p className="inspector-kicker">{selected.node.group}</p>
                <h3>{captionOf(selected.node)}</h3>
              </div>
            )}
            {selected?.kind === "edge" && <EdgeCard edge={selected.edge} />}
          </aside>
        </div>
      )}
    </div>
  );
}

function EdgeCard({ edge }: { edge: GraphEdge }) {
  const props = edge.properties || {};
  const evidence = String(props.evidence || "");
  const source = String(props.source_file || "");
  const conf = props.confidence;
  return (
    <div>
      <p className="inspector-kicker">{edge.label}</p>
      <h3>
        {edge.from_name || edge.from} → {edge.to_name || edge.to}
      </h3>
      {evidence && <blockquote className="inspector-quote">{evidence}</blockquote>}
      <dl className="inspector-meta">
        {source && (
          <>
            <dt>Источник</dt>
            <dd>{source}</dd>
          </>
        )}
        {conf != null && conf !== "" && (
          <>
            <dt>Уверенность</dt>
            <dd>{Number(conf).toFixed(2)}</dd>
          </>
        )}
      </dl>
      {evidence && (
        <button type="button" className="ghost-btn" onClick={() => copyText(evidence)}>
          Копировать цитату
        </button>
      )}
    </div>
  );
}

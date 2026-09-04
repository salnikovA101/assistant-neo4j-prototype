import { useEffect, useMemo, useRef, useState } from "react";
import { DataSet, Network } from "vis-network/standalone";
import "vis-network/styles/vis-network.min.css";
import type { GraphCollectionItem, GraphEdge, GraphNode, GraphPayload } from "../types";
import { tripletCaption } from "../format";

type Selected = { kind: "node"; node: GraphNode } | { kind: "edge"; edge: GraphEdge } | null;
export type GraphExpansionUi = { loaded: number; total: number; hasMore: boolean; busy: boolean };
export type GraphAppendEvent = { id: string; anchorNodeId: string; nodes: GraphNode[]; edges: GraphEdge[] };
type MutableDataSet = {
  getIds: () => Array<string | number>;
  add: (items: any[]) => void;
  update: (items: any[]) => void;
  remove: (ids: Array<string | number>) => void;
};
type NodePosition = { x: number; y: number };
type LayoutPositions = Record<string, NodePosition>;
type MutableNetwork = Network & {
  stopSimulation: () => void;
  setOptions: (options: { physics: { enabled: boolean } }) => void;
  getPosition: (nodeId: string) => NodePosition;
  getPositions: (nodeIds?: string[]) => LayoutPositions;
};

const NETWORK_OPTIONS = {
  autoResize: false,
  physics: {
    enabled: true,
    solver: "forceAtlas2Based" as const,
    forceAtlas2Based: { gravitationalConstant: -90, centralGravity: 0.01, springLength: 180, springConstant: 0.08, damping: 0.4, avoidOverlap: 0.6 },
    stabilization: { enabled: true, iterations: 160, fit: true },
  },
  interaction: { hover: true, tooltipDelay: 180, zoomView: true, dragView: true },
  layout: { hierarchical: { enabled: false } },
  nodes: { shape: "dot", size: 18, font: { size: 13, face: "Inter, sans-serif", color: "#ececec" }, borderWidth: 0 },
  edges: {
    arrows: { to: { enabled: true, scaleFactor: 0.55 } },
    color: { color: "rgba(255,255,255,0.28)", highlight: "#4d9fff" },
    font: { size: 10, color: "#9a9a9a", strokeWidth: 0 },
    smooth: { enabled: true, type: "cubicBezier", roundness: 0.35 },
  },
};

// vis-network caps fit() at 1.0 by default. Small evidence UNITs then occupy
// only a small patch in a large pane despite having enough room to be legible.
const FIT_MAX_ZOOM = 1.7;
const SMALL_GRAPH_FIT_MAX_ZOOM = 3;

function fitVisibleGraph(network: Network, nodeCount: number): void {
  network.fit({
    animation: false,
    maxZoomLevel: nodeCount <= 12 ? SMALL_GRAPH_FIT_MAX_ZOOM : FIT_MAX_ZOOM,
  });
}

function captionOf(node: GraphNode): string { return (node.caption || "").trim() || node.group || node.id; }
function nodeKey(node: GraphNode): string { return `node:${node.id}`; }
function edgeKey(edge: GraphEdge): string { return `edge:${edge.id}`; }

function collectionNode(edge: GraphEdge, side: "from" | "to"): GraphNode {
  const id = side === "from" ? edge.from : edge.to;
  const caption = side === "from" ? edge.from_name || id : edge.to_name || id;
  const group = side === "from" ? edge.from_group || "Сущность" : edge.to_group || "Сущность";
  return { id, label: caption, caption, group, color: side === "from" ? "#6ea8fe" : "#b48ad8", properties: {} };
}

async function copyText(text: string): Promise<void> {
  if (navigator.clipboard?.writeText) { await navigator.clipboard.writeText(text); return; }
  const ta = document.createElement("textarea");
  ta.value = text;
  ta.style.cssText = "position:fixed;opacity:0";
  document.body.appendChild(ta);
  ta.select();
  document.execCommand("copy");
  ta.remove();
}

function hasSize(el: HTMLElement): boolean { return el.clientWidth >= 16 && el.clientHeight >= 16; }

export function GraphCanvas({
  payload,
  viewId,
  emptyHint,
  hideSearch = false,
  focusEdgeId = "",
  layoutKey = "default",
  appendEvent,
  expansionByNode = {},
  onExpandNode,
  onCanvasInteraction,
  onUseCollection,
}: {
  payload: GraphPayload | null;
  viewId: string | "all";
  emptyHint: string;
  hideSearch?: boolean;
  focusEdgeId?: string;
  layoutKey?: string;
  appendEvent?: GraphAppendEvent | null;
  expansionByNode?: Record<string, GraphExpansionUi>;
  onExpandNode?: (nodeId: string, direction: "all" | "incoming" | "outgoing") => void;
  onCanvasInteraction?: () => void;
  onUseCollection?: (items: GraphCollectionItem[]) => void;
}) {
  const hostRef = useRef<HTMLDivElement>(null);
  const bodyRef = useRef<HTMLDivElement>(null);
  const inspectorRef = useRef<HTMLElement>(null);
  const netRef = useRef<Network | null>(null);
  const nodeDataRef = useRef<MutableDataSet | null>(null); // vis-network item shape is intentionally dynamic.
  const edgeDataRef = useRef<MutableDataSet | null>(null);
  const filteredRef = useRef<{ nodes: GraphNode[]; edges: GraphEdge[] }>({ nodes: [], edges: [] });
  const interactionRef = useRef(onCanvasInteraction);
  const appendAppliedRef = useRef("");
  const layoutPositionsRef = useRef<Map<string, LayoutPositions>>(new Map());
  const activeViewRef = useRef<string>(viewId);
  const [selected, setSelected] = useState<Selected>(null);
  const [query, setQuery] = useState("");
  const [suggestionsOpen, setSuggestionsOpen] = useState(false);
  const [collection, setCollection] = useState<GraphCollectionItem[]>([]);
  const [inspectorMode, setInspectorMode] = useState<"detail" | "collection">("detail");
  const [inspectorHeight, setInspectorHeight] = useState(220);
  const [expandDirection, setExpandDirection] = useState<"all" | "incoming" | "outgoing">("all");

  interactionRef.current = onCanvasInteraction;

  const resizeInspector = (next: number) => {
    const available = bodyRef.current?.clientHeight || window.innerHeight;
    setInspectorHeight(Math.max(110, Math.min(next, Math.max(110, available - 180))));
  };

  const graph = useMemo(() => {
    if (!payload) return { nodes: [] as GraphNode[], edges: [] as GraphEdge[] };
    if (viewId === "all") return payload.all;
    return payload.views.find((item) => item.id === viewId) || payload.all;
  }, [payload, viewId]);

  useEffect(() => { setSuggestionsOpen(false); }, [payload, viewId]);
  useEffect(() => {
    setSelected(null);
    setInspectorMode(collection.length ? "collection" : "detail");
    appendAppliedRef.current = "";
  }, [layoutKey]); // eslint-disable-line react-hooks/exhaustive-deps

  const q = query.trim().toLowerCase();
  const hits = useMemo(() => graph.edges.filter((edge) => {
    if (!q) return true;
    return [edge.label, edge.from_name, edge.to_name, edge.hub_name, String(edge.properties?.evidence || ""), String(edge.properties?.source_file || "")]
      .join(" ").toLowerCase().includes(q);
  }), [graph.edges, q]);

  const filtered = useMemo(() => {
    const edgesById = new Map<string, GraphEdge>();
    for (const edge of (q ? hits : graph.edges)) edgesById.set(edge.id, edge);
    for (const item of collection) if (item.kind === "edge") edgesById.set(item.edge.id, item.edge);
    const nodesById = new Map<string, GraphNode>(graph.nodes.map((node) => [node.id, node]));
    for (const item of collection) if (item.kind === "node") nodesById.set(item.node.id, item.node);
    for (const edge of edgesById.values()) {
      if (!nodesById.has(edge.from)) nodesById.set(edge.from, collectionNode(edge, "from"));
      if (!nodesById.has(edge.to)) nodesById.set(edge.to, collectionNode(edge, "to"));
    }
    return { nodes: [...nodesById.values()], edges: [...edgesById.values()] };
  }, [collection, graph, hits, q]);
  filteredRef.current = filtered;

  const collectedNodeIds = useMemo(() => new Set(collection.filter((item) => item.kind === "node").map((item) => (item as Extract<GraphCollectionItem, { kind: "node" }>).node.id)), [collection]);
  const collectedEdgeIds = useMemo(() => new Set(collection.filter((item) => item.kind === "edge").map((item) => (item as Extract<GraphCollectionItem, { kind: "edge" }>).edge.id)), [collection]);

  const nodeVis = (node: GraphNode, position?: { x: number; y: number }) => ({
    id: node.id,
    label: captionOf(node),
    color: {
      background: node.color || "#a5abb6",
      border: collectedNodeIds.has(node.id) ? "#8ab4ff" : node.color || "#a5abb6",
      highlight: { background: node.color || "#a5abb6", border: "#ececec" },
    },
    borderWidth: collectedNodeIds.has(node.id) ? 3 : 0,
    title: `${node.group}: ${captionOf(node)}`,
    ...(position || {}),
  });
  const edgeVis = (edge: GraphEdge) => ({
    id: edge.id,
    from: edge.from,
    to: edge.to,
    label: edge.label,
    title: `${edge.from_name || edge.from} —${edge.label}→ ${edge.to_name || edge.to}`,
    width: collectedEdgeIds.has(edge.id) ? 2.4 : 1,
    color: collectedEdgeIds.has(edge.id) ? { color: "#8ab4ff", highlight: "#ffffff" } : undefined,
  });

  const hasRenderableGraph = filtered.nodes.length > 0;
  useEffect(() => {
    const host = hostRef.current;
    if (!host || !hasRenderableGraph) return;
    layoutPositionsRef.current.clear();
    activeViewRef.current = viewId;
    let cancelled = false;
    let network: Network | null = null;
    let observer: ResizeObserver | null = null;
    const nodes = new DataSet(filteredRef.current.nodes.map((node) => nodeVis(node)));
    const edges = new DataSet(filteredRef.current.edges.map((edge) => edgeVis(edge)));
    nodeDataRef.current = nodes as unknown as MutableDataSet;
    edgeDataRef.current = edges as unknown as MutableDataSet;

    const applySize = (fit: boolean) => {
      if (!network || cancelled) return;
      const width = host.clientWidth;
      const height = host.clientHeight;
      if (width < 16 || height < 16) return;
      network.setSize(`${width}px`, `${height}px`);
      network.redraw();
      if (fit) fitVisibleGraph(network, filteredRef.current.nodes.length);
    };
    const start = () => {
      if (cancelled || network) return;
      network = new Network(host, { nodes, edges }, NETWORK_OPTIONS);
      netRef.current = network;
      network.on("selectNode", (event: { nodes: string[] }) => {
        setSuggestionsOpen(false);
        interactionRef.current?.();
        const node = filteredRef.current.nodes.find((item) => item.id === event.nodes[0]);
        setSelected(node ? { kind: "node", node } : null);
        setInspectorMode("detail");
      });
      network.on("selectEdge", (event: { edges: string[]; nodes: string[] }) => {
        setSuggestionsOpen(false);
        interactionRef.current?.();
        if (event.nodes.length) return;
        const edge = filteredRef.current.edges.find((item) => item.id === event.edges[0]);
        setSelected(edge ? { kind: "edge", edge } : null);
        setInspectorMode("detail");
      });
      network.on("click", () => { setSuggestionsOpen(false); interactionRef.current?.(); });
      network.once("stabilizationIterationsDone", () => {
        if (!network) return;
        const mutableNetwork = network as MutableNetwork;
        mutableNetwork.stopSimulation();
        mutableNetwork.setOptions({ physics: { enabled: false } });
        layoutPositionsRef.current.set(activeViewRef.current, mutableNetwork.getPositions());
        applySize(true);
      });
      applySize(true);
      observer?.disconnect();
      observer = new ResizeObserver(() => applySize(false));
      observer.observe(host);
    };
    if (hasSize(host)) start();
    else {
      observer = new ResizeObserver(() => { if (hasSize(host)) start(); });
      observer.observe(host);
      requestAnimationFrame(() => { if (hasSize(host)) start(); });
    }
    return () => {
      cancelled = true;
      observer?.disconnect();
      network?.destroy();
      if (netRef.current === network) netRef.current = null;
      if (nodeDataRef.current === nodes) nodeDataRef.current = null;
      if (edgeDataRef.current === edges) edgeDataRef.current = null;
    };
  }, [layoutKey, hasRenderableGraph]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    const network = netRef.current;
    const nodes = nodeDataRef.current;
    const edges = edgeDataRef.current;
    if (!network || !nodes || !edges) return;
    const desiredNodeIds = new Set(filtered.nodes.map((node) => node.id));
    const desiredEdgeIds = new Set(filtered.edges.map((edge) => edge.id));
    const currentNodeIds = new Set(nodes.getIds().map(String));
    const currentEdgeIds = new Set(edges.getIds().map(String));
    const newNodes = filtered.nodes.filter((node) => !currentNodeIds.has(node.id));
    const mutableNetwork = network as MutableNetwork;
    const currentPositions = mutableNetwork.getPositions();
    const previousView = activeViewRef.current;
    const previousPositions = layoutPositionsRef.current.get(previousView) || {};
    layoutPositionsRef.current.set(previousView, { ...previousPositions, ...currentPositions });
    const targetPositions = layoutPositionsRef.current.get(viewId) || {};
    const allPositions = layoutPositionsRef.current.get("all") || {};
    const positionFor = (nodeId: string): NodePosition | undefined => (
      targetPositions[nodeId] || allPositions[nodeId] || currentPositions[nodeId]
    );
    const anchor = appendEvent && appendEvent.id !== appendAppliedRef.current
      ? mutableNetwork.getPosition(appendEvent.anchorNodeId)
      : { x: 0, y: 0 };
    if (newNodes.length) {
      const perRing = 12;
      nodes.add(newNodes.map((node, index) => {
        const savedPosition = positionFor(node.id);
        if (savedPosition) return nodeVis(node, savedPosition);
        const ring = Math.floor(index / perRing);
        const angle = (Math.PI * 2 * (index % perRing)) / Math.min(perRing, newNodes.length);
        const radius = 120 + ring * 78;
        return nodeVis(node, { x: anchor.x + Math.cos(angle) * radius, y: anchor.y + Math.sin(angle) * radius });
      }));
    }
    nodes.update(filtered.nodes.map((node) => nodeVis(node, positionFor(node.id))));
    edges.update(filtered.edges.map((edge) => edgeVis(edge)));
    const removedEdges = [...currentEdgeIds].filter((id) => !desiredEdgeIds.has(id));
    const removedNodes = [...currentNodeIds].filter((id) => !desiredNodeIds.has(id));
    if (removedEdges.length) edges.remove(removedEdges);
    if (removedNodes.length) nodes.remove(removedNodes);
    if (appendEvent) appendAppliedRef.current = appendEvent.id;
    activeViewRef.current = viewId;
    network.redraw();
  }, [appendEvent, collectedEdgeIds, collectedNodeIds, filtered, viewId]); // eslint-disable-line react-hooks/exhaustive-deps

  // UNIT tabs reuse the loaded payload and existing static layout. Only the
  // camera must be refitted after the visible node set changes; restarting
  // physics here would make the graph jump and rotate again.
  useEffect(() => {
    const network = netRef.current;
    if (!network || !filtered.nodes.length) return;
    const frame = requestAnimationFrame(() => {
      if (netRef.current === network) fitVisibleGraph(network, filteredRef.current.nodes.length);
    });
    return () => cancelAnimationFrame(frame);
  }, [viewId]);

  useEffect(() => {
    if (!focusEdgeId) return;
    const edge = filtered.edges.find((item) => item.id === focusEdgeId);
    if (!edge) return;
    setSelected({ kind: "edge", edge });
    setInspectorMode("detail");
    const network = netRef.current;
    if (network) { network.selectEdges([edge.id]); network.focus(edge.from, { animation: false, scale: 1.08 }); }
  }, [focusEdgeId, filtered.edges]);

  const legend = useMemo(() => {
    const seen = new Map<string, string>();
    for (const node of filtered.nodes) if (!seen.has(node.group)) seen.set(node.group, node.color);
    return [...seen.entries()];
  }, [filtered.nodes]);

  const pickTriplet = (edge: GraphEdge) => {
    setSelected({ kind: "edge", edge });
    setSuggestionsOpen(false);
    setInspectorMode("detail");
    const network = netRef.current;
    if (!network || !filtered.edges.some((item) => item.id === edge.id)) return;
    network.selectEdges([edge.id]);
    network.focus(edge.from, { animation: false, scale: 1.1 });
  };
  const pickNode = (node: GraphNode) => {
    setSelected({ kind: "node", node });
    setInspectorMode("detail");
    const network = netRef.current;
    if (!network || !filtered.nodes.some((item) => item.id === node.id)) return;
    network.selectNodes([node.id]);
    network.focus(node.id, { animation: false, scale: 1.1 });
  };
  const toggleNode = (node: GraphNode) => {
    const key = nodeKey(node);
    setCollection((items) => items.some((item) => item.key === key) ? items.filter((item) => item.key !== key) : [...items, { kind: "node", key, node }]);
  };
  const toggleEdge = (edge: GraphEdge) => {
    const key = edgeKey(edge);
    setCollection((items) => items.some((item) => item.key === key) ? items.filter((item) => item.key !== key) : [...items, { kind: "edge", key, edge }]);
  };

  const selectedExpansion = selected?.kind === "node" ? expansionByNode[`${selected.node.id}:${expandDirection}`] : undefined;
  const selectedLoadedEdges = selected?.kind === "node" ? graph.edges.filter((edge) => edge.from === selected.node.id || edge.to === selected.node.id).length : 0;

  return (
    <div className="graph-stage">
      {!hideSearch && (
        <div className="graph-local-search">
          <input
            className="graph-search"
            value={query}
            onChange={(event) => { setQuery(event.target.value); setSuggestionsOpen(true); }}
            onFocus={() => setSuggestionsOpen(true)}
            placeholder="Например: kefir, Lactobacillus, GABA, 37 °C"
            aria-label="Поиск на схеме по английским именам и evidence"
            aria-describedby="graph-search-lang-hint"
          />
          <p id="graph-search-lang-hint" className="search-lang-hint">Имена в базе английские.</p>
          {suggestionsOpen && q && hits.length > 0 && (
            <div className="edge-suggestions" role="listbox" aria-label="Связи на схеме">
              {hits.slice(0, 12).map((edge) => (
                <button key={edge.id} type="button" onClick={() => pickTriplet(edge)}>
                  <strong>{tripletCaption(edge)}</strong>
                  {Boolean(edge.properties?.evidence) && <span>{String(edge.properties.evidence)}</span>}
                </button>
              ))}
            </div>
          )}
        </div>
      )}
      {!filtered.nodes.length && !collection.length ? <div className="graph-empty">{emptyHint || "Ничего не найдено"}</div> : (
        <div ref={bodyRef} className="graph-body" style={{ gridTemplateRows: `minmax(180px, 1fr) 7px ${inspectorHeight}px` }}>
          <div className="graph-canvas-wrap">
            {filtered.nodes.length ? <>
              <div ref={hostRef} className="graph-canvas" />
              <div className="graph-legend">{legend.map(([group, color]) => <span key={group} className="graph-legend-item"><i style={{ background: color }} />{group}</span>)}</div>
            </> : <div className="graph-empty">{emptyHint}</div>}
          </div>
          <div
            className="graph-resize-handle"
            role="separator"
            tabIndex={0}
            aria-label="Изменить высоту панели деталей"
            aria-orientation="horizontal"
            aria-valuenow={inspectorHeight}
            onKeyDown={(event) => {
              if (event.key === "ArrowUp") { event.preventDefault(); resizeInspector(inspectorHeight + 20); }
              if (event.key === "ArrowDown") { event.preventDefault(); resizeInspector(inspectorHeight - 20); }
            }}
            onPointerDown={(event) => {
              event.preventDefault();
              const startY = event.clientY;
              const startHeight = inspectorHeight;
              const move = (nextEvent: PointerEvent) => resizeInspector(startHeight + startY - nextEvent.clientY);
              const stop = () => {
                window.removeEventListener("pointermove", move);
                window.removeEventListener("pointerup", stop);
                window.removeEventListener("pointercancel", stop);
                document.body.style.cursor = "";
                document.body.style.userSelect = "";
              };
              document.body.style.cursor = "ns-resize";
              document.body.style.userSelect = "none";
              window.addEventListener("pointermove", move);
              window.addEventListener("pointerup", stop);
              window.addEventListener("pointercancel", stop);
            }}
          ><span /></div>
          <aside ref={inspectorRef} className="graph-inspector">
            <div className="inspector-tabs">
              <button type="button" className={inspectorMode === "detail" ? "is-on" : ""} onClick={() => setInspectorMode("detail")}>Детали</button>
              <button type="button" className={inspectorMode === "collection" ? "is-on" : ""} onClick={() => setInspectorMode("collection")}>Подборка {collection.length || ""}</button>
            </div>
            {inspectorMode === "detail" ? <>
              {!selected && <p className="muted">Выберите сущность или связь. Найденный контекст останется на схеме.</p>}
              {selected?.kind === "node" && <div>
                <p className="inspector-kicker">{selected.node.group}</p>
                <h3>{captionOf(selected.node)}</h3>
                <p className="muted">Загружено связей: {selectedExpansion?.loaded ?? selectedLoadedEdges}{selectedExpansion?.total ? ` из ${selectedExpansion.total}` : ""}.</p>
                <div className="inspector-actions"><button type="button" className="primary-btn" onClick={() => toggleNode(selected.node)}>{collectedNodeIds.has(selected.node.id) ? "Убрать из подборки" : "В подборку"}</button></div>
                {onExpandNode && graph.nodes.some((item) => item.id === selected.node.id) && <div className="node-expansion-controls">
                  <div className="direction-toggle" aria-label="Направление связей">
                    {(["all", "incoming", "outgoing"] as const).map((direction) => <button key={direction} type="button" className={expandDirection === direction ? "is-on" : ""} onClick={() => setExpandDirection(direction)}>{direction === "all" ? "Все" : direction === "incoming" ? "Входящие" : "Исходящие"}</button>)}
                  </div>
                  <button type="button" className="graph-expand-btn" disabled={selectedExpansion?.busy || selectedExpansion?.hasMore === false} onClick={() => onExpandNode(selected.node.id, expandDirection)}>
                    {selectedExpansion?.busy ? "Загрузка…" : selectedExpansion?.hasMore === false ? "Все связи раскрыты" : selectedExpansion?.loaded ? "Раскрыть ещё" : "Раскрыть связи"}
                  </button>
                </div>}
              </div>}
              {selected?.kind === "edge" && <EdgeCard edge={selected.edge} inCollection={collectedEdgeIds.has(selected.edge.id)} onToggleCollection={() => toggleEdge(selected.edge)} />}
            </> : <div className="evidence-collection">
              {!collection.length && <p className="muted">Добавляйте сущности и доказательные связи, чтобы собрать контекст разработки.</p>}
              {collection.some((item) => item.kind === "node") && <p className="collection-section-title">Сущности</p>}
              {collection.filter((item): item is Extract<GraphCollectionItem, { kind: "node" }> => item.kind === "node").map((item) => <button key={item.key} type="button" onClick={() => pickNode(item.node)}><strong>{captionOf(item.node)}</strong><span>{item.node.group}</span></button>)}
              {collection.some((item) => item.kind === "edge") && <p className="collection-section-title">Факты</p>}
              {collection.filter((item): item is Extract<GraphCollectionItem, { kind: "edge" }> => item.kind === "edge").map((item) => <button key={item.key} type="button" onClick={() => pickTriplet(item.edge)}><strong>{tripletCaption(item.edge)}</strong><span>{String(item.edge.properties?.source_file || "Источник не указан")}</span></button>)}
              {collection.length > 0 && <div className="collection-footer">
                {onUseCollection && <button type="button" className="primary-btn" onClick={() => onUseCollection(collection)}>Вставить в чат</button>}
                <button type="button" className="collection-clear" onClick={() => setCollection([])}>Очистить подборку</button>
              </div>}
            </div>}
          </aside>
        </div>
      )}
    </div>
  );
}

function EdgeCard({ edge, inCollection, onToggleCollection }: { edge: GraphEdge; inCollection: boolean; onToggleCollection: () => void }) {
  const properties = edge.properties || {};
  const evidence = String(properties.evidence || "");
  const source = String(properties.source_file || "");
  const confidence = properties.confidence;
  return <div>
    <p className="inspector-kicker">{edge.label}</p>
    <h3>{edge.from_name || edge.from} → {edge.to_name || edge.to}</h3>
    {evidence && <blockquote className="inspector-quote">{evidence}</blockquote>}
    <dl className="inspector-meta">{source && <><dt>Источник</dt><dd>{source}</dd></>}{confidence != null && confidence !== "" && <><dt>Уверенность экстракции</dt><dd>{Number(confidence).toFixed(2)}</dd></>}</dl>
    <div className="inspector-actions">
      <button type="button" className="primary-btn" onClick={onToggleCollection}>{inCollection ? "Убрать из подборки" : "В подборку"}</button>
      {evidence && <button type="button" className="ghost-btn" onClick={() => copyText(evidence)}>Копировать данные</button>}
    </div>
  </div>;
}

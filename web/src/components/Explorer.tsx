import { useCallback, useEffect, useMemo, useState } from "react";
import type { GraphCollectionItem, GraphFacetItem, GraphFacets, GraphFilters, GraphPayload } from "../types";
import { fetchGraphExpand, fetchGraphExplore, fetchGraphFacets, fetchGraphSchema } from "../api";
import { GraphCanvas, type GraphAppendEvent, type GraphExpansionUi } from "./GraphCanvas";
import { tripletCaption } from "../format";

const DEFAULT_LIMIT = 100;
const MIN_LIMIT = 1;
const MAX_LIMIT = 5000;
const EMPTY_FILTERS: GraphFilters = {
  node_labels: [],
  relationship_types: [],
  sources: [],
  min_confidence: null,
};

function hasFilters(filters: GraphFilters): boolean {
  return Boolean(filters.node_labels.length || filters.relationship_types.length || filters.sources.length || filters.min_confidence != null);
}

function toggleValue(values: string[], value: string): string[] {
  return values.includes(value) ? values.filter((item) => item !== value) : [...values, value];
}

function mergeFacetValues(values: string[], items: GraphFacetItem[]): GraphFacetItem[] {
  const counts = new Map(items.map((item) => [item.value, item.count]));
  return [...new Set([...values, ...items.map((item) => item.value)])]
    .map((value) => ({ value, count: counts.get(value) || 0 }))
    .sort((left, right) => right.count - left.count || left.value.localeCompare(right.value));
}

function collectionDraft(items: GraphCollectionItem[]): string {
  const nodes = items.filter((item): item is Extract<GraphCollectionItem, { kind: "node" }> => item.kind === "node");
  const edges = items.filter((item): item is Extract<GraphCollectionItem, { kind: "edge" }> => item.kind === "edge");
  const lines = ["Используй эту подборку из базы знаний как данные для ответа."];
  if (nodes.length) {
    lines.push("", "Сущности:");
    for (const item of nodes) lines.push(`- ${item.node.group}: ${item.node.caption || item.node.label || item.node.id}`);
  }
  if (edges.length) {
    lines.push("", "Доказательные факты:");
    for (const item of edges) {
      const edge = item.edge;
      lines.push(`- ${tripletCaption(edge)}`);
      const evidence = String(edge.properties?.evidence || "").trim();
      const source = String(edge.properties?.source_file || "").trim();
      const confidence = edge.properties?.confidence;
      if (evidence) lines.push(`  Evidence: ${evidence}`);
      if (source) lines.push(`  Источник: ${source}`);
      if (confidence != null && confidence !== "") lines.push(`  Уверенность экстракции: ${Number(confidence).toFixed(2)}`);
    }
  }
  return lines.join("\n");
}

export function Explorer({ onUseCollection }: { onUseCollection?: (draft: string) => void }) {
  const [q, setQ] = useState("");
  const [limitText, setLimitText] = useState(String(DEFAULT_LIMIT));
  const [filters, setFilters] = useState<GraphFilters>(EMPTY_FILTERS);
  const [payload, setPayload] = useState<GraphPayload | null>(null);
  const [facets, setFacets] = useState<GraphFacets | null>(null);
  const [schema, setSchema] = useState<{ nodeLabels: string[]; relationshipTypes: string[] }>({ nodeLabels: [], relationshipTypes: [] });
  const [filtersOpen, setFiltersOpen] = useState(false);
  const [sourceQuery, setSourceQuery] = useState("");
  const [sourceBusy, setSourceBusy] = useState(false);
  const [focusEdgeId, setFocusEdgeId] = useState("");
  const [suggestionsOpen, setSuggestionsOpen] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [layoutRevision, setLayoutRevision] = useState(0);
  const [appendEvent, setAppendEvent] = useState<GraphAppendEvent | null>(null);
  const [expansionByNode, setExpansionByNode] = useState<Record<string, GraphExpansionUi>>({});

  const parsedLimit = Number(limitText);
  const limit = Number.isInteger(parsedLimit) && parsedLimit >= MIN_LIMIT && parsedLimit <= MAX_LIMIT ? parsedLimit : null;
  const activeFilterCount = filters.node_labels.length + filters.relationship_types.length + filters.sources.length + Number(filters.min_confidence != null);

  useEffect(() => { void fetchGraphSchema().then(setSchema).catch(() => undefined); }, []);

  useEffect(() => {
    let alive = true;
    const timer = window.setTimeout(() => {
      void fetchGraphFacets(q, filters, sourceQuery).then((next) => {
        if (alive) setFacets(next);
      }).catch((err: Error) => { if (alive) setError(err.message); });
    }, 280);
    return () => { alive = false; window.clearTimeout(timer); };
  }, [filters, q, sourceQuery]);

  useEffect(() => {
    if (limit == null) return undefined;
    let alive = true;
    const timer = window.setTimeout(() => {
      if (!q.trim() && !hasFilters(filters)) {
        setPayload(null);
        setFocusEdgeId("");
        setExpansionByNode({});
        setAppendEvent(null);
        setLayoutRevision((value) => value + 1);
        setError("");
        return;
      }
      setBusy(true);
      setError("");
      void fetchGraphExplore(q, limit, "all", "", filters).then((next) => {
        if (!alive) return;
        setPayload(next);
        setFocusEdgeId("");
        setExpansionByNode({});
        setAppendEvent(null);
        setLayoutRevision((value) => value + 1);
      }).catch((err: Error) => { if (alive) setError(err.message); })
        .finally(() => { if (alive) setBusy(false); });
    }, 280);
    return () => { alive = false; window.clearTimeout(timer); };
  }, [filters, limit, q]);

  const suggestions = payload?.all.edges.slice(0, 12) || [];
  const closeSuggestions = useCallback(() => setSuggestionsOpen(false), []);

  const expandNode = useCallback((nodeId: string, direction: "all" | "incoming" | "outgoing") => {
    const currentPayload = payload;
    const expansionKey = `${nodeId}:${direction}`;
    if (!currentPayload || expansionByNode[expansionKey]?.busy) return;
    const incident = currentPayload.all.edges.filter((edge) => edge.from === nodeId || edge.to === nodeId);
    const excluded = incident.map((edge) => edge.id).slice(0, 5000);
    setExpansionByNode((current) => ({
      ...current,
      [expansionKey]: { loaded: incident.length, total: current[expansionKey]?.total || 0, hasMore: true, busy: true },
    }));
    void fetchGraphExpand(nodeId, limit ?? DEFAULT_LIMIT, excluded, direction, filters).then((next) => {
      const existingNodeIds = new Set(currentPayload.all.nodes.map((node) => node.id));
      const existingEdgeIds = new Set(currentPayload.all.edges.map((edge) => edge.id));
      const addedNodes = next.all.nodes.filter((node) => !existingNodeIds.has(node.id));
      const addedEdges = next.all.edges.filter((edge) => !existingEdgeIds.has(edge.id));
      setPayload((current) => {
        if (!current) return next;
        const nodes = new Map([...current.all.nodes, ...next.all.nodes].map((node) => [node.id, node]));
        const edges = new Map([...current.all.edges, ...next.all.edges].map((edge) => [edge.id, edge]));
        return { ...current, all: { nodes: [...nodes.values()], edges: [...edges.values()] } };
      });
      const expansion = next.expansion;
      setAppendEvent({ id: `${Date.now()}:${nodeId}`, anchorNodeId: nodeId, nodes: addedNodes, edges: addedEdges });
      setExpansionByNode((current) => ({
        ...current,
        [expansionKey]: {
          loaded: incident.length + addedEdges.length,
          total: expansion?.totalMatching || incident.length + addedEdges.length,
          hasMore: Boolean(expansion?.hasMore),
          busy: false,
        },
      }));
    }).catch((err: Error) => {
      setError(err.message);
      setExpansionByNode((current) => ({
        ...current,
        [expansionKey]: { ...(current[expansionKey] || { loaded: incident.length, total: 0, hasMore: true }), busy: false },
      }));
    });
  }, [expansionByNode, filters, limit, payload]);

  const normalizeLimit = () => {
    const raw = limitText.trim();
    const next = Number(raw);
    setLimitText(String(raw && Number.isInteger(next) ? Math.max(MIN_LIMIT, Math.min(next, MAX_LIMIT)) : DEFAULT_LIMIT));
  };

  const nodeFacets = useMemo(() => mergeFacetValues(schema.nodeLabels, facets?.nodeLabels || []), [facets?.nodeLabels, schema.nodeLabels]);
  const relationshipFacets = useMemo(() => mergeFacetValues(schema.relationshipTypes, facets?.relationshipTypes || []), [facets?.relationshipTypes, schema.relationshipTypes]);

  const loadMoreSources = () => {
    const cursor = facets?.sources.nextCursor;
    if (!cursor || sourceBusy) return;
    setSourceBusy(true);
    void fetchGraphFacets(q, filters, sourceQuery, cursor).then((next) => {
      setFacets((current) => current ? {
        ...current,
        sources: {
          ...next.sources,
          items: mergeFacetValues([], [...current.sources.items, ...next.sources.items]),
        },
      } : next);
    }).catch((err: Error) => setError(err.message)).finally(() => setSourceBusy(false));
  };

  return <div className="explorer">
    <header className="explorer-bar">
      <div className="edge-search-wrap">
        <input
          value={q}
          onChange={(event) => { setQ(event.target.value); setSuggestionsOpen(true); }}
          onFocus={() => setSuggestionsOpen(true)}
          onKeyDown={(event) => { if (event.key === "Escape") setSuggestionsOpen(false); }}
          placeholder="Например: kefir, Lactobacillus, GABA, 37 °C"
          aria-label="Поиск по всей базе по английским именам и evidence"
          aria-describedby="explorer-search-lang-hint"
          autoFocus
        />
        <p id="explorer-search-lang-hint" className="search-lang-hint">Имена в базе английские.</p>
        {suggestionsOpen && q.trim() && suggestions.length > 0 && <div className="edge-suggestions" role="listbox" aria-label="Найденные связи">
          {suggestions.map((edge) => <button key={edge.id} type="button" onClick={() => { setFocusEdgeId(edge.id); setSuggestionsOpen(false); }}><strong>{tripletCaption(edge)}</strong>{Boolean(edge.properties?.evidence) && <span>{String(edge.properties.evidence)}</span>}</button>)}
        </div>}
      </div>
      <button type="button" className={`explorer-filter-btn ${filtersOpen ? "is-on" : ""}`} onClick={() => setFiltersOpen((value) => !value)} aria-expanded={filtersOpen}>Фильтры{activeFilterCount ? ` ${activeFilterCount}` : ""}</button>
      <label className="explorer-limit" title={`Сколько связей показать, от ${MIN_LIMIT} до ${MAX_LIMIT}`}><span>Связей</span><input type="number" min={MIN_LIMIT} max={MAX_LIMIT} step={1} inputMode="numeric" value={limitText} onChange={(event) => setLimitText(event.target.value)} onBlur={normalizeLimit} aria-label="Сколько связей показать" /></label>
      {busy && <span className="search-spinner">поиск…</span>}
    </header>
    {activeFilterCount > 0 && <div className="active-filters">
      {filters.node_labels.map((value) => <button key={`node:${value}`} type="button" onClick={() => setFilters((current) => ({ ...current, node_labels: toggleValue(current.node_labels, value) }))}>{value} ×</button>)}
      {filters.relationship_types.map((value) => <button key={`rel:${value}`} type="button" onClick={() => setFilters((current) => ({ ...current, relationship_types: toggleValue(current.relationship_types, value) }))}>{value} ×</button>)}
      {filters.sources.map((value) => <button key={`source:${value}`} type="button" onClick={() => setFilters((current) => ({ ...current, sources: toggleValue(current.sources, value) }))}>{value} ×</button>)}
      {filters.min_confidence != null && <button type="button" onClick={() => setFilters((current) => ({ ...current, min_confidence: null }))}>уверенность экстракции ≥ {filters.min_confidence} ×</button>}
      <button type="button" className="active-filters-clear" onClick={() => setFilters(EMPTY_FILTERS)}>Сбросить всё</button>
    </div>}
    {error && <p className="explorer-status is-error">{error}</p>}
    {(payload || facets) && !error && <p className="explorer-status">Показано связей: {payload?.all.edges.length || 0} из {facets?.matchingRelationships || 0} · сущностей: {facets?.matchingNodes || 0}</p>}
    <div className="explorer-content">
      {filtersOpen && <aside className="graph-filter-panel">
        <div className="filter-panel-head"><strong>Фильтры данных</strong><button type="button" onClick={() => setFiltersOpen(false)} aria-label="Закрыть фильтры">×</button></div>
        <FacetSection title="Тип сущности" items={nodeFacets} selected={filters.node_labels} onToggle={(value) => setFilters((current) => ({ ...current, node_labels: toggleValue(current.node_labels, value) }))} />
        <FacetSection title="Тип отношения" items={relationshipFacets} selected={filters.relationship_types} onToggle={(value) => setFilters((current) => ({ ...current, relationship_types: toggleValue(current.relationship_types, value) }))} />
        <section className="facet-section"><h3>Источник</h3><input className="facet-search" value={sourceQuery} onChange={(event) => setSourceQuery(event.target.value)} placeholder="Найти статью" />
          <FacetList items={facets?.sources.items || []} selected={filters.sources} onToggle={(value) => setFilters((current) => ({ ...current, sources: toggleValue(current.sources, value) }))} />
          {facets?.sources.hasMore && <button type="button" className="facet-more" disabled={sourceBusy} onClick={loadMoreSources}>{sourceBusy ? "Загрузка…" : "Показать ещё"}</button>}
        </section>
        <section className="facet-section"><h3>Качество данных</h3>
          <p className="facet-note">В выборку попадают только связи с evidence.</p>
          <label className="confidence-field"><span>уверенность экстракции ≥</span><input type="number" min="0" max="1" step="0.05" value={filters.min_confidence ?? ""} placeholder="без ограничения" aria-label="уверенность экстракции ≥" onChange={(event) => { const value = event.target.value; setFilters((current) => ({ ...current, min_confidence: value === "" ? null : Math.max(0, Math.min(1, Number(value))) })); }} /></label>
        </section>
      </aside>}
      <GraphCanvas
        payload={payload}
        viewId="all"
        emptyHint={q.trim() || hasFilters(filters) ? "По всей базе не найдено подходящих данных." : "Наберите английское имя, фрагмент evidence или выберите фильтр."}
        hideSearch
        focusEdgeId={focusEdgeId}
        layoutKey={`explorer:${layoutRevision}`}
        appendEvent={appendEvent}
        expansionByNode={expansionByNode}
        onCanvasInteraction={closeSuggestions}
        onExpandNode={expandNode}
        onUseCollection={onUseCollection ? (items) => onUseCollection(collectionDraft(items)) : undefined}
      />
    </div>
  </div>;
}

function FacetSection({ title, items, selected, onToggle }: { title: string; items: GraphFacetItem[]; selected: string[]; onToggle: (value: string) => void }) {
  return <section className="facet-section"><h3>{title}</h3><FacetList items={items} selected={selected} onToggle={onToggle} /></section>;
}

function FacetList({ items, selected, onToggle }: { items: GraphFacetItem[]; selected: string[]; onToggle: (value: string) => void }) {
  return <div className="facet-list">{items.map((item) => <label key={item.value} className={item.count === 0 && !selected.includes(item.value) ? "is-disabled" : ""}><input type="checkbox" checked={selected.includes(item.value)} disabled={item.count === 0 && !selected.includes(item.value)} onChange={() => onToggle(item.value)} /><span title={item.value}>{item.value}</span><b>{item.count.toLocaleString("ru-RU")}</b></label>)}</div>;
}

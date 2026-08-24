import { useEffect, useState } from "react";
import type { GraphPayload } from "../types";
import { fetchGraphExpand, fetchGraphExplore } from "../api";
import { GraphCanvas } from "./GraphCanvas";
import { tripletCaption } from "../format";

export function Explorer() {
  const [q, setQ] = useState("");
  const [payload, setPayload] = useState<GraphPayload | null>(null);
  const [focusEdgeId, setFocusEdgeId] = useState("");
  const [suggestionsOpen, setSuggestionsOpen] = useState(false);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function run(nextQ: string) {
    if (!nextQ.trim()) {
      setPayload(null);
      setFocusEdgeId("");
      setError("");
      return;
    }
    setBusy(true);
    setError("");
    try {
      setPayload(await fetchGraphExplore(nextQ, 100, "all"));
      setFocusEdgeId("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка загрузки");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    const timer = window.setTimeout(() => void run(q), 220);
    return () => window.clearTimeout(timer);
  }, [q]); // eslint-disable-line react-hooks/exhaustive-deps

  const suggestions = payload?.all.edges.slice(0, 12) || [];

  return (
    <div className="explorer">
      <header className="explorer-bar">
        <div className="edge-search-wrap">
          <input
            value={q}
            onChange={(e) => { setQ(e.target.value); setSuggestionsOpen(true); }}
            onFocus={() => setSuggestionsOpen(true)}
            placeholder="Сущность, тип связи, evidence или источник"
            aria-label="Поиск по всей базе"
            autoFocus
          />
          {suggestionsOpen && q.trim() && suggestions.length > 0 && (
            <div className="edge-suggestions" role="listbox" aria-label="Найденные рёбра">
              {suggestions.map((edge) => (
                <button
                  key={edge.id}
                  type="button"
                  onClick={() => { setFocusEdgeId(edge.id); setSuggestionsOpen(false); }}
                >
                  <strong>{tripletCaption(edge)}</strong>
                  {Boolean(edge.properties?.evidence) && <span>{String(edge.properties.evidence)}</span>}
                </button>
              ))}
            </div>
          )}
        </div>
        {busy && <span className="search-spinner">поиск…</span>}
      </header>
      {error && <p className="explorer-status is-error">{error}</p>}
      <GraphCanvas
        payload={payload}
        viewId="all"
        emptyHint={q.trim() ? "По всей базе не найдено подходящих рёбер." : "Начните вводить запрос — поиск пройдёт по всему corpus."}
        hideSearch
        focusEdgeId={focusEdgeId}
        onExpandNode={(nodeId) => {
          void fetchGraphExpand(nodeId, 100).then((next) => {
            setPayload((current) => {
              if (!current) return next;
              const nodes = new Map([...current.all.nodes, ...next.all.nodes].map((node) => [node.id, node]));
              const edges = new Map([...current.all.edges, ...next.all.edges].map((edge) => [edge.id, edge]));
              return { ...current, all: { nodes: [...nodes.values()], edges: [...edges.values()] } };
            });
          }).catch((err: Error) => setError(err.message));
        }}
      />
    </div>
  );
}

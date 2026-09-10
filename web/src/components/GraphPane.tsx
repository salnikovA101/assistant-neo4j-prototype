import { useEffect, useState, type ReactNode } from "react";
import type { GraphPayload } from "../types";
import { fetchCheckpointGraph } from "../api";
import { GraphCanvas } from "./GraphCanvas";
import { IconClose } from "./Icons";
import { chainLabel } from "../uiLabels";

export function GraphPane({
  checkpointId,
  onClose,
  onBackToMap,
  onOpenStep,
  embedded = false,
}: {
  checkpointId: string;
  onClose: () => void;
  onBackToMap?: () => void;
  onOpenStep?: (stepId: string, branchId?: string) => void;
  embedded?: boolean;
}) {
  const [payload, setPayload] = useState<GraphPayload | null>(null);
  const [viewId, setViewId] = useState<string | "all">("all");
  const [error, setError] = useState("");
  useEffect(() => {
    setViewId("all");
  }, [checkpointId]);

  useEffect(() => {
    let alive = true;
    setPayload(null);
    setError("");
    fetchCheckpointGraph(checkpointId, "mode_default")
      .then((data) => {
        if (!alive) return;
        setPayload(data);
        setViewId("all");
      })
      .catch((err: Error) => {
        if (alive) setError(err.message);
      });
    return () => {
      alive = false;
    };
  }, [checkpointId]);

  const views = payload?.views || [];
  const title = payload?.mode === "staged" ? "Данные этого варианта" : "Данные этого ответа";
  const emptyHint = payload?.mode === "auto"
    ? "На этом шаге поиск ещё не выполнялся."
    : "В этом представлении нет связанных данных.";
  const selectedView = viewId === "all" ? null : views.find((view) => view.id === viewId) || null;
  const originGroups = Array.from(views.reduce((groups, view) => {
    const key = view.origin?.step_id || "unknown";
    const current = groups.get(key) || { origin: view.origin, count: 0 };
    current.count += 1;
    groups.set(key, current);
    return groups;
  }, new Map<string, { origin: (typeof views)[number]["origin"]; count: number }>()).values());

  const viewIndex = viewId === "all" ? 0 : views.findIndex((view) => view.id === viewId) + 1;
  const moveView = (offset: number) => {
    const next = Math.max(0, Math.min(views.length, viewIndex + offset));
    setViewId(next === 0 ? "all" : views[next - 1].id);
  };
  const toolbar = (search?: ReactNode) => <>
      <header className="graph-pane-bar">
        {embedded && search}
        <div className="graph-view-picker">
          {!embedded && <span>{title}</span>}
          {embedded && <button type="button" className="graph-chain-arrow" aria-label="Предыдущая цепочка" title="Предыдущая цепочка" disabled={!payload || viewIndex === 0} onClick={() => moveView(-1)}>←</button>}
          <select aria-label="Представление данных" value={viewId} onChange={(event) => setViewId(event.target.value)}>
            <option value="all">Все цепочки · {views.length}</option>
            {views.map((view, index) => <option key={view.id} value={view.id}>{embedded ? `Цепочка ${index + 1} из ${views.length}` : chainLabel(view.label, view.unit_no)}{view.origin?.step_no ? ` · вопрос ${view.origin.step_no}` : ""}{view.is_new && payload?.mode === "staged" ? " · новая" : ""}</option>)}
          </select>
          {embedded && <button type="button" className="graph-chain-arrow" aria-label="Следующая цепочка" title="Следующая цепочка" disabled={!payload || viewIndex >= views.length} onClick={() => moveView(1)}>→</button>}
        </div>
        {!embedded && <button type="button" className="icon-btn" onClick={onClose} aria-label="Скрыть данные"><IconClose /></button>}
      </header>
      {payload && views.length > 0 && (
        <details className="graph-origins" key={viewId}><summary>Откуда эти данные</summary>
          {selectedView ? <div><p>{selectedView.origin?.question || "Исходный вопрос не определён"}</p>{selectedView.origin?.step_id && <button type="button" className="ghost-btn" onClick={() => onOpenStep?.(selectedView.origin!.step_id!, selectedView.origin?.branch_id || undefined)}>Перейти к вопросу {selectedView.origin?.step_no || ""}</button>}</div> : originGroups.map((group, index) => <button className="graph-origin-link" key={group.origin?.step_id || index} type="button" disabled={!group.origin?.step_id} onClick={() => group.origin?.step_id && onOpenStep?.(group.origin.step_id, group.origin.branch_id || undefined)}><strong>Вопрос {group.origin?.step_no || "—"} · {group.count} цепочек</strong><span>{group.origin?.question || "Исходный вопрос не определён"}</span></button>)}
        </details>
      )}
  </>;

  return (
    <section className={`graph-pane ${embedded ? "is-embedded" : ""}`}>
      {!embedded && toolbar()}
      {!embedded && onBackToMap && (
        <button type="button" className="graph-back-map" onClick={onBackToMap}>← К карте хода</button>
      )}
      {error && <p className="explorer-status is-error">{error}</p>}
      {!payload && !error && <p className="explorer-status">Загрузка данных…</p>}
      <GraphCanvas
        payload={payload}
        viewId={viewId}
        emptyHint={emptyHint}
        embeddedMode={embedded}
        renderToolbar={embedded ? toolbar : undefined}
      />
    </section>
  );
}

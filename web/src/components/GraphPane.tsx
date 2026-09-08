import { useEffect, useState } from "react";
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

  return (
    <section className={`graph-pane ${embedded ? "is-embedded" : ""}`}>
      <header className="graph-pane-bar">
        <label className="graph-view-picker"><span>{title}</span>
          <select aria-label="Представление данных" value={viewId} onChange={(event) => setViewId(event.target.value)}>
            <option value="all">Все цепочки · {views.length}</option>
            {views.map((view) => <option key={view.id} value={view.id}>{chainLabel(view.label, view.unit_no)}{view.origin?.step_no ? ` · вопрос ${view.origin.step_no}` : ""}{view.is_new && payload?.mode === "staged" ? " · новая" : ""}</option>)}
          </select>
        </label>
        {!embedded && <button type="button" className="icon-btn" onClick={onClose} aria-label="Скрыть данные"><IconClose /></button>}
      </header>
      {payload && views.length > 0 && (
        <details className="graph-origins" key={viewId}><summary>Откуда эти данные</summary>
          {selectedView ? <div><p>{selectedView.origin?.question || "Исходный вопрос не определён"}</p>{selectedView.origin?.step_id && <button type="button" className="ghost-btn" onClick={() => onOpenStep?.(selectedView.origin!.step_id!, selectedView.origin?.branch_id || undefined)}>Перейти к вопросу {selectedView.origin?.step_no || ""}</button>}</div> : originGroups.map((group, index) => <button className="graph-origin-link" key={group.origin?.step_id || index} type="button" disabled={!group.origin?.step_id} onClick={() => group.origin?.step_id && onOpenStep?.(group.origin.step_id, group.origin.branch_id || undefined)}><strong>Вопрос {group.origin?.step_no || "—"} · {group.count} цепочек</strong><span>{group.origin?.question || "Исходный вопрос не определён"}</span></button>)}
        </details>
      )}
      {!embedded && onBackToMap && (
        <button type="button" className="graph-back-map" onClick={onBackToMap}>← К карте хода</button>
      )}
      {error && <p className="explorer-status is-error">{error}</p>}
      {!payload && !error && <p className="explorer-status">Загрузка данных…</p>}
      <GraphCanvas
        payload={payload}
        viewId={viewId}
        emptyHint={emptyHint}
      />
    </section>
  );
}

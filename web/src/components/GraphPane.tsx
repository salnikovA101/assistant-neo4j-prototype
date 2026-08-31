import { useEffect, useState } from "react";
import type { GraphPayload } from "../types";
import { fetchCheckpointGraph } from "../api";
import { GraphCanvas } from "./GraphCanvas";
import { IconClose } from "./Icons";

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
  const title = payload?.mode === "staged" ? "Факты этого варианта" : "Факты этого ответа";
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
        <div className="panel-title">
          <strong>{title}</strong>
          <span>{views.length ? `${views.length} ${views.length === 1 ? "цепочка" : "цепочек"}` : "связи этого шага"}</span>
        </div>
        <div className="chain-nav" aria-label="Представление фактов">
          <button
            type="button"
            className={viewId === "all" ? "is-on" : ""}
            onClick={() => setViewId("all")}
          >
            Все
          </button>
          {views.map((view) => (
            <button
              key={view.id}
              type="button"
              className={`unit-tab ${viewId === view.id ? "is-on" : ""} ${view.is_new && payload?.mode === "staged" ? "has-new" : ""}`}
              onClick={() => setViewId(view.id)}
              title={view.is_new && payload?.mode === "staged" ? "Новая цепочка этого шага" : view.label}
              aria-label={view.is_new && payload?.mode === "staged" ? `${view.label}, новое` : view.label}
            >
              <span>{view.label}</span>
              <small>{view.origin?.step_no ? `из вопроса ${view.origin.step_no}` : "источник не определён"}</small>
              {view.is_new && payload?.mode === "staged" && <span className="unit-new">новое</span>}
            </button>
          ))}
        </div>
        {!embedded && <button type="button" className="icon-btn" onClick={onClose} aria-label="Скрыть факты"><IconClose /></button>}
      </header>
      {payload && views.length > 0 && (
        <div className="graph-origin-bar">
          {selectedView ? (
            <>
              <span><b>Вопрос {selectedView.origin?.step_no || "—"}:</b> {selectedView.origin?.question || "Исходный вопрос не определён"}</span>
              {selectedView.origin?.step_id && (
                <button type="button" onClick={() => onOpenStep?.(selectedView.origin!.step_id!, selectedView.origin?.branch_id || undefined)}>Перейти к шагу</button>
              )}
            </>
          ) : (
            <>
              <span><b>{views.length} {views.length === 1 ? "цепочка" : "цепочек"}</b> из {originGroups.length} {originGroups.length === 1 ? "вопроса" : "вопросов"}</span>
              <div>
                {originGroups.map((group, index) => (
                  <button
                    key={group.origin?.step_id || `unknown-${index}`}
                    type="button"
                    disabled={!group.origin?.step_id}
                    title={group.origin?.question || "Исходный вопрос не определён"}
                    onClick={() => group.origin?.step_id && onOpenStep?.(group.origin.step_id, group.origin.branch_id || undefined)}
                  >
                    Вопрос {group.origin?.step_no || "—"} · {group.count}
                  </button>
                ))}
              </div>
            </>
          )}
        </div>
      )}
      {!embedded && onBackToMap && (
        <button type="button" className="graph-back-map" onClick={onBackToMap}>← К карте хода</button>
      )}
      {error && <p className="explorer-status is-error">{error}</p>}
      {!payload && !error && <p className="explorer-status">Загрузка фактов…</p>}
      <GraphCanvas
        payload={payload}
        viewId={viewId}
        emptyHint={emptyHint}
      />
    </section>
  );
}

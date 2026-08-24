import { useEffect, useState } from "react";
import type { GraphPayload } from "../types";
import { fetchCheckpointGraph, fetchGraphViz } from "../api";
import { GraphCanvas } from "./GraphCanvas";
import { IconClose } from "./Icons";

export function GraphPane({
  runId,
  checkpointId,
  onClose,
}: {
  runId?: string;
  checkpointId?: string;
  onClose: () => void;
}) {
  const [payload, setPayload] = useState<GraphPayload | null>(null);
  const [viewId, setViewId] = useState<string | "all">("all");
  const [error, setError] = useState("");
  const [scope, setScope] = useState<"context" | "new_in_answer" | "all_branches">("context");

  useEffect(() => {
    let alive = true;
    setPayload(null);
    setError("");
    const load = checkpointId
      ? fetchCheckpointGraph(checkpointId, scope)
      : runId
        ? fetchGraphViz(runId)
        : Promise.reject(new Error("Checkpoint не выбран"));
    load
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
  }, [checkpointId, runId, scope]);

  const views = payload?.views || [];

  return (
    <section className="graph-pane">
      <header className="graph-pane-bar">
        <div className="panel-title">
          <strong>{checkpointId ? "Graph Workspace" : "Граф ответа"}</strong>
          <span>{views.length ? `${views.length} UNIT` : "связи checkpoint"}</span>
        </div>
        <div className="chain-nav">
          {checkpointId && (
            <>
              <button type="button" className={scope === "context" ? "is-on" : ""} onClick={() => setScope("context")}>Все</button>
              <button type="button" className={scope === "new_in_answer" ? "is-on" : ""} onClick={() => setScope("new_in_answer")}>Новые</button>
              <button type="button" className={scope === "all_branches" ? "is-on" : ""} onClick={() => setScope("all_branches")}>Все ветки</button>
            </>
          )}
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
              className={viewId === view.id ? "is-on" : ""}
              onClick={() => setViewId(view.id)}
            >
              {view.label}
            </button>
          ))}
        </div>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Скрыть граф">
          <IconClose />
        </button>
      </header>
      {error && <p className="explorer-status is-error">{error}</p>}
      {!payload && !error && <p className="explorer-status">Загрузка графа…</p>}
      <GraphCanvas
        payload={payload}
        viewId={viewId}
        emptyHint="В этом ответе нет цепей графа."
      />
    </section>
  );
}

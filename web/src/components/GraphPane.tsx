import { useEffect, useState } from "react";
import type { GraphPayload } from "../types";
import { fetchGraphViz } from "../api";
import { GraphCanvas } from "./GraphCanvas";
import { IconClose } from "./Icons";

export function GraphPane({
  runId,
  onClose,
}: {
  runId: string;
  onClose: () => void;
}) {
  const [payload, setPayload] = useState<GraphPayload | null>(null);
  const [viewId, setViewId] = useState<string | "all">("all");
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    setPayload(null);
    setError("");
    fetchGraphViz(runId)
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
  }, [runId]);

  const views = payload?.views || [];

  return (
    <section className="graph-pane">
      <header className="graph-pane-bar">
        <div className="panel-title">
          <strong>Граф ответа</strong>
          <span>{views.length ? `${views.length} цепей` : "связи из ответа"}</span>
        </div>
        <div className="chain-nav">
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

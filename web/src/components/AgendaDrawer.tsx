import { useMemo, useState } from "react";
import type { AgendaItem } from "../types";
import { IconClose } from "./Icons";

const STATUS_MARK: Record<AgendaItem["status"], string> = { not_closed: "○", partial: "◐", closed: "✓", deferred: "Ⅱ" };
const STATUS_LABEL: Record<AgendaItem["status"], string> = { not_closed: "Не закрыт", partial: "Закрыт частично", closed: "Закрыт", deferred: "Отложен" };

function directionNo(item: AgendaItem, fallback: number): number {
  const match = item.ref.match(/:(\d+)$/);
  return match ? Number(match[1]) : fallback;
}

export function AgendaDrawer({
  agenda,
  onClose,
  onStatus,
  embedded = false,
  locked = false,
}: {
  agenda: AgendaItem[];
  onClose?: () => void;
  onStatus: (item: AgendaItem, status: AgendaItem["status"]) => Promise<void>;
  embedded?: boolean;
  locked?: boolean;
}) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [closedOpen, setClosedOpen] = useState(false);
  const [deferredOpen, setDeferredOpen] = useState(false);
  const activeItems = useMemo(() => agenda.filter((item) => (item.status === "not_closed" || item.status === "partial")), [agenda]);
  const closedItems = useMemo(() => agenda.filter((item) => item.status === "closed"), [agenda]);
  const deferredItems = useMemo(() => agenda.filter((item) => item.status === "deferred"), [agenda]);
  const partialCount = activeItems.filter((item) => item.status === "partial").length;
  const notClosedCount = activeItems.length - partialCount;

  const renderItem = (item: AgendaItem, index: number) => (
    <article
      key={item.ref}
      className={`agenda-item is-${item.status} ${item.reviewRecommended && (item.status === "not_closed" || item.status === "partial") ? "needs-review" : ""}`}
    >
      <div>
        <strong>Пункт {directionNo(item, index + 1)}</strong>
        <p>{item.text}</p>
        <div className="agenda-meta">
          <span className="agenda-chain-count">{item.unitCount === 1 ? "1 цепочка" : `${item.unitCount} цепочек`}</span>
          <details className="agenda-status-menu" onKeyDown={(event) => {
            if (event.key === "Escape") { event.currentTarget.open = false; event.currentTarget.querySelector("summary")?.focus(); }
          }} onBlur={(event) => {
            if (!event.currentTarget.contains(event.relatedTarget as Node | null)) event.currentTarget.open = false;
          }}>
            <summary className={`agenda-coverage is-${item.status}`} aria-label={`Статус пункта ${directionNo(item, index + 1)}`}>
              {STATUS_MARK[item.status]} {STATUS_LABEL[item.status]}
            </summary>
            <div className="agenda-status-options" role="group" aria-label="Изменить статус">
              {(Object.keys(STATUS_LABEL) as AgendaItem["status"][]).map((status) => (
                <button key={status} type="button" disabled={busy || locked} aria-pressed={item.status === status}
                  onClick={async (event) => {
                    const menu = event.currentTarget.closest("details");
                    if (menu) menu.open = false;
                    if (status === item.status) return;
                    setBusy(true);
                    setError("");
                    try { await onStatus(item, status); }
                    catch { setError("Не удалось изменить статус. Попробуйте ещё раз."); }
                    finally { setBusy(false); }
                  }}>{STATUS_MARK[status]} {STATUS_LABEL[status]}</button>
              ))}
            </div>
          </details>
        </div>
        {item.statusReason && <span className="agenda-reason">{item.statusReason}</span>}
        <span className="agenda-origin">{item.statusOrigin === "assistant" ? "Оценил ассистент" : item.statusOrigin === "user" ? "Изменено вами" : "Исходный статус"}</span>
        {item.reviewRecommended && (item.status === "not_closed" || item.status === "partial") && <em>Пора уточнить</em>}
      </div>
    </article>
  );

  return (
    <section className={`agenda-drawer ${embedded ? "is-embedded" : ""}`}>
      <header>
        <div>
          <strong>Исследовательские вопросы</strong>
          <span>{notClosedCount} не закрыто · {partialCount} частично · {closedItems.length} закрыто · {deferredItems.length} отложено</span>
        </div>
        {!embedded && onClose && <button type="button" className="icon-btn" onClick={onClose} aria-label="Закрыть исследовательские вопросы"><IconClose /></button>}
      </header>
      {error && <p role="alert" className="agenda-status-error">{error}</p>}
      <div className="agenda-list">
        {agenda.length === 0 && <p className="muted">Пункты появятся после первого подтверждённого поиска.</p>}
        {activeItems.map(renderItem)}
        {deferredItems.length > 0 && (
          <details className="agenda-closed" open={deferredOpen} onToggle={(event) => setDeferredOpen(event.currentTarget.open)}>
            <summary>Отложенные · {deferredItems.length}</summary>
            {deferredItems.map(renderItem)}
          </details>
        )}
        {closedItems.length > 0 && (
          <details className="agenda-closed" open={closedOpen} onToggle={(event) => setClosedOpen(event.currentTarget.open)}>
            <summary>Закрытые · {closedItems.length}</summary>
            {closedItems.map((item, index) => renderItem(item, activeItems.length + index))}
          </details>
        )}
      </div>
    </section>
  );
}

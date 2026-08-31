import { useMemo, useState } from "react";
import type { AgendaItem } from "../types";
import { IconClose } from "./Icons";

function directionNo(item: AgendaItem, fallback: number): number {
  const match = item.ref.match(/:(\d+)$/);
  return match ? Number(match[1]) : fallback;
}

export function AgendaDrawer({
  agenda,
  onClose,
  onToggle,
  embedded = false,
}: {
  agenda: AgendaItem[];
  onClose?: () => void;
  onToggle: (item: AgendaItem) => Promise<void>;
  embedded?: boolean;
}) {
  const [busy, setBusy] = useState(false);
  const [closedOpen, setClosedOpen] = useState(false);
  const openItems = useMemo(() => agenda.filter((item) => item.status === "open"), [agenda]);
  const closedItems = useMemo(() => agenda.filter((item) => item.status !== "open"), [agenda]);

  const renderItem = (item: AgendaItem, index: number) => (
    <article
      key={item.ref}
      className={`agenda-item is-${item.status} ${item.reviewRecommended && item.status === "open" ? "needs-review" : ""}`}
    >
      <button
        type="button"
        className="agenda-status"
        disabled={busy}
        onClick={async () => {
          setBusy(true);
          try { await onToggle(item); } finally { setBusy(false); }
        }}
        aria-label={item.status === "open" ? "Отметить выполненным" : "Вернуть в работу"}
        title={item.status === "open" ? "Отметить выполненным" : "Вернуть в работу"}
      >
        {item.status === "open" ? "○" : "✓"}
      </button>
      <div>
        <strong>Пункт {directionNo(item, index + 1)}</strong>
        <p>{item.text}</p>
        <span>{item.unitCount === 1 ? "1 цепочка" : `${item.unitCount} цепочек`}</span>
        {item.reviewRecommended && item.status === "open" && <em>Пора уточнить</em>}
      </div>
    </article>
  );

  return (
    <section className={`agenda-drawer ${embedded ? "is-embedded" : ""}`}>
      <header>
        <div>
          <strong>План поиска</strong>
          <span>{openItems.length} открыто · за один поиск до 5 пунктов</span>
        </div>
        {!embedded && onClose && <button type="button" className="icon-btn" onClick={onClose} aria-label="Закрыть план"><IconClose /></button>}
      </header>
      <div className="agenda-list">
        {agenda.length === 0 && <p className="muted">Пункты появятся после первого подтверждённого поиска.</p>}
        {openItems.map(renderItem)}
        {closedItems.length > 0 && (
          <details className="agenda-closed" open={closedOpen} onToggle={(event) => setClosedOpen(event.currentTarget.open)}>
            <summary>Выполненные · {closedItems.length}</summary>
            {closedItems.map((item, index) => renderItem(item, openItems.length + index))}
          </details>
        )}
      </div>
    </section>
  );
}

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
  const [closedOpen, setClosedOpen] = useState(false);
  const activeItems = useMemo(() => agenda.filter((item) => item.status !== "closed"), [agenda]);
  const closedItems = useMemo(() => agenda.filter((item) => item.status === "closed"), [agenda]);
  const partialCount = activeItems.filter((item) => item.status === "partial").length;
  const notClosedCount = activeItems.length - partialCount;

  const renderItem = (item: AgendaItem, index: number) => (
    <article
      key={item.ref}
      className={`agenda-item is-${item.status} ${item.reviewRecommended && item.status !== "closed" ? "needs-review" : ""}`}
    >
      <select
        className="agenda-status"
        disabled={busy || locked}
        value={item.status}
        onChange={async (event) => {
          const status = event.currentTarget.value as AgendaItem["status"];
          setBusy(true);
          try { await onStatus(item, status); } finally { setBusy(false); }
        }}
        aria-label={`Статус пункта ${directionNo(item, index + 1)}`}
        title={locked ? "Дождитесь окончания ответа" : "Изменить статус исследовательского вопроса"}
      >
        <option value="not_closed">Не закрыт</option>
        <option value="partial">Закрыт частично</option>
        <option value="closed">Закрыт</option>
      </select>
      <div>
        <strong>Пункт {directionNo(item, index + 1)}</strong>
        <p>{item.text}</p>
        <span className="agenda-chain-count">{item.unitCount === 1 ? "1 цепочка" : `${item.unitCount} цепочек`}</span>
        <span className={`agenda-coverage is-${item.status}`}><i aria-hidden="true">{item.status === "closed" ? "✓" : item.status === "partial" ? "◐" : "○"}</i>{item.status === "closed" ? "Закрыт" : item.status === "partial" ? "Закрыт частично" : "Не закрыт"}</span>
        {item.statusReason && <span className="agenda-reason">{item.statusReason}</span>}
        <span className="agenda-origin">{item.statusOrigin === "assistant" ? "Оценил ассистент" : item.statusOrigin === "user" ? "Изменено вами" : "Исходный статус"}</span>
        {item.reviewRecommended && item.status !== "closed" && <em>Пора уточнить</em>}
      </div>
    </article>
  );

  return (
    <section className={`agenda-drawer ${embedded ? "is-embedded" : ""}`}>
      <header>
        <div>
          <strong>Исследовательские вопросы</strong>
          <span>{notClosedCount} не закрыто · {partialCount} частично · {closedItems.length} закрыто</span>
        </div>
        {!embedded && onClose && <button type="button" className="icon-btn" onClick={onClose} aria-label="Закрыть исследовательские вопросы"><IconClose /></button>}
      </header>
      <div className="agenda-list">
        {agenda.length === 0 && <p className="muted">Пункты появятся после первого подтверждённого поиска.</p>}
        {activeItems.map(renderItem)}
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

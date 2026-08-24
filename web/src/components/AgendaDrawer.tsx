import { useState } from "react";
import type { AgendaItem } from "../types";
import { IconClose, IconPlus } from "./Icons";

export function AgendaDrawer({
  agenda,
  onClose,
  onAdd,
  onToggle,
}: {
  agenda: AgendaItem[];
  onClose: () => void;
  onAdd: (text: string) => Promise<void>;
  onToggle: (item: AgendaItem) => Promise<void>;
}) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);

  return (
    <aside className="agenda-drawer">
      <header>
        <div>
          <strong>Текущие SQ</strong>
          <span>{agenda.filter((item) => item.status === "open").length} открыто</span>
        </div>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Закрыть SQ">
          <IconClose />
        </button>
      </header>
      <div className="agenda-list">
        {agenda.length === 0 && <p className="muted">SQ появятся после декомпозиции вопроса.</p>}
        {agenda.map((item) => (
          <article key={item.id} className={`agenda-item is-${item.status}`}>
            <button
              type="button"
              className="agenda-status"
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                try { await onToggle(item); } finally { setBusy(false); }
              }}
              aria-label={item.status === "open" ? "Закрыть SQ" : "Открыть SQ"}
            >
              {item.status === "open" ? "○" : "✓"}
            </button>
            <div>
              <p>{item.text}</p>
              <span>{item.questionCount} ит. · {item.unitCount} UNIT</span>
              {item.reviewRecommended && item.status === "open" && (
                <em>Рекомендуется уточнить или закрыть SQ</em>
              )}
            </div>
          </article>
        ))}
      </div>
      <form
        className="agenda-add"
        onSubmit={async (event) => {
          event.preventDefault();
          if (!text.trim() || busy) return;
          setBusy(true);
          try {
            await onAdd(text.trim());
            setText("");
          } finally {
            setBusy(false);
          }
        }}
      >
        <input value={text} onChange={(event) => setText(event.target.value)} placeholder="Добавить SQ" />
        <button type="submit" className="icon-btn" disabled={!text.trim() || busy} aria-label="Добавить SQ">
          <IconPlus />
        </button>
      </form>
    </aside>
  );
}

import { useEffect, useState } from "react";
import type { ExploreField, GraphPayload } from "../types";
import { fetchGraphExplore } from "../api";
import { GraphCanvas } from "./GraphCanvas";
import { IconClose } from "./Icons";

const LIMITS = [10, 100, 1000] as const;
const FIELDS: { id: ExploreField; label: string }[] = [
  { id: "all", label: "Всё" },
  { id: "name", label: "Имя" },
  { id: "rel", label: "Связь" },
  { id: "evidence", label: "Цитата" },
];

const PLACEHOLDER: Record<ExploreField, string> = {
  all: "Имя вершины, тип связи или цитата",
  name: "Имя вершины",
  rel: "Тип связи, например PRODUCES",
  evidence: "Текст evidence или имя файла",
};

export function Explorer({ onClose }: { onClose: () => void }) {
  const [q, setQ] = useState("");
  const [limit, setLimit] = useState<(typeof LIMITS)[number]>(100);
  const [field, setField] = useState<ExploreField>("all");
  const [payload, setPayload] = useState<GraphPayload | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function run(
    nextQ = q,
    nextLimit = limit,
    nextField = field
  ) {
    setBusy(true);
    setError("");
    try {
      setPayload(await fetchGraphExplore(nextQ, nextLimit, nextField));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Ошибка загрузки");
    } finally {
      setBusy(false);
    }
  }

  useEffect(() => {
    void run("", 100, "all");
    // initial connected sample
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="explorer">
      <header className="explorer-bar">
        <strong>Граф базы</strong>
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") void run();
          }}
          placeholder={PLACEHOLDER[field]}
          aria-label="Поиск по графу"
        />
        <button type="button" className="explorer-search" onClick={() => void run()} disabled={busy}>
          Найти
        </button>
        <div className="field-chips">
          {FIELDS.map((item) => (
            <button
              key={item.id}
              type="button"
              className={item.id === field ? "is-on" : ""}
              onClick={() => {
                setField(item.id);
                void run(q, limit, item.id);
              }}
            >
              {item.label}
            </button>
          ))}
        </div>
        <div className="limit-chips">
          {LIMITS.map((item) => (
            <button
              key={item}
              type="button"
              className={item === limit ? "is-on" : ""}
              onClick={() => {
                setLimit(item);
                void run(q, item, field);
              }}
            >
              {item}
            </button>
          ))}
        </div>
        <button type="button" className="icon-btn" onClick={onClose} aria-label="Закрыть">
          <IconClose />
        </button>
      </header>
      {busy && <p className="explorer-status">Загрузка триплетов…</p>}
      {error && <p className="explorer-status is-error">{error}</p>}
      <GraphCanvas
        payload={payload}
        viewId="all"
        emptyHint="Нет триплетов по этому запросу. Смените поле поиска или limit."
        hideSearch
      />
    </div>
  );
}

import { useEffect, useMemo, useState } from "react";
import { deleteDocumentBatchItem } from "../api";
import { documentStatusLabel } from "../documents";
import type { DocumentBatch } from "../types";
import { DocumentReader } from "./DocumentReader";
import { IconClose } from "./Icons";

const FILTERS = [
  { id: "all", label: "Все" },
  { id: "articles", label: "Научные статьи" },
  { id: "regulations", label: "Нормативные документы" },
  { id: "uploads", label: "Мои файлы" },
] as const;

type FilterId = (typeof FILTERS)[number]["id"];
type DocumentKind = "upload" | "article" | "regulation";

const KIND_LABEL: Record<DocumentKind, string> = {
  upload: "Мой файл",
  article: "Научная",
  regulation: "Норматив",
};

const EMPTY_COPY: Record<FilterId, string> = {
  all: "Документов пока нет",
  articles: "Научных статей пока нет",
  regulations: "Нормативных документов пока нет",
  uploads: "Загруженных файлов пока нет",
};

type LibraryDoc = {
  id: string;
  title: string;
  kind: DocumentKind;
  batchId?: string;
  status?: string;
  error?: string | null;
  canRemove?: boolean;
};

function statusTone(status: string): "ok" | "warn" | "danger" | "faint" {
  if (status === "completed") return "ok";
  if (status === "failed") return "danger";
  if (status === "processing" || status === "queued") return "warn";
  return "faint";
}

function canRemoveStatus(status: string): boolean {
  return status === "unavailable" || status === "queued" || status === "cancelled";
}

function matchesFilter(doc: LibraryDoc, filter: FilterId): boolean {
  if (filter === "all") return true;
  if (filter === "uploads") return doc.kind === "upload";
  if (filter === "articles") return doc.kind === "article";
  return doc.kind === "regulation";
}

export function LibraryWorkspace({
  ingestEnabled,
  batches,
  loading = false,
  focusUploadsKey = 0,
  onOpenUpload,
  onNotice,
  onUploadsVisible,
  onBatchesChanged,
}: {
  ingestEnabled: boolean;
  ingestReady: boolean;
  batches: DocumentBatch[];
  loading?: boolean;
  focusUploadsKey?: number;
  onOpenUpload: () => void;
  onNotice: (text: string) => void;
  onUploadsVisible: (visible: boolean) => void;
  onBatchesChanged: () => void;
}) {
  const [filter, setFilter] = useState<FilterId>("all");
  const [query, setQuery] = useState("");
  const [selectedId, setSelectedId] = useState("");
  const [pendingId, setPendingId] = useState("");
  const [busyId, setBusyId] = useState("");

  const documents = useMemo<LibraryDoc[]>(() => (
    batches.flatMap((batch) => batch.items.map((item) => ({
      id: item.id,
      title: item.filename,
      kind: "upload" as const,
      batchId: batch.id,
      status: item.status,
      error: item.error || (item.status === "unavailable" ? batch.message : null),
      canRemove: canRemoveStatus(item.status),
    })))
  ), [batches]);

  useEffect(() => {
    if (ingestEnabled && focusUploadsKey > 0) {
      setFilter("uploads");
      setSelectedId("");
    }
  }, [focusUploadsKey, ingestEnabled]);

  useEffect(() => {
    const visible = ingestEnabled && filter === "uploads";
    onUploadsVisible(visible);
    return () => onUploadsVisible(false);
  }, [filter, ingestEnabled, onUploadsVisible]);

  useEffect(() => {
    if (selectedId && !documents.some((doc) => doc.id === selectedId)) setSelectedId("");
    if (pendingId && !documents.some((doc) => doc.id === pendingId)) setPendingId("");
  }, [documents, pendingId, selectedId]);

  useEffect(() => {
    if (!pendingId) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      setPendingId("");
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [pendingId]);

  const selected = documents.find((doc) => doc.id === selectedId) || null;
  const needle = query.trim().toLowerCase();
  const inFilter = documents.filter((doc) => matchesFilter(doc, filter));
  const visible = needle ? inFilter.filter((doc) => doc.title.toLowerCase().includes(needle)) : inFilter;

  const removeItem = async (doc: LibraryDoc) => {
    if (!doc.batchId) return;
    setBusyId(doc.id);
    try {
      await deleteDocumentBatchItem(doc.batchId, doc.id);
      setPendingId("");
      onBatchesChanged();
    } catch (error) {
      onNotice(error instanceof Error ? error.message : "Не удалось удалить файл");
    } finally {
      setBusyId("");
    }
  };

  if (selected) {
    return (
      <section className="ws-page" aria-label="Документы">
        <DocumentReader
          title={selected.title}
          kindLabel={KIND_LABEL[selected.kind]}
          error={selected.error}
          onClose={() => setSelectedId("")}
        />
      </section>
    );
  }

  return (
    <section className="ws-page" aria-label="Документы">
      <div className="ws-toolbar">
        <label className="ws-search">
          <span className="sr-only">Поиск по документам</span>
          <input type="search" placeholder="Найти документ по названию" value={query} onChange={(event) => setQuery(event.target.value)} />
        </label>
        <div className="ws-seg" role="group" aria-label="Тип документов">
          {FILTERS.map((item) => (
            <button
              key={item.id}
              type="button"
              aria-pressed={filter === item.id}
              onClick={() => setFilter(item.id)}
            >{item.label}</button>
          ))}
        </div>
        {ingestEnabled && <button type="button" className="primary-btn" onClick={onOpenUpload}>Загрузить PDF</button>}
      </div>
      <div className="ws-body">
        {loading && documents.length === 0 && <p className="ws-empty">Загрузка списка…</p>}
        {!loading && inFilter.length === 0 && (
          <>
            <p className="ws-empty">{EMPTY_COPY[filter]}</p>
            <div className="ws-tile-grid" aria-hidden="true">
              {[0, 1, 2, 3, 4].map((index) => (
                <div key={index} className="ws-tile is-ghost">
                  <div className="ws-tile-preview" />
                  <div className="ws-tile-meta"><span className="ws-tile-title" /></div>
                </div>
              ))}
            </div>
          </>
        )}
        {inFilter.length > 0 && visible.length === 0 && <p className="ws-empty">Ничего не найдено</p>}
        {visible.length > 0 && (
          <div className="ws-tile-grid">
            {visible.map((doc) => (
              <article key={doc.id} className={pendingId === doc.id ? "ws-tile is-confirming" : "ws-tile"}>
                <button type="button" className="ws-tile-open" onClick={() => { if (pendingId !== doc.id) setSelectedId(doc.id); }}>
                  <div className="ws-tile-preview" aria-hidden="true" />
                  <div className="ws-tile-meta">
                    <strong className="ws-tile-title">{doc.title}</strong>
                    <div className="ws-tile-badges">
                      <span className="ws-badge">{KIND_LABEL[doc.kind]}</span>
                      {doc.kind === "upload" && doc.status && (
                        <span className={`ws-badge is-${statusTone(doc.status)}`}>{documentStatusLabel(doc.status)}</span>
                      )}
                    </div>
                  </div>
                </button>
                {doc.canRemove && pendingId !== doc.id && (
                  <button
                    type="button"
                    className="icon-btn ws-tile-remove"
                    aria-label={`Убрать ${doc.title}`}
                    title={`Убрать ${doc.title}`}
                    disabled={busyId === doc.id}
                    onClick={() => setPendingId(doc.id)}
                  >
                    <IconClose />
                  </button>
                )}
                {pendingId === doc.id && (
                  <div className="ws-tile-confirm" role="dialog" aria-label={`Убрать ${doc.title}?`}>
                    <p>Убрать этот файл?</p>
                    <div className="ws-tile-confirm-actions">
                      <button type="button" className="ghost-btn" onClick={() => setPendingId("")}>Отмена</button>
                      <button type="button" className="ghost-btn danger-btn" disabled={busyId === doc.id} onClick={() => void removeItem(doc)}>Убрать</button>
                    </div>
                  </div>
                )}
              </article>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

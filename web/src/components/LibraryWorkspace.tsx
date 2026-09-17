import { useEffect, useState } from "react";
import { deleteDocumentBatchItem } from "../api";
import { documentQueueSummary, documentStatusLabel, formatEta, formatFileSize, isActiveDocumentStatus } from "../documents";
import type { DocumentBatch } from "../types";
import { IconClose, IconLibrary } from "./Icons";

const catalog = [
  {
    id: "articles",
    label: "Научные статьи",
    description: "Здесь будут научные публикации, на основе которых создана база знаний.",
    features: ["Поиск по статьям", "Метаданные и источники", "Связь с базой"],
  },
  {
    id: "regulations",
    label: "Нормативные документы",
    description: "Здесь будут нормативные документы для проверки заквасок и продуктов. Проверка ассистентом — в разработке.",
    features: ["Поиск по нормативам", "Проверка карточек", "Использование ассистентом"],
  },
];

export function LibraryWorkspace({
  ingestEnabled,
  ingestReady,
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
  const tabs = ingestEnabled
    ? [{ id: "uploads", label: "Загрузки" }, ...catalog]
    : catalog;
  const [activeTab, setActiveTab] = useState(tabs[0].id);
  const [busyId, setBusyId] = useState("");

  useEffect(() => {
    if (!tabs.some((tab) => tab.id === activeTab)) setActiveTab(tabs[0].id);
  }, [activeTab, ingestEnabled]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (ingestEnabled && focusUploadsKey > 0) setActiveTab("uploads");
  }, [focusUploadsKey, ingestEnabled]);

  useEffect(() => {
    const visible = ingestEnabled && activeTab === "uploads";
    onUploadsVisible(visible);
    return () => onUploadsVisible(false);
  }, [activeTab, ingestEnabled, onUploadsVisible]);

  const files = batches.flatMap((batch) => batch.items.map((item) => ({ batch, item })));
  const summary = documentQueueSummary(batches);

  const removeItem = async (batchId: string, itemId: string) => {
    setBusyId(itemId);
    try {
      await deleteDocumentBatchItem(batchId, itemId);
      onBatchesChanged();
    } catch (error) {
      onNotice(error instanceof Error ? error.message : "Не удалось удалить файл");
    } finally {
      setBusyId("");
    }
  };

  return (
    <section className="article-library" aria-labelledby="article-library-title">
      <div className="article-library-inner">
        <header className="article-library-header">
          <div>
            <p className="article-library-kicker">ИСТОЧНИКИ ЗНАНИЙ</p>
            <h1 id="article-library-title">Библиотека документов</h1>
            <p>{ingestEnabled ? "Загрузка PDF в очередь обработки и будущая библиотека корпуса." : "Научные статьи и нормативные документы."}</p>
          </div>
          {ingestEnabled ? (
            <button type="button" className="primary-btn" onClick={onOpenUpload}>Загрузить PDF</button>
          ) : (
            <span className="article-library-status">В разработке</span>
          )}
        </header>

        <div className="document-library-tabs" role="tablist" aria-label="Виды документов">
          {tabs.map((section, index) => (
            <button
              key={section.id}
              type="button"
              role="tab"
              id={`document-tab-${section.id}`}
              aria-controls={`document-panel-${section.id}`}
              aria-selected={activeTab === section.id}
              tabIndex={activeTab === section.id ? 0 : -1}
              onClick={() => setActiveTab(section.id)}
              onKeyDown={(event) => {
                let nextIndex = index;
                if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
                else if (event.key === "ArrowLeft") nextIndex = (index + tabs.length - 1) % tabs.length;
                else if (event.key === "Home") nextIndex = 0;
                else if (event.key === "End") nextIndex = tabs.length - 1;
                else return;
                event.preventDefault();
                setActiveTab(tabs[nextIndex].id);
                document.getElementById(`document-tab-${tabs[nextIndex].id}`)?.focus();
              }}
            >{section.label}</button>
          ))}
        </div>

        {ingestEnabled && (
          <div role="tabpanel" id="document-panel-uploads" aria-labelledby="document-tab-uploads" hidden={activeTab !== "uploads"} tabIndex={0}>
            <p className="document-upload-hint">
              {ingestReady
                ? "Статус каждого файла обновляется по мере обработки."
                : "PDF можно выбрать и отправить. Сервис обработки ещё не подключён, поэтому файлы остаются в очереди со статусом «Ожидает сервис»."}
            </p>
            {summary && <p className="document-queue-summary">{summary}</p>}
            {loading && files.length === 0 && <p className="document-upload-hint">Загрузка списка…</p>}
            {!loading && files.length === 0 && (
              <div className="article-library-placeholder">
                <div className="article-library-icon" aria-hidden="true"><IconLibrary /></div>
                <h2>Загрузок пока нет</h2>
                <p>Добавьте PDF кнопкой «Загрузить PDF» в шапке или скрепкой в поле сообщения. Файлы появятся здесь вместе со статусом обработки.</p>
              </div>
            )}
            {files.length > 0 && (
              <ul className="document-batch-list">
                {files.map(({ batch, item }) => {
                  const eta = isActiveDocumentStatus(item.status) ? formatEta(item.etaSeconds) : "";
                  const canRemove = item.status === "unavailable" || item.status === "queued" || item.status === "cancelled";
                  return (
                    <li key={item.id} className={`document-batch-row is-${item.status}`}>
                      <div>
                        <strong>{item.filename}</strong>
                        <span>
                          {formatFileSize(item.size)}
                          <i aria-hidden="true">·</i>
                          {documentStatusLabel(item.status)}
                          {eta ? <><i aria-hidden="true">·</i>{eta}</> : null}
                        </span>
                        {(item.error || (item.status === "unavailable" && batch.message)) && (
                          <small>{item.error || batch.message}</small>
                        )}
                      </div>
                      {canRemove && (
                        <button
                          type="button"
                          className="icon-btn"
                          aria-label={`Убрать ${item.filename}`}
                          disabled={busyId === item.id}
                          onClick={() => void removeItem(batch.id, item.id)}
                        >
                          <IconClose />
                        </button>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        )}

        {catalog.map((section) => (
          <div key={section.id} role="tabpanel" id={`document-panel-${section.id}`} aria-labelledby={`document-tab-${section.id}`} hidden={activeTab !== section.id} tabIndex={0}>
            <div className="article-library-placeholder">
              <div className="article-library-icon" aria-hidden="true"><IconLibrary /></div>
              <h2>{section.label} · Скоро</h2>
              <p>{section.description}</p>
              <div className="article-library-plan" aria-label="Запланированные возможности">
                {section.features.map((feature) => <span key={feature}>{feature}</span>)}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

import { useEffect, useRef, useState } from "react";
import type { DragEvent } from "react";
import { createDocumentBatch } from "../api";
import { formatFileSize, isPdfFile } from "../documents";
import { IconAttach, IconClose } from "./Icons";

type SelectedFile = { id: string; file: File };

function nextId(): string {
  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

export function DocumentUploadModal({
  open,
  ingestReady,
  maxFiles,
  maxFileBytes,
  onClose,
  onCreated,
}: {
  open: boolean;
  ingestReady: boolean;
  maxFiles: number;
  maxFileBytes: number;
  onClose: () => void;
  onCreated: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [selected, setSelected] = useState<SelectedFile[]>([]);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    if (!open) {
      setSelected([]);
      setError("");
      setBusy(false);
      setDragOver(false);
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [busy, onClose, open]);

  const addFiles = (list: FileList | File[]) => {
    const incoming = Array.from(list);
    if (!incoming.length) return;
    const rejected: string[] = [];
    setSelected((current) => {
      const next = [...current];
      for (const file of incoming) {
        if (!isPdfFile(file)) {
          rejected.push(`${file.name}: нужен PDF`);
          continue;
        }
        if (file.size > maxFileBytes) {
          rejected.push(`${file.name}: больше ${formatFileSize(maxFileBytes)}`);
          continue;
        }
        const duplicate = next.some((item) => item.file.name === file.name && item.file.size === file.size && item.file.lastModified === file.lastModified);
        if (duplicate) continue;
        if (next.length >= maxFiles) {
          rejected.push(`Можно выбрать не больше ${maxFiles} файлов`);
          break;
        }
        next.push({ id: nextId(), file });
      }
      return next;
    });
    setError(rejected[0] || "");
  };

  const onDrop = (event: DragEvent<HTMLElement>) => {
    event.preventDefault();
    setDragOver(false);
    if (event.dataTransfer.files?.length) addFiles(event.dataTransfer.files);
  };

  const submit = async () => {
    if (!selected.length || busy) return;
    setBusy(true);
    setError("");
    try {
      await createDocumentBatch(selected.map((item) => item.file));
      onCreated();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Не удалось отправить документы");
    } finally {
      setBusy(false);
    }
  };

  if (!open) return null;

  return (
    <div
      className="document-upload-overlay"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !busy) onClose();
      }}
    >
      <div className="document-upload-modal" role="dialog" aria-modal="true" aria-labelledby="document-upload-title">
        <header>
          <div>
            <h2 id="document-upload-title">Добавить PDF в базу</h2>
            <p>
              {ingestReady
                ? "Файлы уйдут в обработку отдельно от этого чата."
                : "Файлы принимаются в очередь. Обработка документов будет подключена позже."}
            </p>
          </div>
          <button type="button" className="icon-btn" aria-label="Закрыть" disabled={busy} onClick={onClose}>
            <IconClose />
          </button>
        </header>

        <button
          type="button"
          className={`document-upload-dropzone ${dragOver ? "is-over" : ""}`}
          onClick={() => inputRef.current?.click()}
          onDragEnter={(event) => { event.preventDefault(); setDragOver(true); }}
          onDragOver={(event) => { event.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
        >
          <IconAttach />
          <strong>Перетащите PDF сюда</strong>
          <span>Или нажмите, чтобы выбрать</span>
          <span>До {maxFiles} файлов, каждый до {formatFileSize(maxFileBytes)}</span>
        </button>
        <input
          ref={inputRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          hidden
          onChange={(event) => {
            if (event.currentTarget.files) addFiles(event.currentTarget.files);
            event.currentTarget.value = "";
          }}
        />

        {selected.length > 0 && (
          <ul className="document-upload-list">
            {selected.map((item) => (
              <li key={item.id}>
                <span>
                  <strong>{item.file.name}</strong>
                  <small>{formatFileSize(item.file.size)}</small>
                </span>
                <button
                  type="button"
                  className="icon-btn"
                  aria-label={`Убрать ${item.file.name}`}
                  disabled={busy}
                  onClick={() => setSelected((current) => current.filter((entry) => entry.id !== item.id))}
                >
                  <IconClose />
                </button>
              </li>
            ))}
          </ul>
        )}

        {error && <p className="document-upload-error" role="alert">{error}</p>}

        <footer>
          <button type="button" className="ghost-btn document-upload-cancel" disabled={busy} onClick={onClose}>Отмена</button>
          <button type="button" className="primary-btn" disabled={busy || selected.length === 0} onClick={() => void submit()}>
            {busy ? "Отправляю…" : "Отправить"}
          </button>
        </footer>
      </div>
    </div>
  );
}

import { useEffect } from "react";
import { IconClose } from "./Icons";

function IconBack() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
      <path d="M15 6 9 12l6 6" />
    </svg>
  );
}

export function DocumentReader({
  title,
  kindLabel,
  error,
  onClose,
}: {
  title: string;
  kindLabel: string;
  error?: string | null;
  onClose: () => void;
}) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      event.preventDefault();
      onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="document-reader" role="region" aria-label="Читалка документа">
      <header className="document-reader-bar">
        <button type="button" className="icon-btn" aria-label="Назад" onClick={onClose}>
          <IconBack />
        </button>
        <h2 title={title}>{title}</h2>
        <span className="ws-badge">{kindLabel}</span>
        <span className="document-reader-esc">Esc</span>
        <button type="button" className="icon-btn" aria-label="Закрыть" onClick={onClose}>
          <IconClose />
        </button>
      </header>
      <div className="document-reader-stage">
        <div className="document-reader-sheet">
          Страница появится, когда подключим хранилище.
        </div>
        {error ? <p className="document-reader-error">{error}</p> : null}
      </div>
    </div>
  );
}

import { useEffect, useRef } from "react";
import { IconCards, IconLibrary, IconMic, IconSend, IconStop } from "./Icons";

export function Composer({
  text,
  onText,
  onSubmit,
  onStop,
  busy,
  audioEnabled,
  onMic,
  recording,
  centered,
  mode,
  cardsEnabled,
  cardActionsEnabled,
  onOpenCards,
  documentsEnabled,
  onOpenDocuments,
  disabled = false,
  focusKey = 0,
}: {
  text: string;
  onText: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  busy: boolean;
  audioEnabled: boolean;
  onMic: () => void;
  recording: boolean;
  centered: boolean;
  mode: "auto" | "staged";
  cardsEnabled: boolean;
  cardActionsEnabled: boolean;
  onOpenCards: () => void;
  documentsEnabled: boolean;
  onOpenDocuments: () => void;
  disabled?: boolean;
  focusKey?: number;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [text]);

  useEffect(() => {
    if (focusKey > 0) ref.current?.focus();
  }, [focusKey]);

  return (
    <form
      className={`composer ${centered ? "is-centered" : ""}`}
      onSubmit={(e) => {
        e.preventDefault();
        if (busy) onStop();
        else onSubmit();
      }}
    >
      <textarea
        ref={ref}
        rows={1}
        value={text}
        placeholder={mode === "staged" ? "Что исследовать?" : "Спросите базу…"}
        aria-label="Сообщение для Neo4j Assistant"
        disabled={disabled}
        onChange={(e) => onText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            if (busy) onStop();
            else onSubmit();
          }
        }}
      />
      <div className="composer-actions">
        {documentsEnabled && (
          <button
            type="button"
            className="chip-btn composer-action-btn composer-upload-button"
            aria-label="Добавить PDF в базу"
            title="Добавить PDF в очередь обработки — не к этому сообщению"
            onClick={onOpenDocuments}
          >
            <IconLibrary />
            <span className="composer-action-label">Добавить PDF</span>
          </button>
        )}
        {cardsEnabled && (
          <button
            type="button"
            className="chip-btn composer-action-btn composer-cards-button"
            aria-label="Открыть карточки"
            title={cardActionsEnabled ? "Открыть карточки" : "Карточки доступны после первого ответа и вне активной генерации."}
            disabled={!cardActionsEnabled}
            onClick={onOpenCards}
          >
            <IconCards />
            <span className="composer-action-label">Карточки</span>
          </button>
        )}
        {audioEnabled && (
          <button
            type="button"
            className={`chip-btn ${recording ? "is-live" : ""}`}
            onClick={onMic}
            disabled={disabled}
            aria-label={recording ? "Остановить запись" : "Начать запись"}
            title={recording ? "Остановить запись" : "Записать голосом"}
          >
            <IconMic />
          </button>
        )}
        <button
          type="submit"
          className="send-btn"
          aria-label={busy ? "Остановить" : "Отправить"}
          disabled={!busy && (!text.trim() || disabled)}
          title={busy ? "Остановить ответ" : "Отправить сообщение"}
        >
          {busy ? <IconStop /> : <IconSend />}
        </button>
      </div>
    </form>
  );
}

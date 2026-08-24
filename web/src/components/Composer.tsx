import { useEffect, useRef } from "react";
import type { SearchDepth, UiModel } from "../types";
import { DEPTH_LABELS, effortLabel } from "../format";
import { IconMic, IconSend, IconStop } from "./Icons";

export function Composer({
  text,
  onText,
  onSubmit,
  onStop,
  busy,
  audioEnabled,
  onMic,
  recording,
  depth,
  onDepth,
  effort,
  effortOptions,
  onEffort,
  profile,
  models,
  onProfile,
  centered,
  disabled = false,
}: {
  text: string;
  onText: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  busy: boolean;
  audioEnabled: boolean;
  onMic: () => void;
  recording: boolean;
  depth: SearchDepth;
  onDepth: (value: SearchDepth) => void;
  effort: string;
  effortOptions: string[];
  onEffort: (value: string) => void;
  profile: string;
  models: UiModel[];
  onProfile: (id: string) => void;
  centered: boolean;
  disabled?: boolean;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [text]);

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
        placeholder="Задайте вопрос по вашей базе знаний…"
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
        <label className="chip" title="Глубина поиска в графе">
          <span className="sr-only">Глубина поиска</span>
          <select value={depth} onChange={(e) => onDepth(e.target.value as SearchDepth)}>
            {(["low", "medium", "high"] as const).map((id) => (
              <option key={id} value={id}>
                {DEPTH_LABELS[id]}
              </option>
            ))}
          </select>
        </label>
        {effortOptions.length > 0 && (
          <label className="chip" title="Глубина рассуждения">
            <span className="sr-only">Глубина рассуждения</span>
            <select value={effort} onChange={(e) => onEffort(e.target.value)}>
              {effortOptions.map((id) => (
                <option key={id} value={id}>
                  {effortLabel(id)}
                </option>
              ))}
            </select>
          </label>
        )}
        {models.length > 0 && (
          <label className="chip chip-model" title="Модель">
            <span className="sr-only">Модель</span>
            <select value={profile} onChange={(e) => onProfile(e.target.value)}>
              {models.map((model) => (
                <option key={model.id} value={model.id}>
                  {model.label}
                </option>
              ))}
            </select>
          </label>
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

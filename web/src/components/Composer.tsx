import { useEffect, useRef } from "react";
import type { CardTemplate, SearchDepth, UiModel } from "../types";
import { DEPTH_LABELS, effortLabel } from "../format";
import { IconCards, IconMic, IconSend, IconStop } from "./Icons";

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
  mode,
  onMode,
  stagedEnabled,
  cardTemplates,
  cardBusy,
  cardEnabled,
  onGenerateCard,
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
  mode: "auto" | "staged";
  onMode: (value: "auto" | "staged") => void;
  stagedEnabled: boolean;
  cardTemplates: CardTemplate[];
  cardBusy: boolean;
  cardEnabled: boolean;
  onGenerateCard: (templateVersionId: string) => void;
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
        {cardTemplates.length > 0 && (
          <label className="chip card-chip" title={cardEnabled ? "Заполнить карточку из контекста ветки" : "Сначала начните диалог"}>
            <IconCards />
            <select
              value=""
              disabled={!cardEnabled || cardBusy || busy}
              aria-label="Создать карточку"
              onChange={(event) => {
                const value = event.target.value;
                if (value) onGenerateCard(value);
              }}
            >
              <option value="">{cardBusy ? "Карточка…" : "Карточка"}</option>
              {cardTemplates.map((template) => (
                <option key={template.latestVersion.id} value={template.latestVersion.id}>
                  {template.name}
                </option>
              ))}
            </select>
          </label>
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
        <label className="chip" title="Режим retrieval">
          <span className="sr-only">Режим</span>
          <select value={mode} onChange={(e) => onMode(e.target.value as "auto" | "staged")}>
            <option value="auto">Auto</option>
            {stagedEnabled && <option value="staged">По этапам</option>}
          </select>
        </label>
        {mode === "auto" ? (
          <label className="chip" title="Бюджет UNIT">
            <span className="sr-only">Бюджет UNIT</span>
            <select value={depth} onChange={(e) => onDepth(e.target.value as SearchDepth)}>
              {(["low", "medium", "high"] as const).map((id) => (
                <option key={id} value={id}>
                  {{ low: "5 UNIT", medium: "10 UNIT", high: "15 UNIT" }[id] || DEPTH_LABELS[id]}
                </option>
              ))}
            </select>
          </label>
        ) : <span className="manual-budget">1 UNIT / open SQ</span>}
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

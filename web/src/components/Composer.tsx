import { useEffect, useRef } from "react";
import type { SearchDepth, UiModel } from "../types";
import { effortLabel } from "../format";
import { IconCards, IconChevron, IconMic, IconPlus, IconSend, IconStop } from "./Icons";

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
  branchMode,
  cardsEnabled,
  cardActionsEnabled,
  onOpenCardTemplates,
  onOpenCardLibrary,
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
  branchMode?: "auto" | "staged";
  cardsEnabled: boolean;
  cardActionsEnabled: boolean;
  onOpenCardTemplates: () => void;
  onOpenCardLibrary: () => void;
  disabled?: boolean;
  focusKey?: number;
}) {
  const ref = useRef<HTMLTextAreaElement>(null);
  const formRef = useRef<HTMLFormElement>(null);
  const currentModel = models.find((model) => model.id === profile);

  const closeMenu = (target: HTMLElement) => {
    const menu = target.closest("details");
    if (menu instanceof HTMLDetailsElement) menu.open = false;
  };

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
  }, [text]);

  useEffect(() => {
    if (focusKey > 0) ref.current?.focus();
  }, [focusKey]);

  useEffect(() => {
    const closeOnOutsideClick = (event: MouseEvent) => {
      if (formRef.current?.contains(event.target as Node)) return;
      formRef.current?.querySelectorAll("details[open]").forEach((menu) => {
        (menu as HTMLDetailsElement).open = false;
      });
    };
    document.addEventListener("mousedown", closeOnOutsideClick);
    return () => document.removeEventListener("mousedown", closeOnOutsideClick);
  }, []);

  return (
    <form
      ref={formRef}
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
        {cardsEnabled && (
          <details className="composer-menu composer-add-menu" onToggle={(event) => {
            if (!event.currentTarget.open) return;
            formRef.current?.querySelectorAll("details[open]").forEach((menu) => {
              if (menu !== event.currentTarget) (menu as HTMLDetailsElement).open = false;
            });
          }}>
            <summary title="Карточки" aria-label="Карточки"><IconPlus /></summary>
            <div className="composer-popover composer-action-popover">
              <p className="composer-popover-title">Карточки</p>
              <button type="button" disabled={!cardActionsEnabled} onClick={(event) => { onOpenCardTemplates(); closeMenu(event.currentTarget); }}>
                <IconCards /><span><strong>Заполнить по шаблону</strong><small>Сформировать карточку из текущего диалога</small></span>
              </button>
              <button type="button" disabled={!cardActionsEnabled} onClick={(event) => { onOpenCardLibrary(); closeMenu(event.currentTarget); }}>
                <IconCards /><span><strong>Вставить карточку</strong><small>Добавить сохранённую карточку в контекст</small></span>
              </button>
              {!cardActionsEnabled && <p className="popover-note">Доступно после первого ответа и вне активной генерации.</p>}
            </div>
          </details>
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
        <details className="composer-menu retrieval-menu" onToggle={(event) => {
          if (!event.currentTarget.open) return;
          formRef.current?.querySelectorAll("details[open]").forEach((menu) => {
            if (menu !== event.currentTarget) (menu as HTMLDetailsElement).open = false;
          });
        }}>
          <summary>{mode === "auto" ? "Ответ сразу" : "С планом"}<IconChevron /></summary>
          <div className="composer-popover">
            <p className="composer-popover-title">Режим работы</p>
            {stagedEnabled && branchMode !== "auto" && <label><input type="radio" checked={mode === "staged"} onChange={(event) => { onMode("staged"); closeMenu(event.currentTarget); }} /><span><strong>С планом</strong><small>Разработка: вы видите, что искать, и подтверждаете новые пункты. Находки и план остаются в этой версии.</small></span></label>}
            <label><input type="radio" checked={mode === "auto"} onChange={(event) => { onMode("auto"); closeMenu(event.currentTarget); }} /><span><strong>Ответ сразу</strong><small>{branchMode === "staged" ? "Этот вопрос уйдёт в отдельный вариант диалога. Текущий план и находки сохранятся." : "Консультация по одному вопросу: ассистент сам ищет в базе и сразу отвечает. Следующий вопрос — новый поиск, план не копится."}</small></span></label>
            {branchMode === "auto" && <p className="popover-note">Режим с планом начинается в новом чате. Этот вариант остаётся консультацией.</p>}
            {mode === "auto" ? (
              <div className="popover-setting"><span>Объём данных</span><div>
                {(["low", "medium", "high"] as const).map((id) => <button key={id} type="button" className={depth === id ? "is-on" : ""} onClick={(event) => { onDepth(id); closeMenu(event.currentTarget); }}>{{ low: "Компактно", medium: "Обычно", high: "Расширенно" }[id]}</button>)}
              </div></div>
            ) : <p className="popover-note">За один шаг — один поиск по выбранным пунктам плана.</p>}
          </div>
        </details>
        {models.length > 0 && (
          <details className="composer-menu model-menu" onToggle={(event) => {
            if (!event.currentTarget.open) return;
            formRef.current?.querySelectorAll("details[open]").forEach((menu) => {
              if (menu !== event.currentTarget) (menu as HTMLDetailsElement).open = false;
            });
          }}>
            <summary>{currentModel?.label || "Модель"}<IconChevron /></summary>
            <div className="composer-popover composer-model-popover">
              <p className="composer-popover-title">Модель</p>
              {models.map((model) => <label key={model.id}><input type="radio" checked={profile === model.id} onChange={(event) => { onProfile(model.id); closeMenu(event.currentTarget); }} /><span><strong>{model.label}</strong></span></label>)}
              {effortOptions.length > 1 && <div className="popover-setting"><span>Насколько вдумчиво</span><div>{effortOptions.map((id) => <button key={id} type="button" className={effort === id ? "is-on" : ""} onClick={(event) => { onEffort(id); closeMenu(event.currentTarget); }}>{effortLabel(id)}</button>)}</div></div>}
            </div>
          </details>
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

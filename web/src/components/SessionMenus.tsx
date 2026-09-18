import { useEffect, useLayoutEffect, useRef, useState } from "react";
import type { UiModel } from "../types";
import { effortLabel } from "../format";
import { MODE_LABELS } from "../uiLabels";
import { IconChevron } from "./Icons";

export function SessionMenus({
  showMode,
  mode,
  onMode,
  stagedEnabled,
  branchMode,
  profile,
  models,
  onProfile,
  effort,
  effortOptions,
  onEffort,
  busy,
  disabled = false,
}: {
  showMode: boolean;
  mode: "auto" | "staged";
  onMode: (value: "auto" | "staged") => void;
  stagedEnabled: boolean;
  branchMode?: "auto" | "staged";
  profile: string;
  models: UiModel[];
  onProfile: (id: string) => void;
  effort: string;
  effortOptions: string[];
  onEffort: (value: string) => void;
  busy: boolean;
  disabled?: boolean;
}) {
  const rootRef = useRef<HTMLDivElement>(null);
  const modeMenuRef = useRef<HTMLDetailsElement>(null);
  const modePopoverRef = useRef<HTMLDivElement>(null);
  const modelMenuRef = useRef<HTMLDetailsElement>(null);
  const modelPopoverRef = useRef<HTMLDivElement>(null);
  const [openMenu, setOpenMenu] = useState<"mode" | "model" | null>(null);
  const currentModel = models.find((model) => model.id === profile);

  const closeMenus = () => {
    setOpenMenu(null);
    rootRef.current?.querySelectorAll("details[open]").forEach((menu) => {
      (menu as HTMLDetailsElement).open = false;
    });
  };

  useLayoutEffect(() => {
    const menu = openMenu === "mode" ? modeMenuRef.current : openMenu === "model" ? modelMenuRef.current : null;
    const popover = openMenu === "mode" ? modePopoverRef.current : openMenu === "model" ? modelPopoverRef.current : null;
    if (!menu || !popover) return;
    const position = () => {
      const anchor = menu.getBoundingClientRect();
      const viewport = window.visualViewport;
      const left = viewport?.offsetLeft ?? 0;
      const top = viewport?.offsetTop ?? 0;
      const width = viewport?.width ?? window.innerWidth;
      const height = viewport?.height ?? window.innerHeight;
      const panelWidth = Math.min(320, width - 24);
      const above = anchor.top - top - 20;
      const below = top + height - anchor.bottom - 20;
      const upward = above >= Math.min(300, below);
      popover.style.width = `${panelWidth}px`;
      popover.style.setProperty("--session-menu-left", `${Math.max(left + 12, Math.min(anchor.right - panelWidth, left + width - panelWidth - 12))}px`);
      popover.style.maxHeight = `${Math.max(0, upward ? above : below)}px`;
      popover.style.top = upward ? "auto" : `${anchor.bottom + 8}px`;
      popover.style.bottom = upward ? `${window.innerHeight - anchor.top + 8}px` : "auto";
    };
    position();
    popover.querySelector<HTMLInputElement>("input:checked")?.closest("label")?.scrollIntoView({ block: "nearest" });
    window.addEventListener("resize", position);
    window.addEventListener("scroll", position, true);
    window.visualViewport?.addEventListener("resize", position);
    window.visualViewport?.addEventListener("scroll", position);
    return () => {
      window.removeEventListener("resize", position);
      window.removeEventListener("scroll", position, true);
      window.visualViewport?.removeEventListener("resize", position);
      window.visualViewport?.removeEventListener("scroll", position);
    };
  }, [openMenu, effortOptions.length]);

  const closeMenu = (target: HTMLElement) => {
    const menu = target.closest("details");
    if (menu instanceof HTMLDetailsElement) menu.open = false;
    setOpenMenu(null);
  };

  const onToggle = (which: "mode" | "model") => (event: { currentTarget: HTMLDetailsElement }) => {
    const open = event.currentTarget.open;
    setOpenMenu(open ? which : null);
    if (!open) return;
    rootRef.current?.querySelectorAll("details[open]").forEach((menu) => {
      if (menu !== event.currentTarget) (menu as HTMLDetailsElement).open = false;
    });
  };

  useEffect(() => {
    if (!busy) return;
    closeMenus();
  }, [busy]);

  useEffect(() => {
    const closeOnOutsideClick = (event: MouseEvent) => {
      const target = event.target;
      if (!(target instanceof Element)) return;
      if (target.closest("details.session-menu")) return;
      closeMenus();
    };
    document.addEventListener("mousedown", closeOnOutsideClick);
    return () => document.removeEventListener("mousedown", closeOnOutsideClick);
  }, []);

  if (!showMode && models.length === 0) return null;

  return (
    <div ref={rootRef} className="session-menus">
      {showMode && (
        <details ref={modeMenuRef} className="session-menu composer-menu retrieval-menu" onToggle={onToggle("mode")}>
          <summary><span>{MODE_LABELS[mode]}</span><IconChevron /></summary>
          <div ref={modePopoverRef} className="composer-popover">
            <p className="composer-popover-title">Режим работы</p>
            {stagedEnabled && branchMode !== "auto" && <label><input type="radio" checked={mode === "staged"} disabled={disabled} onChange={(event) => { onMode("staged"); closeMenu(event.currentTarget); }} /><span><strong>Исследование</strong><small>Сохраняет исследовательские вопросы и найденные данные, чтобы продолжать работу по направлениям.</small></span></label>}
            <label><input type="radio" checked={mode === "auto"} disabled={disabled} onChange={(event) => { onMode("auto"); closeMenu(event.currentTarget); }} /><span><strong>Вопрос по базе</strong><small>{branchMode === "staged" ? "Ответит на отдельный вопрос в новом варианте. Текущее исследование сохранится." : "Отвечает на один самостоятельный вопрос без накопления плана."}</small></span></label>
            {branchMode === "auto" && <p className="popover-note">Исследование начинается в новом чате. Этот вариант остаётся в режиме «Вопрос по базе».</p>}
            {mode !== "auto" && <p className="popover-note">За один шаг — один поиск по выбранным исследовательским вопросам.</p>}
          </div>
        </details>
      )}
      {models.length > 0 && (
        <details ref={modelMenuRef} className="session-menu composer-menu model-menu" onToggle={onToggle("model")}>
          <summary><span>{currentModel?.label || "Модель"}</span><IconChevron /></summary>
          <div ref={modelPopoverRef} className="composer-popover composer-model-popover">
            <p className="composer-popover-title">Модель</p>
            <div className="composer-model-list">{models.map((model) => <label key={model.id}><input type="radio" checked={profile === model.id} disabled={disabled} onChange={(event) => { onProfile(model.id); closeMenu(event.currentTarget); }} /><span><strong>{model.label}</strong></span></label>)}</div>
            {effortOptions.length > 0 && <div className="popover-setting"><span>Насколько вдумчиво</span><div>{effortOptions.map((id) => <button key={id} type="button" className={effort === id ? "is-on" : ""} disabled={disabled} onClick={(event) => { onEffort(id); closeMenu(event.currentTarget); }}>{effortLabel(id)}</button>)}</div></div>}
          </div>
        </details>
      )}
    </div>
  );
}

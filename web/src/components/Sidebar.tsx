import { useEffect, useMemo, useState } from "react";
import type { ConversationSummary } from "../types";
import { formatRelativeTime } from "../format";
import { IconCards, IconChat, IconGraph, IconHelp, IconLibrary, IconLogout, IconPlus, IconSettings, IconSidebar, IconTrash } from "./Icons";

const RELATIVE_TICK_MS = 30_000;

export function Sidebar({
  collapsed,
  onToggle,
  sessions,
  currentId,
  username,
  onNewChat,
  onOpenSession,
  onDeleteSession,
  deleteDisabled,
  onExplorer,
  onLibrary,
  onHelp,
  onCards,
  cardsEnabled,
  health,
  settingsOpen,
  onSettings,
  onLogout,
  keyWarning,
  activeWorkspace,
}: {
  collapsed: boolean;
  onToggle: () => void;
  sessions: ConversationSummary[];
  currentId: string;
  username: string;
  onNewChat: () => void;
  onOpenSession: (id: string) => void;
  onDeleteSession: (id: string) => Promise<void>;
  deleteDisabled: boolean;
  onExplorer: () => void;
  onLibrary: () => void;
  onHelp: () => void;
  onCards: () => void;
  cardsEnabled: boolean;
  health: string;
  settingsOpen: boolean;
  onSettings: () => void;
  onLogout: () => void;
  keyWarning: boolean;
  activeWorkspace: "chat" | "graph" | "library" | "help" | "cards";
}) {
  const [now, setNow] = useState(() => Date.now());
  const [historyQuery, setHistoryQuery] = useState("");
  const [historySearchUnlocked, setHistorySearchUnlocked] = useState(false);
  const visibleSessions = useMemo(() => {
    const query = historyQuery.trim().toLocaleLowerCase("ru-RU");
    return query
      ? sessions.filter((session) => session.title.toLocaleLowerCase("ru-RU").includes(query))
      : sessions;
  }, [historyQuery, sessions]);
  useEffect(() => {
    const tick = window.setInterval(() => setNow(Date.now()), RELATIVE_TICK_MS);
    return () => window.clearInterval(tick);
  }, []);

  return (
    <aside className={`sidebar ${collapsed ? "is-collapsed" : ""}`}>
      <div className="sidebar-top">
        <button type="button" className="icon-btn" onClick={onToggle} title={collapsed ? "Показать меню" : "Скрыть меню"} aria-label={collapsed ? "Показать меню" : "Скрыть меню"}>
          <IconSidebar />
        </button>
        {!collapsed && <span className="sidebar-brand">Neo4j Assistant</span>}
      </div>
      <button type="button" className={`sidebar-action sidebar-action-primary ${activeWorkspace === "chat" && !currentId ? "is-active" : ""}`} onClick={onNewChat} title="Новый чат">
        <IconPlus />
        {!collapsed && <span>Новый чат</span>}
      </button>
      <button type="button" className={`sidebar-action ${activeWorkspace === "graph" ? "is-active" : ""}`} onClick={onExplorer} title="Открыть всю базу">
        <IconGraph />
        {!collapsed && <span>Вся база</span>}
      </button>
      <button type="button" className={`sidebar-action ${activeWorkspace === "library" ? "is-active" : ""}`} onClick={onLibrary} title="Открыть документы">
        <IconLibrary />
        {!collapsed && <><span>Документы</span><span className="sidebar-nav-badge">Скоро</span></>}
      </button>
      {cardsEnabled && <button type="button" className={`sidebar-action ${activeWorkspace === "cards" ? "is-active" : ""}`} onClick={onCards} title="Карточки">
        <IconCards />
        {!collapsed && <span>Карточки</span>}
      </button>}
      <div className="sidebar-list" aria-label="История чатов">
        {collapsed && sessions.length > 0 && <button type="button" className="sidebar-chat-launcher" onClick={onToggle} title="Показать чаты" aria-label="Показать чаты"><IconChat /></button>}
        {!collapsed && sessions.length > 0 && <p className="sidebar-section">Недавние</p>}
        {!collapsed && (sessions.length > 6 || Boolean(historyQuery.trim())) && (
          <input
            className="sidebar-history-search"
            type="search"
            name="chat-history-filter"
            autoComplete="off"
            data-1p-ignore="true"
            data-lpignore="true"
            data-form-type="other"
            spellCheck={false}
            readOnly={!historySearchUnlocked}
            onPointerDown={(event) => { event.currentTarget.readOnly = false; setHistorySearchUnlocked(true); }}
            onKeyDown={(event) => { event.currentTarget.readOnly = false; setHistorySearchUnlocked(true); }}
            onBlur={() => setHistorySearchUnlocked(false)}
            value={historyQuery}
            onChange={(event) => {
              if (event.currentTarget.readOnly) { event.currentTarget.value = historyQuery; return; }
              setHistoryQuery(event.target.value);
            }}
            placeholder="Поиск по чатам"
            aria-label="Поиск по истории чатов"
          />
        )}
        {!collapsed && visibleSessions.map((session) => {
          const relative = formatRelativeTime(session.updatedAt, now);
          return (
            <div key={session.id} className={`sidebar-chat-row ${session.id === currentId ? "is-active" : ""}`}>
            <button
              type="button"
              className={`sidebar-chat ${session.id === currentId ? "is-active" : ""}`}
              onClick={() => onOpenSession(session.id)}
              title={session.title}
            >
              <span className="sidebar-chat-title">{session.title}</span>
              <span className="sidebar-chat-time">{relative}</span>
            </button>
            <button type="button" className="sidebar-chat-delete" disabled={deleteDisabled} aria-label={`Удалить чат «${session.title}»`} title="Удалить чат" onClick={() => void onDeleteSession(session.id)}><IconTrash /></button>
            </div>
          );
        })}
        {!collapsed && historyQuery && visibleSessions.length === 0 && <p className="sidebar-empty">Ничего не найдено</p>}
      </div>
      <div className="sidebar-bottom-nav">
        <button type="button" className={`sidebar-action ${activeWorkspace === "help" ? "is-active" : ""}`} onClick={onHelp} title="Открыть помощь">
          <IconHelp />
          {!collapsed && <span>Помощь</span>}
        </button>
      </div>
      <div className="sidebar-footer">
        {!collapsed && <span className="sidebar-user">{username}</span>}
        {health !== "онлайн" && <span className="sidebar-health-error" title={health} aria-label={health} />}
        <button
          type="button"
          className="icon-btn settings-trigger"
          data-settings-trigger
          onClick={onSettings}
          title={keyWarning ? "Настройки: нет личного ключа" : "Настройки"}
          aria-label={keyWarning ? "Настройки, предупреждение: нет личного ключа" : "Настройки"}
          aria-expanded={settingsOpen}
        >
          <IconSettings />
          {keyWarning && <span className="settings-warn-dot" aria-hidden="true" />}
        </button>
        <button type="button" className="icon-btn" onClick={onLogout} title="Выйти">
          <IconLogout />
        </button>
      </div>
    </aside>
  );
}

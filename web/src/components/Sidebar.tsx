import { useEffect, useState } from "react";
import type { ConversationSummary } from "../types";
import { formatRelativeTime } from "../format";
import { IconCards, IconGraph, IconHelp, IconLibrary, IconLogout, IconPlus, IconSettings, IconSidebar } from "./Icons";

const RELATIVE_TICK_MS = 30_000;

export function Sidebar({
  collapsed,
  onToggle,
  sessions,
  currentId,
  username,
  onNewChat,
  onOpenSession,
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
}: {
  collapsed: boolean;
  onToggle: () => void;
  sessions: ConversationSummary[];
  currentId: string;
  username: string;
  onNewChat: () => void;
  onOpenSession: (id: string) => void;
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
}) {
  const [now, setNow] = useState(() => Date.now());
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
      <button type="button" className="sidebar-action sidebar-action-primary" onClick={onNewChat} title="Новый чат">
        <IconPlus />
        {!collapsed && <span>Новый чат</span>}
      </button>
      <button type="button" className="sidebar-action" onClick={onExplorer} title="Открыть всю базу">
        <IconGraph />
        {!collapsed && <span>Вся база</span>}
      </button>
      <button type="button" className="sidebar-action" onClick={onLibrary} title="Открыть статьи">
        <IconLibrary />
        {!collapsed && <span>Статьи</span>}
      </button>
      {cardsEnabled && <button type="button" className="sidebar-action" onClick={onCards} title="Карточки">
        <IconCards />
        {!collapsed && <span>Карточки</span>}
      </button>}
      <button type="button" className="sidebar-action" onClick={onHelp} title="Открыть справку">
        <IconHelp />
        {!collapsed && <span>Справка</span>}
      </button>
      <div className="sidebar-list" aria-label="История чатов">
        {!collapsed && sessions.length > 0 && <p className="sidebar-section">Недавние</p>}
        {sessions.map((session) => {
          const relative = formatRelativeTime(session.updatedAt, now);
          return (
            <button
              key={session.id}
              type="button"
              className={`sidebar-chat ${session.id === currentId ? "is-active" : ""}`}
              onClick={() => onOpenSession(session.id)}
              title={collapsed ? `${session.title} · ${relative}` : session.title}
            >
              {collapsed ? (
                <span className="chat-initial">{session.title.slice(0, 1)}</span>
              ) : (
                <>
                  <span className="sidebar-chat-title">{session.title}</span>
                  <span className="sidebar-chat-time">{relative}</span>
                </>
              )}
            </button>
          );
        })}
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

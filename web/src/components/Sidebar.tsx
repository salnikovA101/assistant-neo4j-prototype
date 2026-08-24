import type { ConversationSummary } from "../types";
import { IconGraph, IconLogout, IconPlus, IconSidebar } from "./Icons";

export function Sidebar({
  collapsed,
  onToggle,
  sessions,
  currentId,
  username,
  onNewChat,
  onOpenSession,
  onExplorer,
  onLogout,
}: {
  collapsed: boolean;
  onToggle: () => void;
  sessions: ConversationSummary[];
  currentId: string;
  username: string;
  onNewChat: () => void;
  onOpenSession: (id: string) => void;
  onExplorer: () => void;
  onLogout: () => void;
}) {
  return (
    <aside className={`sidebar ${collapsed ? "is-collapsed" : ""}`}>
      <div className="sidebar-top">
        <button type="button" className="icon-btn" onClick={onToggle} title="Сайдбар">
          <IconSidebar />
        </button>
        {!collapsed && <span className="sidebar-brand">Neo4j Assistant</span>}
      </div>
      <button type="button" className="sidebar-action sidebar-action-primary" onClick={onNewChat} title="Новый чат">
        <IconPlus />
        {!collapsed && <span>Новый чат</span>}
      </button>
      <button type="button" className="sidebar-action" onClick={onExplorer} title="Открыть граф базы">
        <IconGraph />
        {!collapsed && <span>Граф базы</span>}
      </button>
      <div className="sidebar-list" aria-label="История чатов">
        {!collapsed && sessions.length > 0 && <p className="sidebar-section">Недавние</p>}
        {sessions.map((session) => (
          <button
            key={session.id}
            type="button"
            className={`sidebar-chat ${session.id === currentId ? "is-active" : ""}`}
            onClick={() => onOpenSession(session.id)}
            title={session.title}
          >
            {collapsed ? <span className="chat-initial">{session.title.slice(0, 1)}</span> : <span>{session.title}</span>}
          </button>
        ))}
      </div>
      <div className="sidebar-footer">
        {!collapsed && <span className="sidebar-user">{username}</span>}
        <button type="button" className="icon-btn" onClick={onLogout} title="Сменить аккаунт">
          <IconLogout />
        </button>
      </div>
    </aside>
  );
}

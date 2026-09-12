from __future__ import annotations

from pathlib import Path

from server.service_guide import load_service_guide

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "src"
SEARCH_PLACEHOLDER = 'placeholder="Поиск сущностей, связей и данных — на английском"'
SEARCH_HINT = "Введите название или термин на английском."


def test_explorer_search_uses_english_graph_language():
    source = (WEB / "components" / "Explorer.tsx").read_text(encoding="utf-8")
    assert SEARCH_PLACEHOLDER in source
    assert SEARCH_HINT not in source
    assert "по английским именам и evidence" in source
    assert "Введите английское название, тип связи или фрагмент данных." in source
    assert "уверенность экстракции ≥" in source
    assert "statusHint=" in source
    assert '<p className="explorer-status">Показано связей:' not in source
    assert "relationLabel" not in source
    assert "showTechnical" not in source
    assert "Начните вводить запрос" not in source
    assert "по всему corpus" not in source


def test_graph_canvas_local_search_uses_english_graph_language():
    source = (WEB / "components" / "GraphCanvas.tsx").read_text(encoding="utf-8")
    assert '"Поиск сущностей, связей и данных — на английском"' in source
    assert '"Поиск на графе · EN"' in source
    assert SEARCH_HINT not in source
    assert "по английским именам и evidence" in source
    assert '? edge.label : ""' in source
    assert 'title: visibleTripletCaption(edge)' in source
    assert "relationLabel" not in source
    assert "Копировать данные" in source
    assert "Найти ребро в этом графе" not in source


def test_workspace_inspector_is_horizontally_resizable_from_current_default():
    graph = (WEB / "components" / "GraphCanvas.tsx").read_text(encoding="utf-8")
    styles = (WEB / "styles.css").read_text(encoding="utf-8")
    assert "DEFAULT_WORKSPACE_INSPECTOR_WIDTH = 420" in graph
    assert 'className="graph-inspector-resize-handle"' in graph
    assert 'aria-label="Изменить ширину правой панели"' in graph
    assert 'aria-orientation="vertical"' in graph
    assert 'event.key === "ArrowLeft"' in graph
    assert 'event.key === "ArrowRight"' in graph
    assert "onDoubleClick" in graph
    assert "--graph-inspector-width" in graph
    desktop_workspace = styles.split("@media (min-width:901px) {", 1)[1].split("}", 1)[0]
    assert "7px var(--graph-inspector-width,420px)" in desktop_workspace


def test_service_guide_manual_search_is_english_names_not_chat():
    guide = load_service_guide(ROOT / "prompts")
    section = guide.split("### Поиск и лимит", 1)[1].split("### ", 1)[0]
    assert "английским именам" in section
    assert "не как вопрос в чате" in section
    assert "`kefir`" in section
    assert "`Lactobacillus`" in section


def test_service_guide_distinguishes_auto_gaps_from_staged_sq_coverage():
    guide = load_service_guide(ROOT / "prompts")
    assert "В режиме **Вопрос по базе** технический раздел **GAPS**" in guide
    assert "В режиме **Исследование** вместо GAPS" in guide
    assert "**Состояние исследовательских вопросов**" in guide
    assert "**Не закрыт**, **Закрыт частично** или **Закрыт**" in guide
    assert "такой вопрос больше не участвует в следующем поиске" in guide
    assert "Закрытие не удаляет связанные цепочки" in guide


def test_agenda_ui_exposes_three_editable_coverage_states():
    drawer = (WEB / "components" / "AgendaDrawer.tsx").read_text(encoding="utf-8")
    types = (WEB / "types.ts").read_text(encoding="utf-8")
    styles = (WEB / "styles.css").read_text(encoding="utf-8")
    assert '"not_closed" | "partial" | "closed"' in types
    assert 'not_closed: "Не закрыт"' in drawer
    assert 'partial: "Закрыт частично"' in drawer
    assert 'closed: "Закрыт"' in drawer
    assert "Object.keys(STATUS_LABEL)" in drawer
    assert 'aria-pressed={item.status === status}' in drawer
    assert 'className="agenda-status"' not in drawer
    assert "agenda-coverage is-${item.status}" in drawer
    assert "Оценил ассистент" in drawer and "Изменено вами" in drawer
    assert ".agenda-coverage.is-closed" in styles
    assert ".agenda-coverage.is-partial" in styles
    assert "locked = false" in drawer
    assert "disabled={busy || locked}" in drawer
    assert ".sq-status-warning" in styles
    assert "Исследовательские вопросы" in drawer


def test_recent_chats_use_relative_time_buckets():
    fmt = (WEB / "format.ts").read_text(encoding="utf-8")
    sidebar = (WEB / "components" / "Sidebar.tsx").read_text(encoding="utf-8")
    styles = (WEB / "styles.css").read_text(encoding="utf-8")
    assert 'if (delta < MINUTE_MS) return "сейчас";' in fmt
    assert "мин" in fmt and " ч" in fmt and " д" in fmt
    assert "formatRelativeTime(session.updatedAt, now)" in sidebar
    assert "const RELATIVE_TICK_MS = 30_000;" in sidebar
    assert "sidebar-chat-title" in sidebar
    assert "sidebar-chat-time" in sidebar
    assert "white-space:normal" in styles
    assert ".sidebar-chat-time" in styles
    assert "overflow-y:auto" in styles
    assert "flex:1 1 0" in styles
    assert "min-height:38px" in styles


def test_single_message_and_card_actions_are_direct():
    chat = (WEB / "components" / "ChatThread.tsx").read_text(encoding="utf-8")
    cards = (WEB / "components" / "CardsWorkspace.tsx").read_text(encoding="utf-8")
    assert "desktop-message-menu" not in chat
    assert 'className="source-action"' in chat
    assert "Данные ответа" in chat
    assert "Новый вариант" in chat
    assert "answer-source-panel" in chat
    assert "more-actions-menu" not in cards
    assert "IconTrash" in cards
    assert "Прикрепить к варианту" not in cards
    assert "attachCard" not in cards
    assert "Вставить в диалог" in cards
    assert '{chatMode && onInsert && <button className="primary-btn"' in cards
    assert 'className="ghost-btn danger-btn"' in cards
    assert ">Удалить</button>" in cards


def test_chat_has_one_cards_entry_and_compact_user_question():
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    composer = (WEB / "components" / "Composer.tsx").read_text(encoding="utf-8")
    graph = (WEB / "components" / "GraphCanvas.tsx").read_text(encoding="utf-8")
    styles = (WEB / "styles.css").read_text(encoding="utf-8")
    assert "onOpenCardTemplates" not in app + composer
    assert "onOpenCardLibrary" not in app + composer
    assert composer.count("onOpenCards()") == 1
    assert "Открыть карточки" in composer
    assert 'openResearch("map")' in app
    assert "GRAPH_PANEL_MIN" in app
    assert "min={GRAPH_PANEL_MIN}" in app
    assert graph.index(">Фильтры") < graph.index(">Результаты")
    assert 'workspaceMode ? "filters"' in graph
    assert "current === \"filters\"" in graph
    assert "RESULT_LIST_LIMIT" in graph
    assert "chunk_id" not in graph
    assert "Скрыть панель" in graph
    assert "Показать панель" in graph
    desktop = styles.split("@media (min-width:681px)", 1)[1].split("@media (max-width:680px)", 1)[0]
    assert ".bubble-user { width:fit-content;" in desktop


def test_collapsed_sidebar_does_not_render_chat_initials():
    sidebar = (WEB / "components" / "Sidebar.tsx").read_text(encoding="utf-8")
    styles = (WEB / "styles.css").read_text(encoding="utf-8")
    assert "chat-initial" not in sidebar
    assert "!collapsed && visibleSessions.map" in sidebar
    assert "sidebar-chat-launcher" in sidebar
    assert "Показать чаты" in sidebar
    assert "sessions.length > 6 || Boolean(historyQuery.trim())" in sidebar
    assert ".sidebar.is-collapsed { width:56px; min-width:56px; max-width:56px;" in styles


def test_search_and_api_key_resist_credential_autofill():
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    sidebar = (WEB / "components" / "Sidebar.tsx").read_text(encoding="utf-8")
    styles = (WEB / "styles.css").read_text(encoding="utf-8")
    assert 'readOnly={!historySearchUnlocked}' in sidebar
    assert 'data-1p-ignore="true"' in sidebar
    assert 'name="chat-history-filter"' in sidebar
    assert 'className="secure-key-input"' in app
    assert 'type="password"' in app
    assert 'autoComplete="new-password"' in app
    assert 'readOnly={!qwenKeyInputUnlocked}' in app
    assert "-webkit-text-security:disc" in styles


def test_message_ids_work_on_plain_http_hosts():
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    uid = app.split("function uid()", 1)[1].split("\n}\n", 1)[0]
    assert "crypto.randomUUID" in uid
    assert "crypto.getRandomValues" in uid
    assert "is not a secure context" in uid


def test_empty_chat_welcome_is_only_help():
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    welcome = (WEB / "components" / "Welcome.tsx").read_text(encoding="utf-8")
    assert "<Welcome onHelp" in app
    assert "Как пользоваться" in welcome
    assert "onClick={onHelp}" in welcome
    assert "<h1" not in welcome
    assert "onPrompt" not in welcome


def test_empty_research_branch_placeholder_is_selectable():
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    pane = (WEB / "components" / "ResearchMapPane.tsx").read_text(encoding="utf-8")
    styles = (WEB / "styles.css").read_text(encoding="utf-8")
    assert "function selectEmptyBranch" in app
    assert "onSelectEmptyBranch={selectEmptyBranch}" in app
    assert "onSelectEmptyBranch(branch)" in pane
    assert 'className="research-empty-node"' in pane
    assert "role=\"treeitem\"" in pane
    assert "cursor:pointer" in styles.split(".research-empty-node {", 1)[1].split("}", 1)[0]


def test_research_step_cards_render_markdown_previews():
    pane = (WEB / "components" / "ResearchMapPane.tsx").read_text(encoding="utf-8")
    styles = (WEB / "styles.css").read_text(encoding="utf-8")
    assert "renderMarkdown" in pane
    assert 'className="md research-step-preview"' in pane
    assert "dangerouslySetInnerHTML" in pane
    assert ".research-step-card > .research-step-preview" in styles
    assert "research-step-card > .research-step-preview h2" in styles.replace("\n", "")
    preview_rule = styles.split(".research-step-card > p,.research-step-card > .research-step-preview {", 1)[1].split("}", 1)[0]
    assert "flex:1 1 auto" in preview_rule
    assert "line-clamp" not in preview_rule
    assert "margin:6px 0 8px" in preview_rule


def test_stream_markdown_holds_incomplete_source_groups():
    from server.tools.source_registry import _SOURCE_GROUP_RE

    fmt = (WEB / "format.ts").read_text(encoding="utf-8")
    assert _SOURCE_GROUP_RE.pattern in fmt
    assert r"\d+[^)]*" in fmt
    assert "function holdIncompleteCitation" in fmt
    assert "holdIncompleteCitation(holdIncompleteFence(text))" in fmt
    assert r"\(\s*source\b" in fmt
    assert r"\d+(?:\s*,\s*\d+)*" not in fmt
    assert r"\d+(?:\s*;\s*source\s*:?\s*\d+)*" not in fmt
    assert "function improveAnswerHtml" in fmt
    assert "HTMLHeadingElement" in fmt
    assert "Пробелы в данных" in fmt


def test_journal_shows_full_thinking():
    source = (WEB / "components" / "ChatThread.tsx").read_text(encoding="utf-8")
    styles = (WEB / "styles.css").read_text(encoding="utf-8")
    assert 'step.kind === "think"' in source
    assert "renderReasoningMarkdown" in source
    assert "Размышление" in source
    thinking_rule = styles.split(".trace-thinking {", 1)[1].split("}", 1)[0]
    assert "max-height" not in thinking_rule


def test_history_can_only_be_deleted_and_approved_stream_can_abort():
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    api = (WEB / "api.ts").read_text(encoding="utf-8")
    vite = (ROOT / "web" / "vite.config.ts").read_text(encoding="utf-8")
    assert "Очистить историю" not in app
    assert "clearHistory" not in app
    assert "/clear_history" not in api
    assert "/clear_history" not in vite
    assert "Удалить чат «${session.title}» без возможности восстановления?" in app
    sidebar = (WEB / "components" / "Sidebar.tsx").read_text(encoding="utf-8")
    assert "onDeleteSession(session.id)" in sidebar
    settings = app.split('className="sidebar-settings-pop"', 1)[1]
    assert "deleteConversation" not in settings
    assert "const controller = streaming ? new AbortController() : null;" in app
    assert "setBusy(true);" in app
    assert "controller?.signal" in app
    assert "signal?: AbortSignal" in api
    assert 'setNotice("Продолжение остановлено.")' in app


def test_missing_qwen_key_shows_warning_not_block():
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    sidebar = (WEB / "components" / "Sidebar.tsx").read_text(encoding="utf-8")
    styles = (WEB / "styles.css").read_text(encoding="utf-8")
    help_ws = (WEB / "components" / "HelpWorkspace.tsx").read_text(encoding="utf-8")
    guide = load_service_guide(ROOT / "prompts")
    assert 'keyWarning={!hasUserKey}' in app
    assert 'settings-key-warning' in app
    assert 'id="qwen-key-warning"' in app
    assert "используется демонстрационный" in app
    assert "Как получить ключ" in app
    assert 'openHelp(QWEN_CLOUD_KEY_HEADING)' in app
    assert "disabled={!hasUserKey}" not in app
    assert "settings-warn-dot" in sidebar
    assert "keyWarning" in sidebar
    assert "нет личного ключа" in sidebar
    assert ".settings-warn-dot" in styles
    assert ".settings-key-warning" in styles
    assert "focusHeading" in help_ws
    assert "## Как подключить ключ QwenCloud" in guide

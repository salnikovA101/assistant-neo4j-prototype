from __future__ import annotations

from pathlib import Path

from server.service_guide import load_service_guide

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web" / "src"
SEARCH_PLACEHOLDER = 'placeholder="Например: kefir, Lactobacillus, GABA, 37 °C"'
SEARCH_HINT = "Имена в базе английские."


def test_explorer_search_uses_english_graph_language():
    source = (WEB / "components" / "Explorer.tsx").read_text(encoding="utf-8")
    assert SEARCH_PLACEHOLDER in source
    assert SEARCH_HINT in source
    assert "по английским именам и цитатам" in source
    assert "Наберите английское имя, фрагмент цитаты или выберите фильтр." in source
    assert "Начните вводить запрос" not in source
    assert "по всему corpus" not in source


def test_graph_canvas_local_search_uses_english_graph_language():
    source = (WEB / "components" / "GraphCanvas.tsx").read_text(encoding="utf-8")
    assert SEARCH_PLACEHOLDER in source
    assert SEARCH_HINT in source
    assert "по английским именам и цитатам" in source
    assert "Найти ребро в этом графе" not in source


def test_service_guide_manual_search_is_english_names_not_chat():
    guide = load_service_guide(ROOT / "prompts")
    section = guide.split("### Поиск и лимит", 1)[1].split("### ", 1)[0]
    assert "английским именам" in section
    assert "не как вопрос в чате" in section
    assert "`kefir`" in section
    assert "`Lactobacillus`" in section


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


def test_empty_chat_welcome_is_short_help_not_suggestion_grid():
    app = (WEB / "App.tsx").read_text(encoding="utf-8")
    styles = (WEB / "styles.css").read_text(encoding="utf-8")
    assert 'className="welcome"' in app
    assert "Как пользоваться" in app
    assert 'setWorkspace("help")' in app
    assert 'className="suggestions"' not in app
    assert ".welcome-help-link" in styles
    welcome = app.split('className="welcome"', 1)[1].split("</div>", 1)[0]
    assert "подключённой базе знаний" not in welcome
    assert "Подбери культуры для творога" not in welcome
    assert "или спросите у ассистента" in welcome
    assert welcome.count("<p>") == 1


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
    assert "Удалить этот чат без возможности восстановления?" in app
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

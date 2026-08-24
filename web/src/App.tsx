import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  adoptSessionId,
  bindAccount,
  clearHistory,
  createConversation,
  deleteConversation,
  fetchConversation,
  fetchConversations,
  fetchHealth,
  fetchMe,
  fetchUiConfig,
  getLlmKey,
  getSessionId,
  logout,
  setLlmKey,
  streamBody,
  withHeaders,
} from "./api";
import { ChatThread } from "./components/ChatThread";
import { Composer } from "./components/Composer";
import { Explorer } from "./components/Explorer";
import { GraphPane } from "./components/GraphPane";
import { IconSettings } from "./components/Icons";
import { Sidebar } from "./components/Sidebar";
import { parseSseBlock } from "./format";
import { clearLegacySessions } from "./sessions";
import type { ChatMessage, ChatStep, ConversationSummary, SearchDepth, UiConfig } from "./types";

function uid(): string {
  return crypto.randomUUID();
}

type PanelSide = "sidebar" | "graph";

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function PanelResizer({
  side,
  value,
  min,
  max,
  onChange,
}: {
  side: PanelSide;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  const dragRef = useRef<{ x: number; width: number } | null>(null);
  const label = side === "sidebar" ? "Ширина боковой панели" : "Ширина графовой панели";

  const update = (clientX: number) => {
    const drag = dragRef.current;
    if (!drag) return;
    const delta = clientX - drag.x;
    onChange(clamp(drag.width + (side === "sidebar" ? delta : -delta), min, max));
  };

  const onPointerDown = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    dragRef.current = { x: event.clientX, width: value };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const stopDragging = (event: ReactPointerEvent<HTMLDivElement>) => {
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const onKeyDown = (event: ReactKeyboardEvent<HTMLDivElement>) => {
    const step = event.shiftKey ? 40 : 12;
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      onChange(clamp(value + (side === "sidebar" ? -step : step), min, max));
    }
    if (event.key === "ArrowRight") {
      event.preventDefault();
      onChange(clamp(value + (side === "sidebar" ? step : -step), min, max));
    }
  };

  return (
    <div
      className={`panel-resizer panel-resizer-${side}`}
      role="separator"
      aria-label={label}
      aria-orientation="vertical"
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={Math.round(value)}
      tabIndex={0}
      onPointerDown={onPointerDown}
      onPointerMove={(event) => update(event.clientX)}
      onPointerUp={stopDragging}
      onPointerCancel={stopDragging}
      onKeyDown={onKeyDown}
    />
  );
}

export function App() {
  const [config, setConfig] = useState<UiConfig | null>(null);
  const [health, setHealth] = useState("…");
  const [collapsed, setCollapsed] = useState(() => window.innerWidth <= 680);
  const [sidebarWidth, setSidebarWidth] = useState(252);
  const [graphWidth, setGraphWidth] = useState(520);
  const [sessions, setSessions] = useState<ConversationSummary[]>([]);
  const [currentId, setCurrentId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [depth, setDepth] = useState<SearchDepth>("medium");
  const [effort, setEffort] = useState("");
  const [profile, setProfile] = useState("");
  const [graphRunId, setGraphRunId] = useState<string | null>(null);
  const [explorerOpen, setExplorerOpen] = useState(false);
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [keyDraft, setKeyDraft] = useState("");
  const [recording, setRecording] = useState(false);
  const [notice, setNotice] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const settingsRef = useRef<HTMLDivElement>(null);

  const current = sessions.find((item) => item.id === currentId);
  const model = config?.models.find((item) => item.id === profile);
  const effortOptions =
    model?.reasoning_effort_options || config?.reasoning_effort_options || [];
  const maxGraphWidth = Math.max(
    360,
    window.innerWidth - (collapsed ? 64 : sidebarWidth) - 320
  );

  useEffect(() => {
    Promise.all([fetchUiConfig(), fetchMe(), fetchConversations()])
      .then(([cfg, account, history]) => {
        setConfig(cfg);
        bindAccount(account.id);
        clearLegacySessions();
        setSessions(history);
        const storedId = getSessionId();
        const target = history.some((item) => item.id === storedId)
          ? storedId
          : history[0]?.id || "";
        if (target) {
          adoptSessionId(target);
          setCurrentId(target);
        }
        const storedDepth = localStorage.getItem("search_depth") as SearchDepth | null;
        setDepth(
          storedDepth && cfg.search_depth_options.includes(storedDepth)
            ? storedDepth
            : cfg.search_depth || "medium"
        );
        const storedProfile = localStorage.getItem("llm_profile") || cfg.current_profile;
        const found = cfg.models.find((item) => item.id === storedProfile) || cfg.models[0];
        if (found) {
          setProfile(found.id);
          const storedEffort = localStorage.getItem("reasoning_effort");
          const next =
            storedEffort && found.reasoning_effort_options.includes(storedEffort)
              ? storedEffort
              : found.reasoning_effort || cfg.reasoning_effort;
          setEffort(next);
          setKeyDraft(getLlmKey(found.id));
        }
      })
      .catch(() => setHealth("нет связи"));
    fetchHealth()
      .then((body) => {
        const ok = body.status === "ready" || body.status === "ok";
        setHealth(ok ? "онлайн" : "деградация");
      })
      .catch(() => setHealth("нет связи"));
  }, []);

  useEffect(() => {
    setGraphWidth((value) => Math.min(value, maxGraphWidth));
  }, [maxGraphWidth]);

  useEffect(() => {
    if (!currentId) {
      setMessages([]);
      return;
    }
    let cancelled = false;
    fetchConversation(currentId)
      .then((detail) => {
        if (!cancelled) setMessages(detail.messages);
      })
      .catch((err) => {
        if (!cancelled) setNotice(err instanceof Error ? err.message : "Не удалось открыть чат");
      });
    return () => {
      cancelled = true;
    };
  }, [currentId]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (explorerOpen) {
        setExplorerOpen(false);
        return;
      }
      if (settingsOpen) {
        setSettingsOpen(false);
        return;
      }
      if (graphRunId) setGraphRunId(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [explorerOpen, graphRunId, settingsOpen]);

  useEffect(() => {
    const syncSidebar = () => {
      if (window.innerWidth <= 680) setCollapsed(true);
    };
    window.addEventListener("resize", syncSidebar);
    return () => window.removeEventListener("resize", syncSidebar);
  }, []);

  useEffect(() => {
    const onPointerDown = (event: MouseEvent) => {
      if (settingsRef.current && !settingsRef.current.contains(event.target as Node)) {
        setSettingsOpen(false);
      }
    };
    window.addEventListener("mousedown", onPointerDown);
    return () => window.removeEventListener("mousedown", onPointerDown);
  }, []);

  useEffect(() => {
    if (!notice) return;
    const timeout = window.setTimeout(() => setNotice(""), 4200);
    return () => window.clearTimeout(timeout);
  }, [notice]);

  async function refreshSessions() {
    const history = await fetchConversations();
    setSessions(history);
  }

  async function newChat() {
    abortRef.current?.abort();
    try {
      const created = await createConversation();
      adoptSessionId(created.id);
      setSessions((prev) => [created, ...prev]);
      setCurrentId(created.id);
      setMessages([]);
      setGraphRunId(null);
      setDraft("");
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Не удалось создать чат");
    }
  }

  function openSession(id: string) {
    abortRef.current?.abort();
    adoptSessionId(id);
    setCurrentId(id);
    setGraphRunId(null);
  }

  function patchAssistant(id: string, patch: Partial<ChatMessage>) {
    setMessages((prev) => {
      return prev.map((msg) => (msg.id === id ? { ...msg, ...patch } : msg));
    });
  }

  async function send(text = draft) {
    const value = text.trim();
    if (!value || busy) return;
    let conversationId = currentId;
    if (!conversationId) {
      try {
        const created = await createConversation();
        conversationId = created.id;
        adoptSessionId(created.id);
        setCurrentId(created.id);
        setSessions((prev) => [created, ...prev]);
      } catch (err) {
        setNotice(err instanceof Error ? err.message : "Не удалось создать чат");
        return;
      }
    }
    setDraft("");
    const user: ChatMessage = { id: uid(), role: "user", text: value };
    const assistantId = uid();
    const assistant: ChatMessage = {
      id: assistantId,
      role: "assistant",
      text: "",
      thinking: "",
      tools: [],
      steps: [],
      status: "streaming",
    };
    const started = Date.now();
    const next = [...messages, user, assistant];
    setMessages(next);
    setSessions((prev) =>
      prev.map((item) =>
        item.id === conversationId && item.title === "Новый чат"
          ? { ...item, title: value.split("\n")[0].slice(0, 42), updatedAt: Date.now() }
          : item
      )
    );
    setBusy(true);
    const ac = new AbortController();
    abortRef.current = ac;
    try {
      const res = await fetch("/process_text_stream", {
        method: "POST",
        headers: (() => {
          adoptSessionId(conversationId);
          return withHeaders(profile, {
          "Content-Type": "application/json",
          Accept: "text/event-stream",
          });
        })(),
        body: streamBody(value, {
          search_depth: depth,
          reasoning_effort: effort || undefined,
          profile: profile || undefined,
          turn_id: uid(),
        }),
        signal: ac.signal,
      });
      if (!res.ok || !res.body) {
        if (res.status === 401) {
          window.location.assign("/login");
          return;
        }
        let err = "Ошибка сервера";
        try {
          const body = await res.json();
          err = body.error || err;
        } catch {
          /* keep */
        }
        patchAssistant(assistantId, { text: err, status: "error" });
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let answer = "";
      let thinking = "";
      let tools = [...(assistant.tools || [])];
      let steps: ChatStep[] = [...(assistant.steps || [])];
      const elapsed = () => Math.max(1, Math.round((Date.now() - started) / 1000));
      const appendThink = (delta: string) => {
        thinking += delta;
        const last = steps[steps.length - 1];
        if (last && last.kind === "think") {
          steps = [...steps.slice(0, -1), { kind: "think", text: last.text + delta }];
        } else {
          steps = [...steps, { kind: "think", text: delta }];
        }
      };
      const flush = (
        status: ChatMessage["status"] = "streaming",
        extra: Partial<ChatMessage> = {}
      ) => {
        patchAssistant(assistantId, {
          text: answer,
          thinking,
          tools,
          steps,
          status,
          elapsedSec: elapsed(),
          ...extra,
        });
      };
      const consume = (raw: string) => {
        const parsed = parseSseBlock(raw);
        if (!parsed) return false;
        const { event, data } = parsed;
        if (event === "thinking") appendThink(String(data.delta || ""));
        else if (event === "content") answer += String(data.delta || "");
        else if (event === "content_rewind") {
          const rewind = String(data.text || "");
          if (rewind && answer.endsWith(rewind)) answer = answer.slice(0, -rewind.length);
        } else if (event === "tool_call") {
          const id = String(data.id || `t${tools.length}`);
          const card = {
            id,
            name: String(data.name || "ask_subgraph"),
            status: "running" as const,
            args: data.arguments ?? data.args,
          };
          tools = [...tools.filter((item) => item.id !== id), card];
          steps = [...steps.filter((item) => item.kind !== "tool" || item.id !== id), { kind: "tool", ...card }];
        } else if (event === "tool_result") {
          const id = String(data.id || tools.at(-1)?.id || "tool");
          const status = data.ok === false ? "error" : "done";
          const result = String(data.result ?? data.preview ?? "");
          tools = tools.map((item) =>
            item.id === id ? { ...item, status, result } : item
          );
          steps = steps.map((item) =>
            item.kind === "tool" && item.id === id ? { ...item, status, result } : item
          );
        } else if (event === "done") {
          if (data.final_content) answer = String(data.final_content);
          const runId = data.graph_run_id ? String(data.graph_run_id) : "";
          const chains = Number(data.graph_chain_count) || 0;
          flush("done", {
            graphRunId: runId || undefined,
            graphChainCount: chains || undefined,
          });
          if (runId && chains > 0) setGraphRunId(runId);
          return true;
        } else if (event === "error") {
          flush("error", { text: String(data.message || "Ошибка стрима") });
          return true;
        }
        flush("streaming");
        return false;
      };
      while (true) {
        const { done, value: chunk } = await reader.read();
        if (done) {
          buf += decoder.decode();
          break;
        }
        buf += decoder.decode(chunk, { stream: true });
        // EventSource allows CRLF as well as LF separators. Normalize chunks so
        // an upstream proxy cannot leave the UI waiting for a completed event.
        buf = buf.replace(/\r\n/g, "\n");
        let sep;
        let stop = false;
        while ((sep = buf.indexOf("\n\n")) !== -1) {
          const raw = buf.slice(0, sep);
          buf = buf.slice(sep + 2);
          if (consume(raw)) {
            stop = true;
            break;
          }
        }
        if (stop) return;
      }
      if (buf.trim()) consume(buf);
      if (answer.trim()) flush("done");
      else flush("error", { text: "Поток оборвался" });
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        patchAssistant(assistantId, { status: "aborted" });
      } else {
        patchAssistant(assistantId, {
          status: "error",
          text: "Ошибка соединения с сервером",
        });
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
      void refreshSessions();
    }
  }

  async function onMic() {
    if (!config?.audio_enabled) return;
    if (recording) {
      mediaRef.current?.stop();
      setRecording(false);
      return;
    }
    if (busy) abortRef.current?.abort();
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setNotice("Не удалось получить доступ к микрофону. Проверьте разрешение браузера.");
      return;
    }
    const rec = new MediaRecorder(stream);
    const chunks: Blob[] = [];
    rec.ondataavailable = (ev) => {
      if (ev.data.size) chunks.push(ev.data);
    };
    rec.onstop = async () => {
      stream.getTracks().forEach((track) => track.stop());
      const blob = new Blob(chunks, { type: "audio/webm" });
      try {
        const res = await fetch("/stt", {
          method: "POST",
          headers: withHeaders(profile),
          body: blob,
        });
        if (!res.ok) throw new Error();
        const data = await res.json();
        if (data.text) await send(String(data.text));
        else setNotice("Не удалось распознать речь. Попробуйте ещё раз.");
      } catch {
        setNotice("Не удалось распознать речь. Проверьте соединение и повторите попытку.");
      }
    };
    mediaRef.current = rec;
    rec.start();
    setRecording(true);
  }

  const empty = messages.length === 0;

  return (
    <div
      className="app-root"
      style={
        {
          "--sidebar-width": `${sidebarWidth}px`,
          "--graph-width": `${graphWidth}px`,
        } as CSSProperties
      }
    >
      <Sidebar
        collapsed={collapsed}
        onToggle={() => setCollapsed((v) => !v)}
        sessions={sessions}
        currentId={currentId}
        username={config?.username || "demo"}
        onNewChat={() => void newChat()}
        onOpenSession={openSession}
        onExplorer={() => setExplorerOpen(true)}
        onLogout={() => void logout()}
      />
      {!collapsed && (
        <PanelResizer
          side="sidebar"
          value={sidebarWidth}
          min={196}
          max={420}
          onChange={setSidebarWidth}
        />
      )}
      <div className="main-col">
        <header className="topbar">
          <div className="topbar-title">{current?.title || "Новый чат"}</div>
          <div className="topbar-right">
            <span className={`health ${health === "онлайн" ? "is-ok" : ""}`}><i />{health}</span>
            <div className="settings-wrap" ref={settingsRef}>
              <button
                type="button"
                className="icon-btn"
                onClick={() => setSettingsOpen((v) => !v)}
                aria-label="Настройки"
              >
                <IconSettings />
              </button>
              {settingsOpen && (
                <div className="settings-pop">
                  <p className="settings-title">Подключение</p>
                  <p className="settings-hint">Ключ хранится только в этой вкладке браузера.</p>
                  <input
                    type="password"
                    value={keyDraft}
                    onChange={(e) => setKeyDraft(e.target.value)}
                    placeholder="из конфига сервера"
                  />
                  <button
                    type="button"
                    className="ghost-btn"
                    onClick={() => {
                      setLlmKey(profile, keyDraft);
                      setSettingsOpen(false);
                    }}
                  >
                    Сохранить
                  </button>
                  <button
                    type="button"
                    className="ghost-btn"
                    disabled={!currentId}
                    onClick={async () => {
                      try {
                        await clearHistory();
                        setMessages([]);
                        setGraphRunId(null);
                        await refreshSessions();
                        setNotice("История текущего чата очищена.");
                      } catch (err) {
                        setNotice(err instanceof Error ? err.message : "Не удалось очистить историю.");
                      }
                    }}
                  >
                    Очистить историю
                  </button>
                  <button
                    type="button"
                    className="ghost-btn"
                    disabled={!currentId}
                    onClick={async () => {
                      if (!currentId || !window.confirm("Удалить этот чат без возможности восстановления?")) return;
                      try {
                        await deleteConversation(currentId);
                        const remaining = sessions.filter((item) => item.id !== currentId);
                        setSessions(remaining);
                        const nextId = remaining[0]?.id || "";
                        if (nextId) adoptSessionId(nextId);
                        else sessionStorage.removeItem("neo4j-assistant.session-id");
                        setCurrentId(nextId);
                        setMessages([]);
                        setGraphRunId(null);
                        setSettingsOpen(false);
                      } catch (err) {
                        setNotice(err instanceof Error ? err.message : "Не удалось удалить чат");
                      }
                    }}
                  >
                    Удалить чат
                  </button>
                </div>
              )}
            </div>
          </div>
        </header>
        <div className={`chat-col ${empty ? "is-empty" : ""} ${graphRunId ? "with-graph" : ""}`}>
          <ChatThread
            messages={messages}
            openGraphId={graphRunId}
            onOpenGraph={(runId) => setGraphRunId(runId)}
          />
          <Composer
            text={draft}
            onText={setDraft}
            onSubmit={() => void send()}
            onStop={() => abortRef.current?.abort()}
            busy={busy}
            audioEnabled={Boolean(config?.audio_enabled)}
            onMic={() => void onMic()}
            recording={recording}
            depth={depth}
            onDepth={(value) => {
              setDepth(value);
              localStorage.setItem("search_depth", value);
            }}
            effort={effort}
            effortOptions={effortOptions}
            onEffort={(value) => {
              setEffort(value);
              localStorage.setItem("reasoning_effort", value);
            }}
            profile={profile}
            models={config?.models || []}
            onProfile={(id) => {
              setProfile(id);
              localStorage.setItem("llm_profile", id);
              const found = config?.models.find((item) => item.id === id);
              if (found?.reasoning_effort) setEffort(found.reasoning_effort);
              setKeyDraft(getLlmKey(id));
            }}
            centered={empty}
          />
        </div>
      </div>
      {graphRunId && (
        <PanelResizer
          side="graph"
          value={graphWidth}
          min={360}
          max={maxGraphWidth}
          onChange={setGraphWidth}
        />
      )}
      {graphRunId && <GraphPane runId={graphRunId} onClose={() => setGraphRunId(null)} />}
      {explorerOpen && (
        <div className="explorer-overlay">
          <Explorer onClose={() => setExplorerOpen(false)} />
        </div>
      )}
      {notice && <div className="toast" role="status">{notice}</div>}
    </div>
  );
}

import {
  lazy,
  Suspense,
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  adoptSessionId,
  agendaEvent,
  bindAccount,
  clearHistory,
  createConversation,
  deleteConversation,
  fetchConversation,
  fetchConversations,
  fetchCardTemplates,
  fetchHealth,
  fetchMe,
  fetchUiConfig,
  forkConversation,
  generateCardDraft,
  getLlmKey,
  getSessionId,
  logout,
  resolveApproval,
  saveCardDraft,
  setLlmKey,
  streamBody,
  withHeaders,
} from "./api";
import { ChatThread } from "./components/ChatThread";
import { Composer } from "./components/Composer";
import { AgendaDrawer } from "./components/AgendaDrawer";
import { IconSettings } from "./components/Icons";
import { Sidebar } from "./components/Sidebar";
import { parseSseBlock } from "./format";
import { clearLegacySessions } from "./sessions";
import type {
  AgendaItem,
  Branch,
  CardTemplate,
  ChatMessage,
  ChatStep,
  ConversationDetail,
  ConversationSummary,
  PendingApproval,
  SearchDepth,
  UiConfig,
} from "./types";

const Explorer = lazy(() => import("./components/Explorer").then((module) => ({ default: module.Explorer })));
const GraphPane = lazy(() => import("./components/GraphPane").then((module) => ({ default: module.GraphPane })));
const CardsWorkspace = lazy(() => import("./components/CardsWorkspace").then((module) => ({ default: module.CardsWorkspace })));

function uid(): string {
  return crypto.randomUUID();
}

type PanelSide = "sidebar" | "graph";
type Workspace = "chat" | "graph" | "cards";

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
  const [branches, setBranches] = useState<Branch[]>([]);
  const [branchId, setBranchId] = useState("");
  const [headCheckpointId, setHeadCheckpointId] = useState("");
  const [agenda, setAgenda] = useState<AgendaItem[]>([]);
  const [pendingApproval, setPendingApproval] = useState<PendingApproval | null>(null);
  const [approvalBusy, setApprovalBusy] = useState(false);
  const [cardBusy, setCardBusy] = useState(false);
  const [cardTemplates, setCardTemplates] = useState<CardTemplate[]>([]);
  const [agendaOpen, setAgendaOpen] = useState(false);
  const [checkpointGraphId, setCheckpointGraphId] = useState("");
  const [workspace, setWorkspace] = useState<Workspace>("chat");
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [depth, setDepth] = useState<SearchDepth>("medium");
  const [effort, setEffort] = useState("");
  const [profile, setProfile] = useState("");
  const [mode, setMode] = useState<"auto" | "staged">(
    () => localStorage.getItem("retrieval_mode") === "staged" ? "staged" : "auto"
  );
  const [graphRunId, setGraphRunId] = useState<string | null>(null);
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

  function applyDetail(detail: ConversationDetail) {
    setMessages(detail.messages);
    setBranches(detail.branches || []);
    setBranchId(detail.activeBranchId || "");
    setHeadCheckpointId(detail.headCheckpointId || "");
    setAgenda(detail.agenda || []);
    setPendingApproval(detail.pendingApproval || null);
  }

  useEffect(() => {
    Promise.all([fetchUiConfig(), fetchMe(), fetchConversations()])
      .then(([cfg, account, history]) => {
        setConfig(cfg);
        if (cfg.cards_enabled) {
          void fetchCardTemplates().then(setCardTemplates).catch(() => setCardTemplates([]));
        }
        if (!cfg.staged_enabled) setMode("auto");
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
    fetchConversation(currentId, branchId)
      .then((detail) => {
        if (!cancelled) applyDetail(detail);
      })
      .catch((err) => {
        if (!cancelled) setNotice(err instanceof Error ? err.message : "Не удалось открыть чат");
      });
    return () => {
      cancelled = true;
    };
  }, [currentId, branchId]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (workspace !== "chat") return;
      if (settingsOpen) {
        setSettingsOpen(false);
        return;
      }
      if (graphRunId) setGraphRunId(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [workspace, graphRunId, settingsOpen]);

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
      setBranchId(created.activeBranchId || "");
      setHeadCheckpointId(created.headCheckpointId || "");
      setBranches([]);
      setAgenda([]);
      setPendingApproval(null);
      setMessages([]);
      setGraphRunId(null);
      setDraft("");
      setWorkspace("chat");
    } catch (err) {
      setNotice(err instanceof Error ? err.message : "Не удалось создать чат");
    }
  }

  function openSession(id: string) {
    abortRef.current?.abort();
    adoptSessionId(id);
    setCurrentId(id);
    setBranchId("");
    setGraphRunId(null);
    setCheckpointGraphId("");
    setWorkspace("chat");
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
    let activeBranchId = branchId;
    let baseCheckpointId = headCheckpointId;
    if (!conversationId) {
      try {
        const created = await createConversation();
        conversationId = created.id;
        adoptSessionId(created.id);
        setCurrentId(created.id);
        setSessions((prev) => [created, ...prev]);
        activeBranchId = created.activeBranchId || "";
        baseCheckpointId = created.headCheckpointId || "";
        setBranchId(activeBranchId);
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
      const endpoint = activeBranchId
        ? `/api/conversations/${encodeURIComponent(conversationId)}/branches/${encodeURIComponent(activeBranchId)}/turns`
        : "/process_text_stream";
      const res = await fetch(endpoint, {
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
          mode,
          branch_id: activeBranchId || undefined,
          base_checkpoint_id: baseCheckpointId || undefined,
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
        } else if (event === "approval_required") {
          const approval = data.approval as PendingApproval;
          setPendingApproval(approval);
          flush("waiting_approval");
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
      if (conversationId) {
        void fetchConversation(conversationId, activeBranchId).then(applyDetail).catch(() => undefined);
      }
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

  async function handleApproval(
    action: "approve" | "revise",
    subquestions: string[],
    feedback = ""
  ) {
    if (!pendingApproval || approvalBusy) return;
    setApprovalBusy(true);
    try {
      const res = await resolveApproval(pendingApproval, action, subquestions, feedback);
      // The server has claimed the approval synchronously. Hide the form before
      // consuming the long-running retrieval/LLM SSE stream so it cannot remain
      // in the thread while the same assistant message continues streaming.
      if (action === "approve") setPendingApproval(null);
      if (res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        const assistantId = pendingApproval.assistantMessageId;
        const existing = messages.find((message) => message.id === assistantId);
        let answer = existing?.text || "";
        let thinking = existing?.thinking || "";
        let steps = [...(existing?.steps || [])];
        while (true) {
          const { done, value } = await reader.read();
          buffer += decoder.decode(value || new Uint8Array(), { stream: !done }).replace(/\r\n/g, "\n");
          let separator = buffer.indexOf("\n\n");
          while (separator !== -1) {
            const parsed = parseSseBlock(buffer.slice(0, separator));
            buffer = buffer.slice(separator + 2);
            if (parsed?.event === "thinking") {
              const delta = String(parsed.data.delta || "");
              thinking += delta;
              const last = steps[steps.length - 1];
              if (last?.kind === "think") steps = [...steps.slice(0, -1), { kind: "think", text: last.text + delta }];
              else steps = [...steps, { kind: "think", text: delta }];
            }
            if (parsed?.event === "content") answer += String(parsed.data.delta || "");
            if (parsed?.event === "done") answer = String(parsed.data.final_content || answer);
            if (parsed?.event === "approval_required") {
              setPendingApproval(parsed.data.approval as PendingApproval);
            }
            if (parsed?.event === "error") setNotice(String(parsed.data.message || "Ошибка продолжения"));
            patchAssistant(assistantId, { text: answer, thinking, steps, status: parsed?.event === "approval_required" ? "waiting_approval" : "streaming" });
            separator = buffer.indexOf("\n\n");
          }
          if (done) break;
        }
      }
      const detail = await fetchConversation(currentId, branchId);
      applyDetail(detail);
      if (action === "approve") {
        setPendingApproval(null);
        if (detail.headCheckpointId) setCheckpointGraphId(detail.headCheckpointId);
      }
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Не удалось продолжить ответ");
    } finally {
      setApprovalBusy(false);
    }
  }

  async function mutateAgenda(
    action: "add" | "close" | "reopen",
    input: { sq_id?: string; text?: string }
  ) {
    if (!branchId || !headCheckpointId) return;
    if (pendingApproval) {
      setNotice("Сначала подтвердите или отклоните текущий план поиска.");
      return;
    }
    try {
      const result = await agendaEvent(branchId, headCheckpointId, action, input);
      setAgenda(result.agenda);
      setHeadCheckpointId(result.checkpointId);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Не удалось изменить SQ");
      const detail = await fetchConversation(currentId, branchId);
      applyDetail(detail);
    }
  }

  async function handleGenerateCard(templateVersionId: string) {
    if (!headCheckpointId || !currentId || cardBusy || busy) return;
    setCardBusy(true);
    try {
      const result = await generateCardDraft(
        headCheckpointId,
        templateVersionId,
        profile,
        effort
      );
      setHeadCheckpointId(result.checkpointId);
      applyDetail(await fetchConversation(currentId, branchId));
      setNotice("Карточка сформирована как draft в текущей ветке.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Не удалось сформировать карточку");
    } finally {
      setCardBusy(false);
    }
  }

  async function handleSaveCard(draftId: string, title: string) {
    try {
      await saveCardDraft(draftId, title);
      applyDetail(await fetchConversation(currentId, branchId));
      setNotice("Карточка сохранена в библиотеку.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Не удалось сохранить карточку");
    }
  }

  async function forkFrom(checkpointId: string) {
    if (!currentId || !checkpointId) return;
    try {
      const branch = await forkConversation(currentId, checkpointId);
      setBranches((items) => [...items, branch]);
      setBranchId(branch.id);
      setHeadCheckpointId(checkpointId);
      setWorkspace("chat");
      setNotice(`Создана ветка «${branch.name}».`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Не удалось создать ветку");
    }
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
        currentId={workspace === "chat" ? currentId : ""}
        username={config?.username || "demo"}
        onNewChat={() => void newChat()}
        onOpenSession={openSession}
        onExplorer={() => setWorkspace("graph")}
        onCards={() => setWorkspace("cards")}
        cardsEnabled={config?.cards_enabled !== false}
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
          <div className="topbar-title">
            <span>{workspace === "graph" ? "Граф базы" : workspace === "cards" ? "Карточки" : current?.title || "Новый чат"}</span>
            {workspace === "chat" && branches.length > 0 && (
              <select value={branchId} onChange={(event) => setBranchId(event.target.value)} aria-label="Ветка чата">
                {branches.map((branch) => <option key={branch.id} value={branch.id}>{branch.name}</option>)}
              </select>
            )}
          </div>
          <div className="topbar-right">
            {workspace === "chat" && (
              <button type="button" className="sq-button" onClick={() => setAgendaOpen((value) => !value)}>
                SQ {agenda.filter((item) => item.status === "open").length}
              </button>
            )}
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
        {workspace === "chat" ? <div className={`chat-col ${empty ? "is-empty" : ""} ${graphRunId || checkpointGraphId ? "with-graph" : ""}`}>
          <ChatThread
            messages={messages}
            openGraphId={graphRunId}
            onOpenGraph={(runId) => setGraphRunId(runId)}
            pendingApproval={pendingApproval}
            approvalBusy={approvalBusy}
            onResolveApproval={(action, sqs, feedback) => void handleApproval(action, sqs, feedback)}
            onCheckpoint={(checkpointId) => {
              setCheckpointGraphId(checkpointId);
              setGraphRunId(null);
            }}
            onFork={(checkpointId) => void forkFrom(checkpointId)}
            onSaveCard={(draftId, title) => void handleSaveCard(draftId, title)}
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
            mode={mode}
            onMode={(value) => {
              setMode(value);
              localStorage.setItem("retrieval_mode", value);
            }}
            stagedEnabled={config?.staged_enabled !== false}
            cardTemplates={cardTemplates}
            cardBusy={cardBusy}
            cardEnabled={Boolean(headCheckpointId && currentId && !pendingApproval && !busy)}
            onGenerateCard={(templateVersionId) => void handleGenerateCard(templateVersionId)}
          />
        </div> : workspace === "graph" ? (
          <Suspense fallback={<p className="explorer-status">Загрузка Graph Workspace…</p>}>
            <Explorer />
          </Suspense>
        ) : (
          <Suspense fallback={<p className="explorer-status">Загрузка карточек…</p>}>
            <CardsWorkspace
              checkpointId={headCheckpointId}
              branchId={branchId}
              onCheckpoint={setHeadCheckpointId}
              onNotice={setNotice}
            />
          </Suspense>
        )}
      </div>
      {workspace === "chat" && (graphRunId || checkpointGraphId) && (
        <PanelResizer
          side="graph"
          value={graphWidth}
          min={360}
          max={maxGraphWidth}
          onChange={setGraphWidth}
        />
      )}
      {workspace === "chat" && (graphRunId || checkpointGraphId) && (
        <Suspense fallback={<aside className="graph-pane"><p className="explorer-status">Загрузка графа…</p></aside>}>
          <GraphPane
            runId={graphRunId || undefined}
            checkpointId={checkpointGraphId || undefined}
            onClose={() => { setGraphRunId(null); setCheckpointGraphId(""); }}
          />
        </Suspense>
      )}
      {agendaOpen && workspace === "chat" && (
        <AgendaDrawer
          agenda={agenda}
          onClose={() => setAgendaOpen(false)}
          onAdd={(text) => mutateAgenda("add", { text })}
          onToggle={(item) => mutateAgenda(item.status === "open" ? "close" : "reopen", { sq_id: item.id })}
        />
      )}
      {notice && <div className="toast" role="status">{notice}</div>}
    </div>
  );
}

import {
  lazy,
  Suspense,
  useEffect,
  useRef,
  useState,
  type ComponentType,
  type CSSProperties,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  adoptSessionId,
  agendaEvent,
  bindAccount,
  createConversation,
  deleteConversation,
  fetchConversation,
  fetchConversations,
  fetchCardTemplates,
  fetchHealth,
  fetchMe,
  fetchResearchMap,
  fetchUiConfig,
  forkConversation,
  getLlmKey,
  getSessionId,
  logout,
  insertCardMessage,
  renameBranch,
  resolveApproval,
  saveCardDraft,
  setLlmKey,
  streamBody,
  updateCardDraft,
  withHeaders,
} from "./api";
import { branchColor } from "./branchVisuals";
import { ChatThread } from "./components/ChatThread";
import { Composer } from "./components/Composer";
import { AgendaDrawer } from "./components/AgendaDrawer";
import { BranchMenu } from "./components/BranchMenu";
import { ResearchPanelShell, type ResearchTab } from "./components/ResearchPanelShell";
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
  ResearchStep,
  ResearchBranch,
  SearchDepth,
  SavedCard,
  TurnFailure,
  UiConfig,
} from "./types";

function lazyWithChunkReload<T extends ComponentType<any>>(
  loader: () => Promise<{ default: T }>,
  chunkName: string,
) {
  return lazy(async () => {
    try {
      const module = await loader();
      sessionStorage.removeItem(`neo4j-chunk-reload:${chunkName}`);
      return module;
    } catch (error) {
      // A running tab can still hold an older hashed entrypoint after the
      // container is rebuilt. Reload once so it picks up the matching chunks
      // instead of leaving React with an unhandled lazy-import error.
      const key = `neo4j-chunk-reload:${chunkName}`;
      if (!sessionStorage.getItem(key)) {
        sessionStorage.setItem(key, "1");
        window.location.reload();
      }
      throw error;
    }
  });
}

const Explorer = lazyWithChunkReload(() => import("./components/Explorer").then((module) => ({ default: module.Explorer })), "explorer");
const GraphPane = lazyWithChunkReload(() => import("./components/GraphPane").then((module) => ({ default: module.GraphPane })), "graph-pane");
const ResearchMapPane = lazyWithChunkReload(() => import("./components/ResearchMapPane").then((module) => ({ default: module.ResearchMapPane })), "research-map");
const CardsWorkspace = lazyWithChunkReload(() => import("./components/CardsWorkspace").then((module) => ({ default: module.CardsWorkspace })), "cards");
const LibraryWorkspace = lazyWithChunkReload(() => import("./components/LibraryWorkspace").then((module) => ({ default: module.LibraryWorkspace })), "library");
const HelpWorkspace = lazyWithChunkReload(() => import("./components/HelpWorkspace").then((module) => ({ default: module.HelpWorkspace })), "help");

function uid(): string {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID();
  // http://host (not localhost) is not a secure context; randomUUID is missing there.
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const hex = [...bytes].map((byte) => byte.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}

const QWEN_CLOUD_KEY_HEADING = "Как подключить ключ QwenCloud";

type PanelSide = "sidebar" | "graph";
type Workspace = "chat" | "graph" | "library" | "help" | "cards";
type RightPanelState =
  | { kind: "closed" }
  | { kind: "research"; tab: ResearchTab; checkpointId?: string }
  | { kind: "cards"; tab: "templates" | "library" };

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
  const label = side === "sidebar" ? "Ширина боковой панели" : "Ширина панели хода работы";

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

const GRAPH_PANEL_MIN = 380;

export function App() {
  const [config, setConfig] = useState<UiConfig | null>(null);
  const [health, setHealth] = useState("…");
  const [collapsed, setCollapsed] = useState(() => window.innerWidth <= 680);
  const [sidebarWidth, setSidebarWidth] = useState(252);
  const [graphWidth, setGraphWidth] = useState(432);
  const [sessions, setSessions] = useState<ConversationSummary[]>([]);
  const [currentId, setCurrentId] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [branches, setBranches] = useState<Branch[]>([]);
  // Keep a successful rename visible while any in-flight conversation/map
  // request catches up. Some responses can still contain the previous branch
  // name, which used to make the map and top bar disagree.
  const [branchNameOverrides, setBranchNameOverrides] = useState<Record<string, string>>({});
  const branchNameOverridesRef = useRef<Record<string, string>>({});
  const [branchId, setBranchId] = useState("");
  const [headCheckpointId, setHeadCheckpointId] = useState("");
  const [viewCheckpointId, setViewCheckpointId] = useState("");
  const [requestedCheckpointId, setRequestedCheckpointId] = useState("");
  const [selectedResearchStep, setSelectedResearchStep] = useState<ResearchStep | null>(null);
  const [rightPanel, setRightPanel] = useState<RightPanelState>({ kind: "closed" });
  const [lastResearchTabs, setLastResearchTabs] = useState<Record<string, ResearchTab>>({});
  const [researchMapRefresh, setResearchMapRefresh] = useState(0);
  const [composerFocusKey, setComposerFocusKey] = useState(0);
  const [agenda, setAgenda] = useState<AgendaItem[]>([]);
  const [pendingApproval, setPendingApproval] = useState<PendingApproval | null>(null);
  const [turnFailures, setTurnFailures] = useState<TurnFailure[]>([]);
  const [dismissedFailures, setDismissedFailures] = useState<Set<number>>(new Set());
  const [approvalBusy, setApprovalBusy] = useState(false);
  const [cardBusy, setCardBusy] = useState(false);
  const [cardTemplates, setCardTemplates] = useState<CardTemplate[]>([]);
  const [lastGraphCheckpointId, setLastGraphCheckpointId] = useState("");
  const [workspace, setWorkspace] = useState<Workspace>("chat");
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [forkingCheckpointId, setForkingCheckpointId] = useState("");
  const [depth, setDepth] = useState<SearchDepth>("medium");
  const [effort, setEffort] = useState("");
  const [profile, setProfile] = useState("");
  const [mode, setMode] = useState<"auto" | "staged">("staged");
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [qwenKeyDraft, setQwenKeyDraft] = useState("");
  const [qwenKeyInputUnlocked, setQwenKeyInputUnlocked] = useState(false);
  const [helpSection, setHelpSection] = useState("");
  const [hasUserKey, setHasUserKey] = useState(() => Boolean(getLlmKey()));
  const [recording, setRecording] = useState(false);
  const [notice, setNotice] = useState("");
  const abortRef = useRef<AbortController | null>(null);
  // A history request may complete after a turn has already put its local
  // placeholder on screen. Until the stream is terminal, that server snapshot
  // is necessarily incomplete and must not erase the in-progress assistant.
  const liveTurnRef = useRef(false);
  const mediaRef = useRef<MediaRecorder | null>(null);
  const settingsRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setQwenKeyInputUnlocked(false);
  }, [settingsOpen]);

  const current = sessions.find((item) => item.id === currentId);
  const currentBranch = branches.find((item) => item.id === branchId);
  const currentBranchIndex = Math.max(0, branches.findIndex((item) => item.id === branchId));
  const activeBranchColor = branchColor(currentBranchIndex);
  const activeBranchLabel = currentBranchIndex === 0 && currentBranch?.name.trim().toLowerCase() === "main"
    ? "Основной вариант"
    : (currentBranch ? branchNameOverrides[currentBranch.id] || currentBranch.name : "Основной вариант");
  const activeBranchMode = currentBranch?.mode || current?.mode;
  const stagedAgendaActive = mode === "staged" && (!currentId || activeBranchMode === "staged");
  const openDirectionCount = agenda.filter((item) => item.status !== "closed").length;
  const checkpointGraphId = rightPanel.kind === "research" && rightPanel.tab === "data"
    ? rightPanel.checkpointId || lastGraphCheckpointId
    : "";
  const rightPanelOpen = rightPanel.kind !== "closed";
  const model = config?.models.find((item) => item.id === profile);
  const effortOptions =
    model?.reasoning_effort_options || config?.reasoning_effort_options || [];
  const maxGraphWidth = Math.max(
    GRAPH_PANEL_MIN,
    window.innerWidth - (collapsed ? 64 : sidebarWidth) - 320
  );

  function applyDetail(detail: ConversationDetail) {
    if (!liveTurnRef.current) setMessages(detail.messages);
    setBranches((detail.branches || []).map((branch) => ({
      ...branch,
      name: branchNameOverridesRef.current[branch.id] || branch.name,
    })));
    setBranchId(detail.activeBranchId || "");
    setHeadCheckpointId(detail.branchHeadCheckpointId || detail.headCheckpointId || "");
    setViewCheckpointId(detail.viewCheckpointId || detail.headCheckpointId || "");
    setAgenda(detail.agenda || []);
    setPendingApproval(detail.pendingApproval || null);
    setTurnFailures(detail.turnFailures || []);
    const selected = (detail.branches || []).find((item) => item.id === detail.activeBranchId);
    if (selected?.mode) {
      setMode(selected.mode);
      localStorage.setItem("retrieval_mode", selected.mode);
    }
  }

  useEffect(() => {
    if (!stagedAgendaActive && rightPanel.kind === "research" && rightPanel.tab === "directions") {
      setRightPanel({ kind: "research", tab: "map" });
    }
  }, [stagedAgendaActive, rightPanel]);

  useEffect(() => {
    if (checkpointGraphId) setLastGraphCheckpointId(checkpointGraphId);
  }, [checkpointGraphId]);

  useEffect(() => {
    Promise.all([fetchUiConfig(), fetchMe(), fetchConversations()])
      .then(([cfg, account, history]) => {
        setConfig(cfg);
        if (cfg.cards_enabled) {
          void fetchCardTemplates().then(setCardTemplates);
        }
        if (!cfg.staged_enabled) setMode("auto");
        bindAccount(account.id);
        setQwenKeyDraft(getLlmKey());
        setHasUserKey(Boolean(getLlmKey()));
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
        }
      })
      .catch(() => setHealth("нет связи"));
    fetchHealth()
      .then((body) => {
        const ok = body.status === "ready" || body.status === "ok";
        setHealth(ok ? "онлайн" : "есть сбои");
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
    fetchConversation(currentId, branchId, requestedCheckpointId)
      .then((detail) => {
        if (!cancelled) applyDetail(detail);
      })
      .catch((err) => {
        if (!cancelled) setNotice(err instanceof Error ? err.message : "Не удалось открыть чат");
      });
    return () => {
      cancelled = true;
    };
  }, [currentId, branchId, requestedCheckpointId]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      if (workspace !== "chat") return;
      if (settingsOpen) {
        setSettingsOpen(false);
        return;
      }
      if (rightPanelOpen) {
        setRightPanel({ kind: "closed" });
        setSelectedResearchStep(null);
        setRequestedCheckpointId("");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [workspace, rightPanelOpen, settingsOpen]);

  useEffect(() => {
    const syncSidebar = () => {
      if (window.innerWidth <= 680) setCollapsed(true);
    };
    window.addEventListener("resize", syncSidebar);
    return () => window.removeEventListener("resize", syncSidebar);
  }, []);

  useEffect(() => {
    const onPointerDown = (event: MouseEvent) => {
      const target = event.target as HTMLElement;
      if (target.closest("[data-settings-trigger]")) return;
      if (settingsRef.current && !settingsRef.current.contains(target)) {
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

  function openHelp(section = "") {
    setSettingsOpen(false);
    setRightPanel({ kind: "closed" });
    setHelpSection(section);
    setWorkspace("help");
  }

  async function refreshSessions() {
    const history = await fetchConversations();
    setSessions(history);
  }

  async function newChat() {
    abortRef.current?.abort();
    setCurrentId("");
    setBranchId("");
    setHeadCheckpointId("");
    setViewCheckpointId("");
    setRequestedCheckpointId("");
    setSelectedResearchStep(null);
    setRightPanel({ kind: "closed" });
    setBranches([]);
    setAgenda([]);
    setPendingApproval(null);
    setTurnFailures([]);
    setDismissedFailures(new Set());
    setMessages([]);
    setDraft("");
    const nextMode = config?.staged_enabled === false ? "auto" : "staged";
    setMode(nextMode);
    localStorage.setItem("retrieval_mode", nextMode);
    setWorkspace("chat");
  }

  function openSession(id: string) {
    abortRef.current?.abort();
    adoptSessionId(id);
    setCurrentId(id);
    setBranchId("");
    setRequestedCheckpointId("");
    setSelectedResearchStep(null);
    setRightPanel({ kind: "closed" });
    setTurnFailures([]);
    setDismissedFailures(new Set());
    setWorkspace("chat");
  }

  function restoreComposer(text: string, ...dropIds: string[]) {
    const remove = new Set(dropIds.filter(Boolean));
    if (remove.size) {
      setMessages((prev) => prev.filter((msg) => !remove.has(msg.id)));
    }
    setPendingApproval(null);
    if (text) setDraft(text);
    setComposerFocusKey((key) => key + 1);
  }

  function responseErrorMessage(body: unknown, fallback: string): string {
    if (!body || typeof body !== "object") return fallback;
    const payload = body as { error?: unknown; detail?: unknown };
    const value = payload.error ?? payload.detail;
    if (typeof value === "string" && value.trim()) return value;
    if (value && typeof value === "object" && "message" in value) {
      const message = (value as { message?: unknown }).message;
      if (typeof message === "string" && message.trim()) return message;
    }
    return fallback;
  }

  function patchAssistant(id: string, patch: Partial<ChatMessage>, fallback?: ChatMessage) {
    setMessages((prev) => {
      const exists = prev.some((msg) => msg.id === id);
      if (exists) return prev.map((msg) => (msg.id === id ? { ...msg, ...patch } : msg));
      return fallback ? [...prev, { ...fallback, ...patch }] : prev;
    });
  }

  async function send(
    text = draft,
    card?: { templateVersionId: string; templateName: string; version: number; schema: Record<string, unknown>; ui?: Record<string, unknown> }
  ) {
    const value = text.trim();
    if (!value || busy) return;
    let conversationId = currentId;
    let activeBranchId = branchId;
    let baseCheckpointId = viewCheckpointId || headCheckpointId;
    if (!conversationId) {
      try {
        const created = await createConversation(mode);
        conversationId = created.id;
        adoptSessionId(created.id);
        setCurrentId(created.id);
        setSessions((prev) => [created, ...prev]);
        activeBranchId = created.activeBranchId || "";
        baseCheckpointId = created.headCheckpointId || "";
        setBranchId(activeBranchId);
        setRequestedCheckpointId("");
      } catch (err) {
        setNotice(err instanceof Error ? err.message : "Не удалось создать чат");
        return;
      }
    }
    const sourceBranch = branches.find((item) => item.id === activeBranchId);
    if (sourceBranch?.mode === "auto" && mode === "staged") {
      setNotice("Исследование начинается в новом чате. Этот вариант остаётся быстрым ответом.");
      return;
    }
    if (sourceBranch?.mode === "staged" && mode === "auto") {
      if (!baseCheckpointId) return;
      try {
        const autoBranch = await forkConversation(
          conversationId,
          baseCheckpointId,
          "auto",
          sourceBranch.id
        );
        activeBranchId = autoBranch.id;
        setBranches((items) => [...items, autoBranch]);
        setBranchId(autoBranch.id);
        setHeadCheckpointId(baseCheckpointId);
        setRightPanel({ kind: "closed" });
      } catch (err) {
        setNotice(err instanceof Error ? err.message : "Не удалось открыть вариант «Быстрый ответ»");
        return;
      }
    }
    const user: ChatMessage = {
      id: uid(),
      role: "user",
      text: value,
      branchId: activeBranchId || undefined,
      cardRequest: card ? {
        templateVersionId: card.templateVersionId,
        templateName: card.templateName,
        version: card.version,
        schema: card.schema,
        ui: card.ui,
      } : undefined,
    };
    const assistantId = uid();
    setDraft("");
    const assistant: ChatMessage = {
      id: assistantId,
      role: "assistant",
      text: "",
      thinking: "",
      tools: [],
      steps: [],
      status: "streaming",
      branchId: activeBranchId || undefined,
    };
    const started = Date.now();
    const next = [...messages, user, assistant];
    liveTurnRef.current = true;
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
          fork_if_needed: Boolean(
            activeBranchId && baseCheckpointId && headCheckpointId && baseCheckpointId !== headCheckpointId
          ),
          intent: card ? "generate_card" : "chat",
          template_version_id: card?.templateVersionId,
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
          err = responseErrorMessage(body, err);
        } catch {
          err = res.statusText || err;
        }
        restoreComposer(value, user.id, assistantId);
        setNotice(err);
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      let answer = "";
      let thinking = "";
      let tools = [...(assistant.tools || [])];
      let steps: ChatStep[] = [...(assistant.steps || [])];
      let modelId = "";
      let modelLabel = "";
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
          ...(modelId ? { modelId, modelLabel } : {}),
          ...extra,
        }, assistant);
      };
      let terminal = false;
      const consume = (raw: string) => {
        const parsed = parseSseBlock(raw);
        if (!parsed) return false;
        const { event, data } = parsed;
        if (event === "branch_context") {
          const resolved = data.branch as Branch | undefined;
          if (resolved?.id) {
            activeBranchId = resolved.id;
            setMessages((items) => items.map((message) =>
              message.id === user.id || message.id === assistantId
                ? { ...message, branchId: resolved.id }
                : message
            ));
            setBranches((items) => {
              const exists = items.some((item) => item.id === resolved.id);
              return exists
                ? items.map((item) => item.id === resolved.id ? resolved : item)
                : [...items, resolved];
            });
            if (data.created) setNotice(`Создан вариант «${resolved.name}».`);
          }
        } else if (event === "thinking") appendThink(String(data.delta || ""));
        else if (event === "model") {
          modelId = String(data.id || "");
          modelLabel = String(data.label || "");
        } else if (event === "content") answer += String(data.delta || "");
        else if (event === "content_rewind") {
          const rewind = String(data.text || "");
          if (rewind && answer.endsWith(rewind)) answer = answer.slice(0, -rewind.length);
        } else if (event === "tool_call") {
          const id = String(data.id || `t${tools.length}`);
          const card = {
            id,
            name: String(data.name || "unknown"),
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
        } else if (event === "card_draft") {
          flush("streaming", {
            cardDraft: data.draft as ChatMessage["cardDraft"],
            cardTemplateName: String(data.template_name || card?.templateName || "Карточка"),
          });
        } else if (event === "done") {
          if (data.final_content) answer = String(data.final_content);
          const checkpointId = data.checkpoint_id ? String(data.checkpoint_id) : "";
          const chains = Number(data.graph_chain_count) || 0;
          if (data.modelId) modelId = String(data.modelId);
          if (data.modelLabel) modelLabel = String(data.modelLabel);
          flush("done", {
            graphChainCount: chains || undefined,
            checkpointId: checkpointId || undefined,
            sqStatusWarning: data.sqStatusWarning ? String(data.sqStatusWarning) : undefined,
          });
          if (checkpointId && data.open_graph) {
            setRightPanel({ kind: "research", tab: "data", checkpointId });
            setLastResearchTabs((tabs) => ({ ...tabs, [conversationId]: "data" }));
          }
          terminal = true;
          return true;
        } else if (event === "approval_required") {
          const approval = data.approval as PendingApproval;
          setPendingApproval(approval);
          flush("waiting_approval");
          terminal = true;
          return true;
        } else if (event === "turn_rolled_back") {
          restoreComposer(String(data.text || value), user.id, assistantId);
          if (data.message) setNotice(String(data.message));
          terminal = true;
          return true;
        } else if (event === "error") {
          flush("error", { text: String(data.message || "Ошибка стрима") });
          terminal = true;
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
      if (!terminal) restoreComposer(value, user.id, assistantId);
    } catch (err) {
      if ((err as Error).name === "AbortError") {
        restoreComposer(value, user.id, assistantId);
      } else {
        restoreComposer(value, user.id, assistantId);
        setNotice("Ошибка соединения с сервером");
      }
    } finally {
      liveTurnRef.current = false;
      setBusy(false);
      abortRef.current = null;
      void refreshSessions();
      if (conversationId) {
        setBranchId(activeBranchId);
        setRequestedCheckpointId("");
        setSelectedResearchStep(null);
        setResearchMapRefresh((value) => value + 1);
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
    action: "approve" | "revise" | "cancel",
    selection: { openSqRefs: string[]; newSubquestions: string[] },
    feedback = ""
  ) {
    if (!pendingApproval || approvalBusy) return;
    const streaming = action !== "cancel";
    const controller = streaming ? new AbortController() : null;
    const approvalAssistantId = pendingApproval.assistantMessageId;
    const assistantIndex = messages.findIndex((message) => message.id === approvalAssistantId);
    const approvalUser = assistantIndex > 0
      ? [...messages.slice(0, assistantIndex)].reverse().find((message) => message.role === "user")
      : undefined;
    setApprovalBusy(true);
    if (streaming) {
      setBusy(true);
      liveTurnRef.current = true;
      abortRef.current = controller;
    }
    try {
      const res = await resolveApproval(
        pendingApproval,
        action,
        selection,
        feedback,
        controller?.signal,
      );
      setPendingApproval(null);
      if (action === "cancel") {
        const payload = await res.json() as { text?: string; message?: string };
        restoreComposer(String(payload.text || ""));
        if (payload.message) setNotice(String(payload.message));
        if (currentId) applyDetail(await fetchConversation(currentId, branchId));
        return;
      }
      let openGraph = false;
      let rolledBack = false;
      if (res.body) {
        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        const assistantId = approvalAssistantId;
        const existing = messages.find((message) => message.id === assistantId);
        let answer = existing?.text || "";
        let thinking = existing?.thinking || "";
        let steps = [...(existing?.steps || [])];
        let status = existing?.status || "streaming";
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
            } else if (parsed?.event === "content") {
              answer += String(parsed.data.delta || "");
            } else if (parsed?.event === "tool_call") {
              const id = String(parsed.data.id || `tool-${steps.length}`);
              const tool = {
                kind: "tool" as const,
                id,
                name: String(parsed.data.name || "unknown"),
                status: "running" as const,
                args: parsed.data.arguments ?? parsed.data.args,
              };
              steps = [...steps.filter((item) => item.kind !== "tool" || item.id !== id), tool];
            } else if (parsed?.event === "tool_result") {
              const id = String(
                parsed.data.id || [...steps].reverse().find((item) => item.kind === "tool")?.id || "tool"
              );
              const result = String(parsed.data.result ?? parsed.data.preview ?? "");
              const toolStatus = parsed.data.ok === false ? "error" : "done";
              steps = steps.map((item) =>
                item.kind === "tool" && item.id === id
                  ? { ...item, status: toolStatus, result }
                  : item
              );
            } else if (parsed?.event === "done") {
              answer = String(parsed.data.final_content || answer);
              status = "done";
              openGraph = Boolean(parsed.data.open_graph);
              if (!rolledBack) {
                patchAssistant(assistantId, {
                  text: answer,
                  thinking,
                  steps,
                  status,
                  sqStatusWarning: parsed.data.sqStatusWarning
                    ? String(parsed.data.sqStatusWarning)
                    : undefined,
                });
              }
              separator = buffer.indexOf("\n\n");
              continue;
            } else if (parsed?.event === "approval_required") {
              setPendingApproval(parsed.data.approval as PendingApproval);
              status = "waiting_approval";
            } else if (parsed?.event === "turn_rolled_back") {
              restoreComposer(String(parsed.data.text || ""));
              if (parsed.data.message) setNotice(String(parsed.data.message));
              rolledBack = true;
              status = "aborted";
            } else if (parsed?.event === "error") {
              setNotice(String(parsed.data.message || "Ошибка продолжения"));
              status = "error";
            }
            if (!rolledBack) patchAssistant(assistantId, { text: answer, thinking, steps, status });
            separator = buffer.indexOf("\n\n");
          }
          if (done) break;
        }
      }
      const detail = await fetchConversation(currentId, branchId);
      applyDetail(detail);
      if (action === "approve" && openGraph && !rolledBack && detail.headCheckpointId) {
        setPendingApproval(null);
        setRightPanel({ kind: "research", tab: "data", checkpointId: detail.headCheckpointId });
        setLastResearchTabs((tabs) => ({ ...tabs, [currentId]: "data" }));
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") {
        if (currentId) {
          try { applyDetail(await fetchConversation(currentId, branchId)); }
          catch { /* The local rollback below still restores the question. */ }
        }
        restoreComposer(
          String(approvalUser?.text || ""),
          approvalUser?.id || "",
          approvalAssistantId,
        );
        setPendingApproval(null);
        setNotice("Продолжение остановлено.");
      } else {
        setNotice(error instanceof Error ? error.message : "Не удалось продолжить ответ");
      }
    } finally {
      setApprovalBusy(false);
      if (streaming) {
        setBusy(false);
        liveTurnRef.current = false;
        if (abortRef.current === controller) abortRef.current = null;
      }
    }
  }

  async function mutateAgenda(
    action: "set_status",
    input: { sq_ref: string; status: AgendaItem["status"] }
  ) {
    if (!branchId || !headCheckpointId) return;
    if (activeBranchMode !== "staged") {
      setNotice("Исследовательские вопросы доступны только в режиме «Исследование».");
      return;
    }
    if (pendingApproval) {
      setNotice("Сначала подтвердите или отклоните текущий список исследовательских вопросов.");
      return;
    }
    if (busy) {
      setNotice("Дождитесь окончания ответа, затем измените список вопросов.");
      return;
    }
    try {
      const result = await agendaEvent(branchId, headCheckpointId, action, input);
      setAgenda(result.agenda);
      setHeadCheckpointId(result.checkpointId);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Не удалось изменить план");
      const detail = await fetchConversation(currentId, branchId);
      applyDetail(detail);
    }
  }

  async function handleGenerateCard(templateVersionId: string) {
    if (!headCheckpointId || !currentId || cardBusy || busy || pendingApproval) return;
    const template = cardTemplates.find((item) => item.latestVersion.id === templateVersionId);
    if (!template) return;
    setCardBusy(true);
    setRightPanel({ kind: "closed" });
    try {
      await send(`Создай карточку «${template.name}» по нашему диалогу.`, {
        templateVersionId,
        templateName: template.name,
        version: template.latestVersion.version,
        schema: template.latestVersion.schema,
        ui: template.latestVersion.ui,
      });
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Не удалось сформировать карточку");
    } finally {
      setCardBusy(false);
    }
  }

  async function handleSaveCard(
    draftId: string,
    title: string,
    data?: Record<string, unknown>,
    provenance?: Record<string, unknown>,
    gaps?: unknown[]
  ) {
    try {
      if (data && provenance) await updateCardDraft(draftId, { data, provenance, gaps });
      await saveCardDraft(draftId, title);
      applyDetail(await fetchConversation(currentId, branchId));
      setNotice("Карточка сохранена в библиотеку.");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Не удалось сохранить карточку");
    }
  }

  async function handleInsertCard(card: SavedCard) {
    if (!branchId || !headCheckpointId || busy || pendingApproval) return;
    try {
      const result = await insertCardMessage(
        branchId,
        headCheckpointId,
        card.latestRevision.id
      );
      setHeadCheckpointId(result.checkpointId);
      if (currentId) applyDetail(await fetchConversation(currentId, branchId));
      setRightPanel({ kind: "closed" });
      setNotice(`Карточка «${card.title}» добавлена в контекст варианта.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Не удалось вставить карточку");
    }
  }

  async function handleRenameBranch(id: string, name: string) {
    const cleanName = name.trim().replace(/\s+/g, " ").slice(0, 64);
    if (!cleanName) return;
    branchNameOverridesRef.current = { ...branchNameOverridesRef.current, [id]: cleanName };
    setBranchNameOverrides((items) => ({ ...items, [id]: cleanName }));
    setBranches((items) => items.map((branch) => branch.id === id ? { ...branch, name: cleanName } : branch));
    try {
      const updated = await renameBranch(id, cleanName);
      branchNameOverridesRef.current = { ...branchNameOverridesRef.current, [id]: updated.name };
      setBranchNameOverrides((items) => ({ ...items, [id]: updated.name }));
      setBranches((items) => items.map((branch) => branch.id === id ? { ...branch, ...updated, name: updated.name } : branch));
      setNotice(`Вариант переименован: «${updated.name}».`);
    } catch (error) {
      const nextOverrides = { ...branchNameOverridesRef.current };
      delete nextOverrides[id];
      branchNameOverridesRef.current = nextOverrides;
      setBranchNameOverrides((items) => {
        const next = { ...items };
        delete next[id];
        return next;
      });
      void fetchConversation(currentId, branchId).then((detail) => applyDetail(detail)).catch(() => undefined);
      setNotice(error instanceof Error ? error.message : "Не удалось переименовать вариант");
      throw error;
    }
  }

  async function forkFromAnswer(checkpointId: string) {
    if (!currentId || !branchId || !checkpointId || busy || forkingCheckpointId) return;
    const sourceBranch = branches.find((item) => item.id === branchId);
    setForkingCheckpointId(checkpointId);
    try {
      const created = await forkConversation(
        currentId,
        checkpointId,
        sourceBranch?.mode || mode,
        branchId
      );
      setRequestedCheckpointId("");
      setSelectedResearchStep(null);
      setRightPanel({ kind: "closed" });
      setBranches((items) => items.some((item) => item.id === created.id) ? items : [...items, created]);
      setBranchId(created.id);
      setHeadCheckpointId(checkpointId);
      setViewCheckpointId(checkpointId);
      setMode(created.mode);
      localStorage.setItem("retrieval_mode", created.mode);
      applyDetail(await fetchConversation(currentId, created.id));
      setResearchMapRefresh((value) => value + 1);
      setComposerFocusKey((value) => value + 1);
      setNotice(`Создан вариант «${created.name}» от выбранного ответа.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Не удалось создать вариант");
    } finally {
      setForkingCheckpointId("");
    }
  }

  function openResearch(tab?: ResearchTab, checkpointId?: string) {
    const remembered = currentId ? lastResearchTabs[currentId] : undefined;
    const requested = tab || remembered || "map";
    const nextTab = requested === "directions" && !stagedAgendaActive ? "map" : requested;
    const panelCheckpoint = rightPanel.kind === "research" ? rightPanel.checkpointId : undefined;
    const nextCheckpoint = nextTab === "data"
      ? checkpointId || panelCheckpoint || lastGraphCheckpointId
      : panelCheckpoint;
    if (nextTab === "data" && !nextCheckpoint) return;
    setRightPanel({ kind: "research", tab: nextTab, checkpointId: nextCheckpoint });
    if (currentId) setLastResearchTabs((tabs) => ({ ...tabs, [currentId]: nextTab }));
  }

  function closeResearch() {
    setRightPanel({ kind: "closed" });
    setSelectedResearchStep(null);
    setRequestedCheckpointId("");
  }

  function selectEmptyBranch(branch: ResearchBranch) {
    const checkpointId = branch.headCheckpointId || branch.createdFromCheckpointId || "";
    setSelectedResearchStep(null);
    setRequestedCheckpointId("");
    setBranchId(branch.id);
    if (checkpointId) {
      setHeadCheckpointId(checkpointId);
      setViewCheckpointId(checkpointId);
    }
    if (branch.mode) {
      setMode(branch.mode);
      localStorage.setItem("retrieval_mode", branch.mode);
    }
    setComposerFocusKey((value) => value + 1);
    if (window.innerWidth <= 900) setRightPanel({ kind: "closed" });
  }

  function selectResearchStep(nextBranchId: string, step: ResearchStep) {
    const checkpointId = step.resumeCheckpointId || step.answerCheckpointId || "";
    if (!checkpointId) return;
    const graphCheckpointId = step.graphCheckpointId || step.answerCheckpointId || checkpointId;
    setSelectedResearchStep(step);
    setBranchId(nextBranchId);
    setRequestedCheckpointId(checkpointId);
    setLastGraphCheckpointId(graphCheckpointId);
    setRightPanel((panel) => panel.kind === "research"
      ? { ...panel, checkpointId: graphCheckpointId }
      : panel);
    if (window.innerWidth <= 900) setRightPanel({ kind: "closed" });
  }

  async function openResearchStep(stepId: string, originBranchId?: string) {
    if (!currentId) return;
    try {
      const map = await fetchResearchMap(currentId, originBranchId || branchId);
      const step = map.steps.find((item) => item.id === stepId);
      if (!step) throw new Error("Исходный шаг не найден");
      openResearch("map");
      selectResearchStep(originBranchId || step.branchId, step);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Не удалось открыть исходный шаг");
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
          "--active-branch-color": activeBranchColor,
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
        onExplorer={() => { setRightPanel({ kind: "closed" }); setWorkspace("graph"); }}
        onLibrary={() => { setRightPanel({ kind: "closed" }); setWorkspace("library"); }}
        onHelp={() => openHelp()}
        onCards={() => { setRightPanel({ kind: "closed" }); setWorkspace("cards"); }}
        cardsEnabled={config?.cards_enabled !== false}
        health={health}
        settingsOpen={settingsOpen}
        onSettings={() => setSettingsOpen((value) => !value)}
        onLogout={() => void logout()}
        keyWarning={!hasUserKey}
        activeWorkspace={workspace}
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
            <span>{workspace === "graph" ? "Вся база" : workspace === "library" ? "Статьи · Скоро" : workspace === "help" ? "Помощь" : workspace === "cards" ? "Карточки" : current?.title || "Новый чат"}</span>
            {workspace === "chat" && branches.length > 0 && (
              <BranchMenu
                branches={branches}
                activeId={branchId}
                open={rightPanel.kind === "research"}
                openDirections={stagedAgendaActive ? openDirectionCount : 0}
                onOpen={() => rightPanel.kind === "research" ? closeResearch() : openResearch("map")}
              />
            )}
          </div>
        </header>
        {workspace === "chat" ? <div className={`chat-col ${empty ? "is-empty" : ""} ${rightPanelOpen ? "with-graph" : ""}`}>
          {!empty && (
          <ChatThread
            messages={messages}
            cardTemplates={cardTemplates}
            agenda={agenda}
            openGraphId={checkpointGraphId}
            onOpenGraph={(checkpointId) => openResearch("data", checkpointId)}
            pendingApproval={pendingApproval}
            approvalBusy={approvalBusy}
            onResolveApproval={(action, sqs, feedback) => void handleApproval(action, sqs, feedback)}
            turnFailures={turnFailures.filter((item) => !dismissedFailures.has(item.createdAt))}
            onDismissFailure={(createdAt) => setDismissedFailures((prev) => new Set(prev).add(createdAt))}
            selectedMessageIds={selectedResearchStep ? [
              selectedResearchStep.question.messageId,
              ...(selectedResearchStep.answer?.messageId ? [selectedResearchStep.answer.messageId] : []),
            ] : []}
            onSaveCard={(draftId, title, data, provenance, gaps) => void handleSaveCard(draftId, title, data, provenance, gaps)}
            onFork={(checkpointId) => void forkFromAnswer(checkpointId)}
            forkingCheckpointId={forkingCheckpointId}
            branchVisuals={Object.fromEntries(branches.map((branch, index) => [
              branch.id,
              {
                color: branchColor(index),
                label: index === 0 && branch.name.trim().toLowerCase() === "main"
                  ? "Основной вариант"
                  : branchNameOverrides[branch.id] || branch.name,
              },
            ]))}
            activeBranchId={branchId}
            showBranchHighlights={rightPanel.kind === "research"}
          />
          )}
          <div className="composer-stack">
          {empty && (
            <div className="welcome">
              <p>
                <button type="button" className="welcome-help-link" onClick={() => openHelp()}>
                  Как пользоваться
                </button>
                <span> — или спросите у ассистента, он сам расскажет</span>
              </p>
            </div>
          )}
          {empty && !hasUserKey && (
            <div className="demo-access" role="status">
              <span>Используется демонстрационный доступ</span>
              <button type="button" onClick={() => setSettingsOpen(true)}>Настроить</button>
            </div>
          )}
          {viewCheckpointId && headCheckpointId && viewCheckpointId !== headCheckpointId && selectedResearchStep && (
            <div className="context-continuation" role="status">
              <span><b>Продолжение от:</b> «{selectedResearchStep.question.preview}» · отправите — начнётся отдельный вариант</span>
              <button type="button" onClick={() => {
                setSelectedResearchStep(null);
                setRequestedCheckpointId("");
              }}>К последнему шагу</button>
            </div>
          )}
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
            }}
            centered={empty}
            mode={mode}
            onMode={(value) => {
              setMode(value);
              localStorage.setItem("retrieval_mode", value);
            }}
            stagedEnabled={config?.staged_enabled !== false}
            branchMode={activeBranchMode}
            cardsEnabled={config?.cards_enabled !== false}
            cardActionsEnabled={Boolean(headCheckpointId && currentId && !pendingApproval && !busy)}
            onOpenCards={() => setRightPanel({ kind: "cards", tab: "templates" })}
            focusKey={composerFocusKey}
          />
          </div>
        </div> : workspace === "graph" ? (
          <Suspense fallback={<p className="explorer-status">Загрузка базы…</p>}>
            <Explorer onUseCollection={(text) => {
              setDraft((currentDraft) => currentDraft.trim() ? `${currentDraft.trim()}\n\n${text}` : text);
              setWorkspace("chat");
              setNotice("Подборка добавлена в черновик сообщения.");
            }} />
          </Suspense>
        ) : workspace === "library" ? (
          <Suspense fallback={<p className="explorer-status">Загрузка библиотеки…</p>}>
            <LibraryWorkspace />
          </Suspense>
        ) : workspace === "help" ? (
          <Suspense fallback={<p className="explorer-status">Загрузка помощи…</p>}>
            <HelpWorkspace focusHeading={helpSection} />
          </Suspense>
        ) : (
          <Suspense fallback={<p className="explorer-status">Загрузка карточек…</p>}>
            <CardsWorkspace
              checkpointId={headCheckpointId}
              branchId={branchId}
              onNotice={setNotice}
            />
          </Suspense>
        )}
      </div>
      {workspace === "chat" && rightPanelOpen && (
        <PanelResizer
          side="graph"
          value={graphWidth}
          min={GRAPH_PANEL_MIN}
          max={maxGraphWidth}
          onChange={setGraphWidth}
        />
      )}
      {workspace === "chat" && rightPanel.kind === "research" && (
        <ResearchPanelShell
          tab={rightPanel.tab}
          branchName={activeBranchLabel}
          staged={stagedAgendaActive}
          openDirections={openDirectionCount}
          dataAvailable={Boolean(lastGraphCheckpointId || rightPanel.checkpointId)}
          onTab={(tab) => openResearch(tab)}
          onClose={closeResearch}
        >
          {rightPanel.tab === "map" ? (
            <Suspense fallback={<p className="explorer-status">Загрузка карты…</p>}>
              <ResearchMapPane
                conversationId={currentId}
                activeBranchId={branchId}
                selectedStepId={selectedResearchStep?.id || ""}
                refreshKey={researchMapRefresh}
                dataCheckpointId={rightPanel.checkpointId || lastGraphCheckpointId}
                onSelectStep={selectResearchStep}
                onSelectEmptyBranch={selectEmptyBranch}
                onOpenGraph={(checkpointId) => openResearch("data", checkpointId)}
                onOpenData={() => openResearch("data")}
                onRenameBranch={handleRenameBranch}
                onClose={closeResearch}
                embedded
              />
            </Suspense>
          ) : rightPanel.tab === "directions" ? (
            <AgendaDrawer
              agenda={agenda}
              embedded
              locked={busy || Boolean(pendingApproval)}
              onStatus={(item, status) => mutateAgenda("set_status", { sq_ref: item.ref, status })}
            />
          ) : checkpointGraphId ? (
            <Suspense fallback={<p className="explorer-status">Загрузка данных…</p>}>
              <GraphPane
                checkpointId={checkpointGraphId}
                onClose={closeResearch}
                onOpenStep={(stepId, originBranchId) => void openResearchStep(stepId, originBranchId)}
                embedded
              />
            </Suspense>
          ) : <p className="explorer-status">В этом диалоге ещё нет данных из базы.</p>}
        </ResearchPanelShell>
      )}
      {workspace === "chat" && rightPanel.kind === "cards" && (
        <Suspense fallback={<aside className="card-side-pane"><p className="explorer-status">Загрузка карточек…</p></aside>}>
          <aside className="card-side-pane">
            <CardsWorkspace
              checkpointId={headCheckpointId}
              branchId={branchId}
              onNotice={setNotice}
              chatMode
              initialTab={rightPanel.tab}
              onGenerate={(templateVersionId) => void handleGenerateCard(templateVersionId)}
              onInsert={(card) => void handleInsertCard(card)}
              onClose={() => setRightPanel({ kind: "closed" })}
            />
          </aside>
        </Suspense>
      )}
      {settingsOpen && <div className="sidebar-settings-pop" ref={settingsRef}>
        <p className="settings-title">Подключение</p>
        <p className="settings-hint">Ключи хранятся только в этой вкладке браузера.</p>
        {!hasUserKey && (
          <p className="settings-key-warning" id="qwen-key-warning" role="status">
            Ключ не вставлен — используется демонстрационный. Вставьте свой ключ QwenCloud.
            {" "}
            <button type="button" className="settings-key-warning-link" onClick={() => openHelp(QWEN_CLOUD_KEY_HEADING)}>
              Как получить ключ
            </button>
          </p>
        )}
        <label className="key-field">
          <span>Qwen</span>
          <input
            className="secure-key-input"
            type="password"
            name="qwen-runtime-token"
            autoComplete="new-password"
            data-1p-ignore="true"
            data-lpignore="true"
            data-form-type="other"
            spellCheck={false}
            readOnly={!qwenKeyInputUnlocked}
            onPointerDown={(event) => { event.currentTarget.readOnly = false; setQwenKeyInputUnlocked(true); }}
            onKeyDown={(event) => { event.currentTarget.readOnly = false; setQwenKeyInputUnlocked(true); }}
            onBlur={() => setQwenKeyInputUnlocked(false)}
            value={qwenKeyDraft}
            onChange={(event) => {
              if (event.currentTarget.readOnly) { event.currentTarget.value = qwenKeyDraft; return; }
              setQwenKeyDraft(event.target.value);
            }}
            placeholder="из конфига сервера"
            aria-describedby={!hasUserKey ? "qwen-key-warning" : undefined}
          />
        </label>
        <button type="button" className="ghost-btn" onClick={() => {
          setLlmKey("qwen_cloud", qwenKeyDraft);
          setHasUserKey(Boolean(qwenKeyDraft.trim()));
          setSettingsOpen(false);
        }}>Сохранить</button>
        <button type="button" className="ghost-btn danger-btn" disabled={!currentId} onClick={async () => {
          if (!currentId || !window.confirm("Удалить этот чат без возможности восстановления?")) return;
          try {
            await deleteConversation(currentId);
            const remaining = sessions.filter((item) => item.id !== currentId);
            setSessions(remaining);
            const nextId = remaining[0]?.id || "";
            if (nextId) adoptSessionId(nextId); else sessionStorage.removeItem("neo4j-assistant.session-id");
            setCurrentId(nextId); setMessages([]); setRightPanel({ kind: "closed" }); setSettingsOpen(false);
          } catch (error) { setNotice(error instanceof Error ? error.message : "Не удалось удалить чат"); }
        }}>Удалить чат</button>
      </div>}
      {notice && <div className="toast" role="status">{notice}</div>}
    </div>
  );
}

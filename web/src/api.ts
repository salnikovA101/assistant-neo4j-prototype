import type {
  Account,
  AgendaItem,
  Branch,
  CardDraft,
  CardTemplate,
  ChatMessage,
  ConversationDetail,
  ConversationSummary,
  GraphPayload,
  PendingApproval,
  SavedCard,
  SearchDepth,
  UiConfig,
} from "./types";

const SESSION_KEY = "neo4j-assistant.session-id";
const ACCOUNT_KEY = "neo4j-assistant.account-id";
const LLM_KEY = "llm_api_key_override";
const LLM_KEY_QWEN = "llm_api_key_override:qwen_cloud";

export function getSessionId(): string {
  return sessionStorage.getItem(SESSION_KEY) || "";
}

export function adoptSessionId(id: string): void {
  sessionStorage.setItem(SESSION_KEY, id);
}

export function bindAccount(id: string): void {
  const previous = sessionStorage.getItem(ACCOUNT_KEY);
  if (previous !== id) {
    sessionStorage.removeItem(SESSION_KEY);
    sessionStorage.removeItem(LLM_KEY);
    sessionStorage.removeItem(LLM_KEY_QWEN);
  }
  sessionStorage.setItem(ACCOUNT_KEY, id);
}

function keyFamily(profile: string): "qwen" | "ollama" {
  const name = profile.trim().toLowerCase();
  if (name === "qwen_cloud" || name.startsWith("qwen_")) return "qwen";
  return "ollama";
}

export function getLlmKey(profile: string): string {
  const storage = keyFamily(profile) === "qwen" ? LLM_KEY_QWEN : LLM_KEY;
  return (sessionStorage.getItem(storage) || "").trim();
}

export function setLlmKey(profile: string, value: string): void {
  const storage = keyFamily(profile) === "qwen" ? LLM_KEY_QWEN : LLM_KEY;
  const key = value.trim();
  if (key) sessionStorage.setItem(storage, key);
  else sessionStorage.removeItem(storage);
}

export function withHeaders(profile: string, extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  const sessionId = getSessionId();
  if (sessionId) headers.set("X-Session-Id", sessionId);
  const key = getLlmKey(profile);
  if (key) headers.set("X-LLM-Api-Key", key);
  return headers;
}

async function json<T>(res: Response, fallback: string): Promise<T> {
  if (res.status === 401) {
    window.location.assign("/login");
    throw new Error("Сессия истекла");
  }
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) {
    const detail = payload.error || payload.detail;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail || fallback));
  }
  return payload as T;
}

export async function fetchMe(): Promise<Account> {
  return json(await fetch("/api/me"), "Не удалось загрузить аккаунт");
}

export async function fetchConversations(): Promise<ConversationSummary[]> {
  const body = await json<{ items: ConversationSummary[] }>(
    await fetch("/api/conversations?limit=100"),
    "Не удалось загрузить историю"
  );
  return body.items;
}

export async function createConversation(): Promise<ConversationSummary> {
  return json(
    await fetch("/api/conversations", { method: "POST" }),
    "Не удалось создать чат"
  );
}

export async function fetchConversation(id: string, branchId = ""): Promise<ConversationDetail> {
  const query = branchId ? `?branch_id=${encodeURIComponent(branchId)}` : "";
  return json(await fetch(`/api/conversations/${encodeURIComponent(id)}${query}`), "Не удалось открыть чат");
}

export async function forkConversation(id: string, checkpointId: string): Promise<Branch> {
  return json(
    await fetch(`/api/conversations/${encodeURIComponent(id)}/forks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ checkpoint_id: checkpointId }),
    }),
    "Не удалось создать ветку"
  );
}

export async function agendaEvent(
  branchId: string,
  baseCheckpointId: string,
  action: "add" | "edit" | "close" | "reopen" | "reorder",
  input: { sq_id?: string; text?: string; ordered_ids?: string[] } = {}
): Promise<{ checkpointId: string; agenda: AgendaItem[] }> {
  return json(
    await fetch(`/api/branches/${encodeURIComponent(branchId)}/agenda-events`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_checkpoint_id: baseCheckpointId, action, ...input }),
    }),
    "Не удалось изменить SQ"
  );
}

export async function deleteConversation(id: string): Promise<void> {
  await json(
    await fetch(`/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE" }),
    "Не удалось удалить чат"
  );
}

export async function logout(): Promise<void> {
  sessionStorage.removeItem(ACCOUNT_KEY);
  sessionStorage.removeItem(SESSION_KEY);
  sessionStorage.removeItem(LLM_KEY);
  sessionStorage.removeItem(LLM_KEY_QWEN);
  await fetch("/logout", { method: "POST", redirect: "manual" });
  window.location.assign("/login");
}

export async function fetchUiConfig(): Promise<UiConfig> {
  const res = await fetch("/ui_config", { headers: withHeaders("") });
  if (res.status === 401) window.location.assign("/login");
  if (!res.ok) throw new Error("ui_config failed");
  return res.json();
}

export async function fetchHealth(): Promise<{ status: string }> {
  const res = await fetch("/health", { headers: withHeaders("") });
  if (!res.ok) return { status: "down" };
  return res.json();
}

export async function clearHistory(): Promise<void> {
  const res = await fetch("/clear_history", {
    method: "POST",
    headers: withHeaders(""),
  });
  if (!res.ok) throw new Error("Не удалось очистить историю на сервере");
}

export async function fetchGraphViz(runId: string): Promise<GraphPayload> {
  const res = await fetch("/graph_viz", {
    method: "POST",
    headers: withHeaders("", { "Content-Type": "application/json" }),
    body: JSON.stringify({ graph_run_id: runId }),
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(payload.error || "Не удалось загрузить граф");
  return payload as GraphPayload;
}

export async function fetchGraphExplore(
  q: string,
  limit: 10 | 100 | 1000,
  field: "all" | "name" | "label" | "rel" | "evidence" | "source" = "all",
  cursor = ""
): Promise<GraphPayload> {
  const res = await fetch("/graph_explore", {
    method: "POST",
    headers: withHeaders("", { "Content-Type": "application/json" }),
    body: JSON.stringify({ q, limit, field, cursor }),
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(payload.error || "Не удалось загрузить граф");
  return payload as GraphPayload;
}

export async function fetchCheckpointGraph(
  checkpointId: string,
  scope: "context" | "new_in_answer" | "unit" | "all_branches" = "context",
  unitId = ""
): Promise<GraphPayload> {
  const params = new URLSearchParams({ scope });
  if (unitId) params.set("unit_id", unitId);
  return json(
    await fetch(`/api/checkpoints/${encodeURIComponent(checkpointId)}/graph?${params}`),
    "Не удалось загрузить checkpoint-граф"
  );
}

export async function fetchGraphExpand(nodeId: string, limit: 10 | 100 | 1000 = 100): Promise<GraphPayload> {
  return json(
    await fetch("/api/graph/expand", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ node_id: nodeId, limit }),
    }),
    "Не удалось раскрыть соседей"
  );
}

export async function resolveApproval(
  approval: PendingApproval,
  action: "approve" | "revise" | "cancel",
  subquestions: string[],
  feedback = ""
): Promise<Response> {
  const res = await fetch(`/api/tool-approvals/${encodeURIComponent(approval.id)}/resolve`, {
    method: "POST",
    headers: withHeaders("", { "Content-Type": "application/json", Accept: "text/event-stream" }),
    body: JSON.stringify({ action, revision: approval.revision, subquestions, feedback }),
  });
  if (!res.ok) await json(res, "Не удалось обработать план поиска");
  return res;
}

export async function fetchCardTemplates(): Promise<CardTemplate[]> {
  return json(await fetch("/api/card-templates"), "Не удалось загрузить шаблоны");
}

export async function fetchCards(): Promise<SavedCard[]> {
  return json(await fetch("/api/cards"), "Не удалось загрузить карточки");
}

export async function createCardTemplate(input: {
  name: string;
  description?: string;
  schema: Record<string, unknown>;
  ui?: Record<string, unknown>;
  instructions?: string;
}): Promise<CardTemplate> {
  return json(
    await fetch("/api/card-templates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
    "Не удалось создать шаблон"
  );
}

export async function archiveCardTemplate(templateId: string): Promise<void> {
  await json(
    await fetch(`/api/card-templates/${encodeURIComponent(templateId)}`, { method: "DELETE" }),
    "Не удалось архивировать шаблон"
  );
}

export async function generateCardDraft(
  checkpointId: string,
  templateVersionId: string,
  profile?: string,
  reasoningEffort?: string
): Promise<{ draft: CardDraft; message: ChatMessage; checkpointId: string }> {
  const payload = await json<{ type: string; draft: CardDraft; message: ChatMessage; checkpoint_id: string }>(
    await fetch("/api/card-drafts/generate", {
      method: "POST",
      headers: withHeaders(profile || "", { "Content-Type": "application/json" }),
      body: JSON.stringify({
        checkpoint_id: checkpointId,
        template_version_id: templateVersionId,
        profile,
        reasoning_effort: reasoningEffort,
      }),
    }),
    "Не удалось сгенерировать карточку"
  );
  return { draft: payload.draft, message: payload.message, checkpointId: payload.checkpoint_id };
}

export async function saveCardDraft(draftId: string, title = ""): Promise<SavedCard> {
  return json(
    await fetch(`/api/card-drafts/${encodeURIComponent(draftId)}/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),
    "Не удалось сохранить карточку"
  );
}

export async function importCardDraft(
  templateVersionId: string,
  data: Record<string, unknown> | Record<string, unknown>[]
): Promise<CardDraft[]> {
  const payload = await json<CardDraft | { items: CardDraft[] }>(
    await fetch("/api/cards/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ template_version_id: templateVersionId, data }),
    }),
    "Импорт не прошёл validation"
  );
  return "items" in payload ? payload.items : [payload];
}

export async function archiveCard(cardId: string): Promise<void> {
  await json(
    await fetch(`/api/cards/${encodeURIComponent(cardId)}`, { method: "DELETE" }),
    "Не удалось удалить карточку"
  );
}

export async function attachCard(
  branchId: string,
  baseCheckpointId: string,
  cardRevisionId: string,
  attached = true
): Promise<{ checkpointId: string }> {
  return json(
    await fetch(`/api/branches/${encodeURIComponent(branchId)}/card-attachments`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        base_checkpoint_id: baseCheckpointId,
        card_revision_id: cardRevisionId,
        attached,
      }),
    }),
    "Не удалось прикрепить карточку"
  );
}

export function streamBody(
  text: string,
  opts: {
    search_depth: SearchDepth;
    reasoning_effort?: string;
    profile?: string;
    turn_id?: string;
    mode?: "auto" | "staged";
    branch_id?: string;
    base_checkpoint_id?: string;
  }
): string {
  const payload: Record<string, string> = {
    text,
    search_depth: opts.search_depth,
  };
  if (opts.profile) payload.profile = opts.profile;
  if (opts.reasoning_effort) payload.reasoning_effort = opts.reasoning_effort;
  if (opts.turn_id) payload.turn_id = opts.turn_id;
  if (opts.mode) payload.mode = opts.mode;
  if (opts.branch_id) payload.branch_id = opts.branch_id;
  if (opts.base_checkpoint_id) payload.base_checkpoint_id = opts.base_checkpoint_id;
  return JSON.stringify(payload);
}

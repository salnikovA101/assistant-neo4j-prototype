import type {
  Account,
  AgendaItem,
  Branch,
  CardDraft,
  CardTemplate,
  ChatMessage,
  ConversationDetail,
  ConversationSummary,
  GraphFacets,
  GraphFilters,
  GraphPayload,
  PendingApproval,
  ResearchMap,
  SavedCard,
  SearchDepth,
  UiConfig,
} from "./types";

const SESSION_KEY = "neo4j-assistant.session-id";
const ACCOUNT_KEY = "neo4j-assistant.account-id";
const LLM_KEY = "llm_api_key_override";
const LLM_KEY_QWEN = "llm_api_key_override:qwen_cloud";

export function getWorkspace(): string {
  const match = window.location.pathname.match(/^\/ui\/([a-z0-9][a-z0-9_-]{0,63})(?:\/|$)/);
  return match?.[1] || "packaging";
}

export function workspaceUiUrl(): string {
  return `/ui/${getWorkspace()}/`;
}

export function workspaceLoginUrl(): string {
  return `/ui/${getWorkspace()}/login`;
}

function isLoginPath(pathname = window.location.pathname): boolean {
  return /\/ui\/[^/]+\/login\/?$/.test(pathname);
}

export function redirectToLogin(): void {
  if (isLoginPath()) return;
  window.location.assign(workspaceLoginUrl());
}

export function workspaceLogoutUrl(): string {
  return `/ui/${getWorkspace()}/logout`;
}

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

export function getLlmKey(_profile = ""): string {
  return (sessionStorage.getItem(LLM_KEY_QWEN) || "").trim();
}

export function setLlmKey(_profile: string, value: string): void {
  const key = value.trim();
  if (key) sessionStorage.setItem(LLM_KEY_QWEN, key);
  else sessionStorage.removeItem(LLM_KEY_QWEN);
}

export function withHeaders(_profile: string, extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  headers.set("X-Workspace", getWorkspace());
  const sessionId = getSessionId();
  if (sessionId) headers.set("X-Session-Id", sessionId);
  const key = getLlmKey();
  if (key) headers.set("X-LLM-Api-Key", key);
  return headers;
}

export function apiFetch(input: RequestInfo | URL, init: RequestInit = {}): Promise<Response> {
  return window.fetch(input, { ...init, headers: withHeaders("", init.headers) });
}

async function json<T>(res: Response, fallback: string): Promise<T> {
  if (res.status === 401) {
    redirectToLogin();
    throw new Error("Сессия истекла");
  }
  let payload: unknown;
  try {
    payload = await res.json();
  } catch {
    throw new Error(res.ok ? "Некорректный JSON ответа" : fallback);
  }
  if (!res.ok) {
    const record = payload && typeof payload === "object" ? (payload as Record<string, unknown>) : {};
    const detail = record.error || record.detail;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail || fallback));
  }
  return payload as T;
}

export async function fetchMe(): Promise<Account> {
  return json(await apiFetch("/api/me"), "Не удалось загрузить аккаунт");
}

export async function fetchServiceGuide(): Promise<string> {
  const payload = await json<{ markdown: string }>(
    await apiFetch("/api/service-guide"),
    "Не удалось загрузить помощь"
  );
  const guide = String(payload.markdown || "").trim();
  if (!guide) throw new Error("Раздел помощи пуст");
  return guide;
}

export async function fetchConversations(): Promise<ConversationSummary[]> {
  const body = await json<{ items: ConversationSummary[] }>(
    await apiFetch("/api/conversations?limit=100"),
    "Не удалось загрузить историю"
  );
  return body.items;
}

export async function createConversation(mode: "auto" | "staged" = "staged"): Promise<ConversationSummary> {
  return json(
    await apiFetch("/api/conversations", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode }),
    }),
    "Не удалось создать чат"
  );
}

export async function fetchConversation(
  id: string,
  branchId = "",
  checkpointId = ""
): Promise<ConversationDetail> {
  const params = new URLSearchParams();
  if (branchId) params.set("branch_id", branchId);
  if (checkpointId) params.set("checkpoint_id", checkpointId);
  const query = params.size ? `?${params.toString()}` : "";
  return json(await apiFetch(`/api/conversations/${encodeURIComponent(id)}${query}`), "Не удалось открыть чат");
}

export async function fetchResearchMap(id: string, branchId = ""): Promise<ResearchMap> {
  const query = branchId ? `?branch_id=${encodeURIComponent(branchId)}` : "";
  return json(
    await apiFetch(`/api/conversations/${encodeURIComponent(id)}/research-map${query}`),
    "Не удалось загрузить карту хода"
  );
}

export async function forkConversation(
  id: string,
  checkpointId: string,
  mode?: "auto" | "staged",
  sourceBranchId?: string
): Promise<Branch> {
  return json(
    await apiFetch(`/api/conversations/${encodeURIComponent(id)}/forks`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        checkpoint_id: checkpointId,
        mode,
        source_branch_id: sourceBranchId,
      }),
    }),
    "Не удалось создать вариант"
  );
}

export async function renameBranch(branchId: string, name: string): Promise<Branch> {
  return json(
    await apiFetch(`/api/branches/${encodeURIComponent(branchId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    }),
    "Не удалось переименовать вариант"
  );
}

export async function agendaEvent(
  branchId: string,
  baseCheckpointId: string,
  action: "add" | "edit" | "close" | "reopen" | "set_status" | "reorder",
  input: { sq_ref?: string; text?: string; status?: AgendaItem["status"]; ordered_refs?: string[] } = {}
): Promise<{ checkpointId: string; agenda: AgendaItem[] }> {
  return json(
    await apiFetch(`/api/branches/${encodeURIComponent(branchId)}/agenda-events`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_checkpoint_id: baseCheckpointId, action, ...input }),
    }),
    "Не удалось изменить план"
  );
}

export async function deleteConversation(id: string): Promise<void> {
  await json(
    await apiFetch(`/api/conversations/${encodeURIComponent(id)}`, { method: "DELETE" }),
    "Не удалось удалить чат"
  );
}

export async function logout(): Promise<void> {
  sessionStorage.removeItem(ACCOUNT_KEY);
  sessionStorage.removeItem(SESSION_KEY);
  sessionStorage.removeItem(LLM_KEY);
  sessionStorage.removeItem(LLM_KEY_QWEN);
  await apiFetch(workspaceLogoutUrl(), { method: "POST", redirect: "manual" });
  window.location.assign(workspaceLoginUrl());
}

export async function fetchUiConfig(): Promise<UiConfig> {
  const res = await apiFetch("/ui_config", { headers: withHeaders("") });
  if (res.status === 401) redirectToLogin();
  if (!res.ok) throw new Error("ui_config failed");
  return res.json();
}

export async function fetchHealth(): Promise<{ status: string }> {
  const res = await apiFetch("/health", { headers: withHeaders("") });
  if (!res.ok) return { status: "down" };
  return res.json();
}

export async function fetchGraphViz(runId: string): Promise<GraphPayload> {
  return json(
    await apiFetch("/graph_viz", {
      method: "POST",
      headers: withHeaders("", { "Content-Type": "application/json" }),
      body: JSON.stringify({ graph_run_id: runId }),
    }),
    "Не удалось загрузить данные"
  );
}

export async function fetchGraphExplore(
  q: string,
  limit: number,
  field: "all" | "name" | "label" | "rel" | "evidence" | "source" = "all",
  cursor = "",
  filters?: GraphFilters
): Promise<GraphPayload> {
  return json(
    await apiFetch("/graph_explore", {
      method: "POST",
      headers: withHeaders("", { "Content-Type": "application/json" }),
      body: JSON.stringify({ q, limit, field, cursor, filters }),
    }),
    "Не удалось загрузить базу"
  );
}

export async function fetchCheckpointGraph(
  checkpointId: string,
  scope: "mode_default" | "context" | "new_in_answer" | "unit" | "all_branches" = "mode_default",
  unitId = ""
): Promise<GraphPayload> {
  const params = new URLSearchParams({ scope });
  if (unitId) params.set("unit_id", unitId);
  return json(
    await apiFetch(`/api/checkpoints/${encodeURIComponent(checkpointId)}/graph?${params}`),
    "Не удалось загрузить данные"
  );
}

export async function fetchGraphExpand(
  nodeId: string,
  limit: number = 100,
  excludeEdgeIds: string[] = [],
  direction: "all" | "incoming" | "outgoing" = "all",
  filters?: GraphFilters
): Promise<GraphPayload> {
  return json(
    await apiFetch("/api/graph/expand", {
      method: "POST",
      headers: withHeaders("", { "Content-Type": "application/json" }),
      body: JSON.stringify({
        node_id: nodeId,
        limit,
        exclude_edge_ids: excludeEdgeIds,
        direction,
        filters,
      }),
    }),
    "Не удалось раскрыть соседей"
  );
}

export async function fetchGraphFacets(
  q: string,
  filters: GraphFilters,
  sourceQuery = "",
  sourceCursor = "",
  sourceLimit = 50
): Promise<GraphFacets> {
  return json(
    await apiFetch("/api/graph/facets", {
      method: "POST",
      headers: withHeaders("", { "Content-Type": "application/json" }),
      body: JSON.stringify({
        q,
        field: "all",
        filters,
        source_query: sourceQuery,
        source_cursor: sourceCursor,
        source_limit: sourceLimit,
      }),
    }),
    "Не удалось загрузить фильтры базы"
  );
}

export async function fetchGraphSchema(): Promise<{ nodeLabels: string[]; relationshipTypes: string[]; runId: string }> {
  return json(await apiFetch("/api/graph/schema", { headers: withHeaders("") }), "Не удалось загрузить схему базы");
}

export async function resolveApproval(
  approval: PendingApproval,
  action: "approve" | "revise" | "cancel",
  selection: { openSqRefs?: string[]; newSubquestions?: string[] },
  feedback = "",
  signal?: AbortSignal,
): Promise<Response> {
  const res = await apiFetch(`/api/tool-approvals/${encodeURIComponent(approval.id)}/resolve`, {
    method: "POST",
    headers: withHeaders("", { "Content-Type": "application/json", Accept: "text/event-stream" }),
    body: JSON.stringify({
      action,
      revision: approval.revision,
      open_sq_refs: selection.openSqRefs || [],
      new_subquestions: selection.newSubquestions || [],
      feedback,
    }),
    signal,
  });
  if (!res.ok) await json(res, "Не удалось обработать план поиска");
  return res;
}

export async function fetchCardTemplates(): Promise<CardTemplate[]> {
  return json(await apiFetch("/api/card-templates"), "Не удалось загрузить шаблоны");
}

export async function fetchCards(): Promise<SavedCard[]> {
  return json(await apiFetch("/api/cards"), "Не удалось загрузить карточки");
}

export async function createCardTemplate(input: {
  name: string;
  description?: string;
  schema: Record<string, unknown>;
  ui?: Record<string, unknown>;
  instructions?: string;
}): Promise<CardTemplate> {
  return json(
    await apiFetch("/api/card-templates", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
    "Не удалось создать шаблон"
  );
}

export async function createCardTemplateVersion(
  templateId: string,
  input: {
    name: string;
    description?: string;
    schema: Record<string, unknown>;
    ui?: Record<string, unknown>;
    instructions?: string;
  }
): Promise<{ id: string; templateId: string; version: number }> {
  return json(
    await apiFetch(`/api/card-templates/${encodeURIComponent(templateId)}/versions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(input),
    }),
    "Не удалось создать новую версию шаблона"
  );
}

export async function archiveCardTemplate(templateId: string): Promise<void> {
  await json(
    await apiFetch(`/api/card-templates/${encodeURIComponent(templateId)}`, { method: "DELETE" }),
    "Не удалось удалить шаблон"
  );
}

export async function generateCardDraft(
  checkpointId: string,
  templateVersionId: string,
  profile?: string,
  reasoningEffort?: string
): Promise<{ draft: CardDraft; message: ChatMessage; checkpointId: string }> {
  const payload = await json<{ type: string; draft: CardDraft; message: ChatMessage; checkpoint_id: string }>(
    await apiFetch("/api/card-drafts/generate", {
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
    await apiFetch(`/api/card-drafts/${encodeURIComponent(draftId)}/save`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    }),
    "Не удалось сохранить карточку"
  );
}

export async function updateCardDraft(
  draftId: string,
  input: {
    data: Record<string, unknown>;
    provenance: Record<string, unknown>;
    gaps?: unknown[];
  }
): Promise<CardDraft> {
  return json(
    await apiFetch(`/api/card-drafts/${encodeURIComponent(draftId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...input, gaps: input.gaps || [] }),
    }),
    "Не удалось обновить черновик"
  );
}

export async function importCardDraft(
  templateVersionId: string,
  data: Record<string, unknown> | Record<string, unknown>[],
  checkpointId?: string
): Promise<CardDraft[]> {
  const payload = await json<CardDraft | { items: CardDraft[] }>(
    await apiFetch("/api/cards/import", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        checkpoint_id: checkpointId || null,
        template_version_id: templateVersionId,
        data,
      }),
    }),
    "Импорт не прошёл validation"
  );
  return "items" in payload ? payload.items : [payload];
}

export async function archiveCard(cardId: string): Promise<void> {
  await json(
    await apiFetch(`/api/cards/${encodeURIComponent(cardId)}`, { method: "DELETE" }),
    "Не удалось удалить карточку"
  );
}

export async function reviseCard(
  cardId: string,
  input: { title: string; data: Record<string, unknown>; editedFields: string[] }
): Promise<SavedCard> {
  return json(
    await apiFetch(`/api/cards/${encodeURIComponent(cardId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: input.title,
        data: input.data,
        edited_fields: input.editedFields,
      }),
    }),
    "Не удалось сохранить новую правку карточки"
  );
}

export async function attachCard(
  branchId: string,
  baseCheckpointId: string,
  cardRevisionId: string,
  attached = true
): Promise<{ checkpointId: string }> {
  return json(
    await apiFetch(`/api/branches/${encodeURIComponent(branchId)}/card-attachments`, {
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

export async function insertCardMessage(
  branchId: string,
  baseCheckpointId: string,
  cardRevisionId: string
): Promise<{ checkpointId: string; message: ChatMessage }> {
  return json(
    await apiFetch(`/api/branches/${encodeURIComponent(branchId)}/card-messages`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        base_checkpoint_id: baseCheckpointId,
        card_revision_id: cardRevisionId,
      }),
    }),
    "Не удалось вставить карточку в диалог"
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
    fork_if_needed?: boolean;
    intent?: "chat" | "generate_card";
    template_version_id?: string;
  }
): string {
  const payload: Record<string, string | boolean> = {
    text,
    search_depth: opts.search_depth,
  };
  if (opts.profile) payload.profile = opts.profile;
  if (opts.profile !== "auto" && opts.reasoning_effort) payload.reasoning_effort = opts.reasoning_effort;
  if (opts.turn_id) payload.turn_id = opts.turn_id;
  if (opts.mode) payload.mode = opts.mode;
  if (opts.branch_id) payload.branch_id = opts.branch_id;
  if (opts.base_checkpoint_id) payload.base_checkpoint_id = opts.base_checkpoint_id;
  if (opts.fork_if_needed) payload.fork_if_needed = true;
  if (opts.intent) payload.intent = opts.intent;
  if (opts.template_version_id) payload.template_version_id = opts.template_version_id;
  return JSON.stringify(payload);
}

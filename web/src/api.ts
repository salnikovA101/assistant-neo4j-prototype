import type { Account, ConversationDetail, ConversationSummary, GraphPayload, SearchDepth, UiConfig } from "./types";

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
  if (!res.ok) throw new Error(payload.error || payload.detail || fallback);
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

export async function fetchConversation(id: string): Promise<ConversationDetail> {
  return json(await fetch(`/api/conversations/${encodeURIComponent(id)}`), "Не удалось открыть чат");
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
  field: "all" | "name" | "rel" | "evidence" = "all"
): Promise<GraphPayload> {
  const res = await fetch("/graph_explore", {
    method: "POST",
    headers: withHeaders("", { "Content-Type": "application/json" }),
    body: JSON.stringify({ q, limit, field }),
  });
  const payload = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(payload.error || "Не удалось загрузить граф");
  return payload as GraphPayload;
}

export function streamBody(
  text: string,
  opts: { search_depth: SearchDepth; reasoning_effort?: string; profile?: string; turn_id?: string }
): string {
  const payload: Record<string, string> = {
    text,
    search_depth: opts.search_depth,
  };
  if (opts.profile) payload.profile = opts.profile;
  if (opts.reasoning_effort) payload.reasoning_effort = opts.reasoning_effort;
  if (opts.turn_id) payload.turn_id = opts.turn_id;
  return JSON.stringify(payload);
}

import type { DocumentBatch, DocumentBatchItem } from "./types";

export const DOCUMENT_STATUS_LABEL: Record<string, string> = {
  unavailable: "Ожидает сервис",
  queued: "В очереди",
  processing: "Обрабатывается",
  completed: "Готово",
  failed: "Ошибка",
  cancelled: "Отменено",
};

export const ACTIVE_DOCUMENT_STATUSES = new Set(["unavailable", "queued", "processing"]);
export const TERMINAL_DOCUMENT_STATUSES = new Set(["completed", "failed", "cancelled"]);

export type DocumentIndicator = "idle" | "progress" | "ready" | "error";

export function documentStatusLabel(status: string): string {
  return DOCUMENT_STATUS_LABEL[status] || status;
}

export function formatFileSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} Б`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1).replace(".", ",")} КБ`;
  return `${(bytes / (1024 * 1024)).toFixed(1).replace(".", ",")} МБ`;
}

export function formatEta(seconds: number | null | undefined): string {
  if (seconds == null || seconds < 0) return "";
  if (seconds < 60) return `~${seconds} с`;
  if (seconds < 3600) return `~${Math.max(1, Math.round(seconds / 60))} мин`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds % 3600) / 60);
  return minutes ? `~${hours} ч ${minutes} мин` : `~${hours} ч`;
}

export function isPdfFile(file: File): boolean {
  const type = (file.type || "").split(";", 1)[0].trim().toLowerCase();
  return file.name.toLowerCase().endsWith(".pdf") && (
    type === "" || type === "application/pdf" || type === "application/x-pdf" || type === "application/octet-stream"
  );
}

export function flattenDocumentItems(batches: DocumentBatch[]): DocumentBatchItem[] {
  return batches.flatMap((batch) => batch.items);
}

export function isActiveDocumentStatus(status: string): boolean {
  return ACTIVE_DOCUMENT_STATUSES.has(status);
}

function ruFiles(count: number): string {
  const abs = Math.abs(count) % 100;
  const digit = abs % 10;
  if (abs > 10 && abs < 20) return `${count} файлов`;
  if (digit === 1) return `${count} файл`;
  if (digit >= 2 && digit <= 4) return `${count} файла`;
  return `${count} файлов`;
}

function ruErrors(count: number): string {
  const abs = Math.abs(count) % 100;
  const digit = abs % 10;
  if (abs > 10 && abs < 20) return `${count} ошибок`;
  if (digit === 1) return `${count} ошибка`;
  if (digit >= 2 && digit <= 4) return `${count} ошибки`;
  return `${count} ошибок`;
}

export function overallDocumentEtaSeconds(batches: DocumentBatch[]): number | null {
  const values: number[] = [];
  for (const batch of batches) {
    const hasActive = batch.items.some((item) => isActiveDocumentStatus(item.status));
    if (!hasActive) continue;
    if (batch.etaSeconds != null && batch.etaSeconds >= 0) values.push(batch.etaSeconds);
    for (const item of batch.items) {
      if (isActiveDocumentStatus(item.status) && item.etaSeconds != null && item.etaSeconds >= 0) {
        values.push(item.etaSeconds);
      }
    }
  }
  return values.length ? Math.max(...values) : null;
}

function ruReadyWord(count: number): string {
  const abs = Math.abs(count) % 100;
  const digit = abs % 10;
  if (abs > 10 && abs < 20) return "готовы";
  if (digit === 1) return "готов";
  return "готовы";
}

export function documentQueueSummary(batches: DocumentBatch[]): string {
  const items = flattenDocumentItems(batches);
  if (!items.length) return "";
  const completed = items.filter((item) => item.status === "completed").length;
  const failed = items.filter((item) => item.status === "failed").length;
  const active = items.filter((item) => isActiveDocumentStatus(item.status));
  if (active.length) {
    if (active.every((item) => item.status === "unavailable") && !completed && !failed) {
      return `${ruFiles(items.length)} ждут сервис обработки`;
    }
    const parts = [`${completed} из ${items.length} ${ruReadyWord(completed)}`];
    const eta = formatEta(overallDocumentEtaSeconds(batches));
    if (eta) parts.push(`осталось ${eta}`);
    return parts.join(" · ");
  }
  if (failed && completed) return `${completed} ${ruReadyWord(completed)}, ${ruErrors(failed)}`;
  if (failed) return `Не удалось обработать ${ruFiles(failed)}`;
  if (completed) return `${ruFiles(completed)} обработаны`;
  return "";
}

export function documentIndicator(
  batches: DocumentBatch[],
  seenIds: ReadonlySet<string>,
): DocumentIndicator {
  const items = flattenDocumentItems(batches);
  if (items.some((item) => isActiveDocumentStatus(item.status))) return "progress";
  const unseenFailed = items.some((item) => item.status === "failed" && !seenIds.has(item.id));
  if (unseenFailed) return "error";
  const unseenReady = items.some((item) => item.status === "completed" && !seenIds.has(item.id));
  if (unseenReady) return "ready";
  return "idle";
}

export function documentIndicatorLabel(kind: DocumentIndicator): string {
  if (kind === "progress") return "Документы: идёт обработка";
  if (kind === "ready") return "Документы: обработка завершена";
  if (kind === "error") return "Документы: ошибка обработки";
  return "Открыть документы";
}

export function terminalDocumentIds(batches: DocumentBatch[]): string[] {
  return flattenDocumentItems(batches)
    .filter((item) => TERMINAL_DOCUMENT_STATUSES.has(item.status))
    .map((item) => item.id);
}

export function seenDocumentStorageKey(accountId: string, workspace: string): string {
  return `neo4j-assistant.document-seen:${accountId}:${workspace}`;
}

export function loadSeenDocumentIds(accountId: string, workspace: string): Set<string> {
  if (!accountId) return new Set();
  try {
    const raw = localStorage.getItem(seenDocumentStorageKey(accountId, workspace));
    const parsed = raw ? JSON.parse(raw) : [];
    return new Set(Array.isArray(parsed) ? parsed.filter((item) => typeof item === "string") : []);
  } catch {
    return new Set();
  }
}

export function saveSeenDocumentIds(accountId: string, workspace: string, ids: Iterable<string>): Set<string> {
  const seen = new Set(ids);
  if (!accountId) return seen;
  try {
    localStorage.setItem(seenDocumentStorageKey(accountId, workspace), JSON.stringify([...seen]));
  } catch {
    /* Private mode or quota should not break the queue UI. */
  }
  return seen;
}

export function markTerminalDocumentsSeen(
  batches: DocumentBatch[],
  accountId: string,
  workspace: string,
): Set<string> {
  const seen = loadSeenDocumentIds(accountId, workspace);
  for (const id of terminalDocumentIds(batches)) seen.add(id);
  return saveSeenDocumentIds(accountId, workspace, seen);
}

export function documentCompletionNotice(
  previous: DocumentBatch[],
  next: DocumentBatch[],
): string {
  const wasActive = flattenDocumentItems(previous).some((item) => isActiveDocumentStatus(item.status));
  const nextItems = flattenDocumentItems(next);
  const stillActive = nextItems.some((item) => isActiveDocumentStatus(item.status));
  if (!wasActive || stillActive) return "";
  const completed = nextItems.filter((item) => item.status === "completed").length;
  const failed = nextItems.filter((item) => item.status === "failed").length;
  if (failed && completed) return `Документы обработаны: ${completed} ${ruReadyWord(completed)}, ${ruErrors(failed)}.`;
  if (failed) return "Не удалось обработать документы.";
  if (completed) return "Документы обработаны. Откройте вкладку Документы.";
  return "";
}

export function noticeOpensDocuments(text: string): boolean {
  return /документ/i.test(text);
}

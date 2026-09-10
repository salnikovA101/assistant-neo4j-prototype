const KEY = "neo4j-assistant.sessions.v1";

export function clearLegacySessions(): void {
  localStorage.removeItem(KEY);
}

export function titleFromText(text: string): string {
  const line = text.trim().split("\n")[0] || "Новый чат";
  return line.length > 42 ? `${line.slice(0, 41)}…` : line;
}

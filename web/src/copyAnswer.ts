/** Copyable Markdown as on screen: [n] in the answer, then ### Источники with [n] - file.

Do not invent http(s) URLs. The corpus files are not public links.
*/

const SOURCES_HEADING = /^###[ \t]*Источники(?:\s|$)/im;
const SOURCE_ENTRY = /^\[(\d+)\]\s+(.+?)\s*$/;
const GAPS_HEADING = /^###[ \t]*GAPS\s*$/gim;

export function answerMarkdownForCopy(text: string): string {
  const raw = text || "";
  const { body, biblio } = splitSourcesSection(raw);
  const visibleBody = body.replace(GAPS_HEADING, "### Пробелы в данных").replace(/\s+$/, "");
  const names = sourceNamesFromBibliography(biblio);
  if (!names.size) return visibleBody ? `${visibleBody}\n` : "";
  const listed = [...names.entries()].map(([id, name]) => `[${id}] - ${name}`).join("\n");
  return `${visibleBody}\n\n### Источники\n${listed}\n`;
}

function splitSourcesSection(markdown: string): { body: string; biblio: string } {
  let last = -1;
  const heading = new RegExp(SOURCES_HEADING.source, "gim");
  let match: RegExpExecArray | null;
  while ((match = heading.exec(markdown)) !== null) last = match.index;
  if (last < 0) return { body: markdown, biblio: "" };
  const lineEnd = markdown.indexOf("\n", last);
  const after = lineEnd === -1 ? markdown.length : lineEnd + 1;
  return { body: markdown.slice(0, last), biblio: markdown.slice(after) };
}

function sourceNamesFromBibliography(biblio: string): Map<string, string> {
  const names = new Map<string, string>();
  for (const line of biblio.split(/\r?\n/)) {
    const match = line.trim().match(SOURCE_ENTRY);
    if (!match) continue;
    const name = match[2].replace(/^[-–—]\s*/, "").trim();
    if (!name || names.has(match[1])) continue;
    names.set(match[1], name);
  }
  return names;
}

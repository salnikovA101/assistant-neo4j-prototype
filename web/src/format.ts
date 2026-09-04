import DOMPurify from "dompurify";
import katex from "katex";
import { marked } from "marked";
import "katex/dist/katex.min.css";

marked.setOptions({ gfm: true, breaks: true });

function escapeHtml(value: string): string {
  return value.replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char] || char);
}

function renderLatexExpression(source: string): string {
  return katex.renderToString(source.trim(), {
    displayMode: false,
    throwOnError: false,
    strict: "ignore",
    trust: false,
    output: "htmlAndMathml",
  });
}

function findMathDelimiter(text: string, from: number, delimiter: "$" | "$$"): number {
  let index = text.indexOf(delimiter, from);
  while (index !== -1) {
    let backslashes = 0;
    for (let cursor = index - 1; cursor >= 0 && text[cursor] === "\\"; cursor -= 1) backslashes += 1;
    if (backslashes % 2 === 0) return index;
    index = text.indexOf(delimiter, index + delimiter.length);
  }
  return -1;
}

function renderLatex(text: string): string {
  let out = "";
  let cursor = 0;
  while (cursor < text.length) {
    if (text.startsWith("```", cursor)) {
      const closing = text.indexOf("```", cursor + 3);
      const end = closing === -1 ? text.length : closing + 3;
      out += text.slice(cursor, end);
      cursor = end;
      continue;
    }
    if (text[cursor] === "`") {
      const closing = text.indexOf("`", cursor + 1);
      const end = closing === -1 ? text.length : closing + 1;
      out += text.slice(cursor, end);
      cursor = end;
      continue;
    }
    const display = text.startsWith("$$", cursor);
    if (display || text[cursor] === "$") {
      const delimiter = display ? "$$" : "$";
      const end = findMathDelimiter(text, cursor + delimiter.length, delimiter);
      const source = end === -1 ? "" : text.slice(cursor + delimiter.length, end);
      if (source.trim()) {
        const rendered = renderLatexExpression(source);
        out += display
          ? `\n<div class="latex-display" role="math" aria-label="${escapeHtml(source)}">${rendered}</div>\n`
          : `<span class="latex-inline" role="math" aria-label="${escapeHtml(source)}">${rendered}</span>`;
        cursor = end + delimiter.length;
        continue;
      }
    }
    out += text[cursor];
    cursor += 1;
  }
  return out;
}

function densifyCitations(text: string): string {
  const map = new Map<number, number>();
  const display = (sid: number) => {
    if (!map.has(sid)) map.set(sid, map.size + 1);
    return map.get(sid);
  };
  const markersFor = (chunk: string) => {
    const markers: string[] = [];
    const seen = new Set<string>();
    const idRe = /source\s*:?\s*(\d+)/gi;
    let match: RegExpExecArray | null;
    while ((match = idRe.exec(chunk)) !== null) {
      const sid = parseInt(match[1], 10);
      if (!Number.isFinite(sid)) continue;
      const marker = `[${display(sid)}]`;
      if (seen.has(marker)) continue;
      seen.add(marker);
      markers.push(marker);
    }
    return markers.join("");
  };
  let rendered = text.replace(
    /\(\s*source\s*:?\s*\d+[^)]*\)/gi,
    (group) => markersFor(group)
  );
  rendered = rendered.replace(/(?<!\w)source\s*:?\s*(\d+)(?!\w)/gi, (_, n) => {
    const sid = parseInt(n, 10);
    return Number.isFinite(sid) ? `[${display(sid)}]` : _;
  });
  return rendered;
}

function holdIncompleteFence(text: string): string {
  let count = 0;
  let last = -1;
  let i = 0;
  while ((i = text.indexOf("```", i)) !== -1) {
    count += 1;
    last = i;
    i += 3;
  }
  if (count % 2 === 1 && last >= 0) return text.slice(0, last);
  return text;
}

function holdIncompleteCitation(text: string): string {
  let last = -1;
  const start = /\(\s*source\b/gi;
  let match: RegExpExecArray | null;
  while ((match = start.exec(text)) !== null) last = match.index;
  if (last < 0 || text.indexOf(")", last) !== -1) return text;
  return text.slice(0, last);
}

export function renderMarkdown(text: string, streaming = false): string {
  const prepared = streaming ? holdIncompleteCitation(holdIncompleteFence(text)) : text;
  const src = renderLatex(densifyCitations(prepared));
  const raw = marked.parse(src) as string;
  return DOMPurify.sanitize(raw);
}

export type AnswerMarkdownParts = {
  bodyHtml: string;
  sourcesHtml: string;
  sourceCount: number;
};

function improveAnswerHtml(html: string): AnswerMarkdownParts {
  const parsed = new DOMParser().parseFromString(html, "text/html");
  const headings = Array.from(parsed.body.querySelectorAll<HTMLHeadingElement>("h1, h2, h3, h4, h5, h6"));

  for (const heading of headings) {
    if (heading.textContent?.trim().toUpperCase() === "GAPS") {
      heading.textContent = "Пробелы в данных";
    }
  }

  const sourcesHeading = headings.find((heading) => /^Источники(?:\s|$)/i.test(heading.textContent?.trim() || ""));
  if (!sourcesHeading) return { bodyHtml: parsed.body.innerHTML, sourcesHtml: "", sourceCount: 0 };

  const sourceNodes: ChildNode[] = [];
  const sourcesLevel = Number(sourcesHeading.tagName.slice(1));
  let next = sourcesHeading.nextSibling;
  while (next) {
    if (
      next instanceof HTMLHeadingElement
      && Number(next.tagName.slice(1)) <= sourcesLevel
    ) {
      break;
    }
    const current = next;
    next = next.nextSibling;
    sourceNodes.push(current);
  }
  if (!sourceNodes.some((node) => node.textContent?.trim())) {
    return { bodyHtml: parsed.body.innerHTML, sourcesHtml: "", sourceCount: 0 };
  }

  const listItemCount = sourceNodes.reduce((count, node) => {
    if (!(node instanceof Element)) return count;
    return count + node.querySelectorAll("li").length;
  }, 0);
  const numberedSourceCount = sourceNodes.reduce(
    (count, node) => count + (node.textContent?.match(/\[\d+\]/g)?.length || 0),
    0,
  );
  const itemCount = listItemCount || numberedSourceCount;
  const sources = parsed.createElement("div");
  sources.append(...sourceNodes);
  sourcesHeading.remove();
  return {
    bodyHtml: parsed.body.innerHTML,
    sourcesHtml: sources.innerHTML,
    sourceCount: itemCount,
  };
}

export function renderAnswerMarkdown(text: string, streaming = false): AnswerMarkdownParts {
  const rendered = renderMarkdown(text, streaming);
  if (streaming) return { bodyHtml: rendered, sourcesHtml: "", sourceCount: 0 };
  const improved = improveAnswerHtml(rendered);
  return {
    bodyHtml: DOMPurify.sanitize(improved.bodyHtml),
    sourcesHtml: DOMPurify.sanitize(improved.sourcesHtml),
    sourceCount: improved.sourceCount,
  };
}

function unwrapMarkdownLikeFences(text: string): string {
  return text.replace(/```(?:markdown|md)?\s*\n([\s\S]*?)\n```/gi, (whole, body: string) => {
    const lines = body.split("\n");
    const isMarkdown = lines.some((line) =>
      /^\s*(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s+)/.test(line)
    );
    return isMarkdown ? body : whole;
  });
}

function decodeMarkedCode(text: string): string {
  return text
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

function promoteMarkdownCodeBlocks(html: string): string {
  return html.replace(/<pre><code(?: [^>]*)?>([\s\S]*?)<\/code><\/pre>/g, (whole, encodedBody: string) => {
    const body = decodeMarkedCode(encodedBody);
    const isMarkdown = body.split("\n").some((line) =>
      /^\s*(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s+)/.test(line)
    );
    return isMarkdown ? (marked.parse(body) as string) : whole;
  });
}

export function renderReasoningMarkdown(text: string, streaming = false): string {
  return DOMPurify.sanitize(
    promoteMarkdownCodeBlocks(renderMarkdown(unwrapMarkdownLikeFences(text), streaming))
  );
}

export function parseSseBlock(raw: string): { event: string; data: Record<string, unknown> } | null {
  let event = "message";
  const dataLines: string[] = [];
  for (const line of raw.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) dataLines.push(line.slice(5).trim());
  }
  if (!dataLines.length) return null;
  try {
    return { event, data: JSON.parse(dataLines.join("\n")) as Record<string, unknown> };
  } catch {
    return null;
  }
}

export const DEPTH_LABELS: Record<string, string> = {
  low: "Компактно",
  medium: "Обычно",
  high: "Расширенно",
};

export const EFFORT_LABELS: Record<string, { label: string; hint: string }> = {
  off: { label: "Выкл", hint: "без рассуждения" },
  none: { label: "Выкл", hint: "без рассуждения" },
  on: { label: "Вкл", hint: "с рассуждением" },
  minimal: { label: "Минимум", hint: "короче цепочка" },
  low: { label: "Коротко", hint: "быстрее" },
  medium: { label: "Обычно", hint: "баланс" },
  high: { label: "Глубоко", hint: "длиннее цепочка" },
  xhigh: { label: "Максимум", hint: "самая длинная цепочка" },
  max: { label: "Предел", hint: "максимальная цепочка" },
};

export function effortLabel(value: string): string {
  return EFFORT_LABELS[value]?.label || value;
}

export function prettyJson(value: unknown): string {
  if (value == null || value === "") return "";
  if (typeof value === "string") {
    try {
      return JSON.stringify(JSON.parse(value), null, 2);
    } catch {
      return value;
    }
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
}

export function tripletCaption(edge: {
  label: string;
  from_name?: string;
  to_name?: string;
  from: string;
  to: string;
}): string {
  return `${edge.from_name || edge.from} —${edge.label}→ ${edge.to_name || edge.to}`;
}

const MINUTE_MS = 60_000;
const HOUR_MS = 60 * MINUTE_MS;
const DAY_MS = 24 * HOUR_MS;

export function formatRelativeTime(timestamp: number, now = Date.now()): string {
  const delta = Math.max(0, now - timestamp);
  if (delta < MINUTE_MS) return "сейчас";
  if (delta < HOUR_MS) return `${Math.floor(delta / MINUTE_MS)} мин`;
  if (delta < DAY_MS) return `${Math.floor(delta / HOUR_MS)} ч`;
  return `${Math.floor(delta / DAY_MS)} д`;
}

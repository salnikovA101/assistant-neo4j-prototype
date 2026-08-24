import DOMPurify from "dompurify";
import { marked } from "marked";

marked.setOptions({ gfm: true, breaks: true });

function densifyCitations(text: string): string {
  const map = new Map<number, number>();
  const display = (sid: number) => {
    if (!map.has(sid)) map.set(sid, map.size + 1);
    return map.get(sid);
  };
  let rendered = text.replace(
    /\(\s*source\s*:\s*(\d+(?:\s*,\s*\d+)*)\s*\)/gi,
    (_, ids: string) =>
      ids
        .split(",")
        .map((part) => {
          const n = parseInt(part.trim(), 10);
          return Number.isFinite(n) ? `[${display(n)}]` : "";
        })
        .join("")
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

export function renderMarkdown(text: string, streaming = false): string {
  const src = densifyCitations(streaming ? holdIncompleteFence(text) : text)
    .replace(/\$\\rightarrow\$/g, "→")
    .replace(/\$\\to\$/g, "→")
    .replace(/\$\\times\$/g, "×");
  const raw = marked.parse(src) as string;
  return DOMPurify.sanitize(raw);
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
    return { event, data: { delta: dataLines.join("\n") } };
  }
}

export const DEPTH_LABELS: Record<string, string> = {
  low: "Узко",
  medium: "Обычно",
  high: "Широко",
};

export const EFFORT_LABELS: Record<string, { label: string; hint: string }> = {
  off: { label: "Выкл", hint: "без рассуждения" },
  none: { label: "Выкл", hint: "без рассуждения" },
  on: { label: "Вкл", hint: "с рассуждением" },
  low: { label: "Коротко", hint: "быстрее" },
  medium: { label: "Обычно", hint: "баланс" },
  high: { label: "Глубоко", hint: "длиннее цепочка" },
  xhigh: { label: "Максимум", hint: "самая длинная цепочка" },
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


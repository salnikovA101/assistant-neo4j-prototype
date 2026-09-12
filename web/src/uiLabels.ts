import type { GraphEdge } from "./types";

export const MODE_LABELS = {
  staged: "Исследование",
  auto: "Вопрос по базе",
} as const;

export function normalizedLabels(labels?: string[]): string[] {
  return [...new Set((labels || []).map((value) => value.trim()).filter(Boolean))]
    .sort((left, right) => left.localeCompare(right, undefined, { sensitivity: "base" }) || left.localeCompare(right));
}

export function visibleNodeRef(labels: string[] | undefined, name: string): string {
  void labels;
  const cleanName = name.trim();
  return cleanName || "?";
}

export function visibleTripletCaption(edge: Pick<GraphEdge, "label" | "from_name" | "to_name" | "from_labels" | "to_labels" | "from" | "to">): string {
  const from = visibleNodeRef(edge.from_labels, edge.from_name || edge.from);
  const to = visibleNodeRef(edge.to_labels, edge.to_name || edge.to);
  return `${from} → ${edge.label || "—"} → ${to}`;
}

export function chainLabel(label: string, unitNo?: number | null): string {
  if (unitNo != null) return `Цепочка ${unitNo}`;
  const match = label.match(/(?:UNIT|Цепочка)\s*(\d+)/i);
  return match ? `Цепочка ${match[1]}` : label;
}

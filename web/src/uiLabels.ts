import type { GraphEdge } from "./types";

export const MODE_LABELS = {
  staged: "Исследование",
  auto: "Быстрый ответ",
} as const;

export function visibleTripletCaption(edge: Pick<GraphEdge, "label" | "from_name" | "to_name" | "from" | "to">): string {
  return `${edge.from_name || edge.from} → ${edge.label || "—"} → ${edge.to_name || edge.to}`;
}

export function chainLabel(label: string, unitNo?: number | null): string {
  if (unitNo != null) return `Цепочка ${unitNo}`;
  const match = label.match(/(?:UNIT|Цепочка)\s*(\d+)/i);
  return match ? `Цепочка ${match[1]}` : label;
}

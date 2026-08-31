import type { Branch } from "../types";
import { IconChevron, IconFork } from "./Icons";

function visibleName(branch: Branch, index: number): string {
  if (index === 0 && branch.name.trim().toLowerCase() === "main") return "Основной вариант";
  return branch.name;
}

export function BranchMenu({
  branches,
  activeId,
  onOpen,
  open,
  openDirections,
}: {
  branches: Branch[];
  activeId: string;
  onOpen: () => void;
  open: boolean;
  openDirections?: number;
}) {
  const activeIndex = Math.max(0, branches.findIndex((branch) => branch.id === activeId));
  const active = branches[activeIndex];
  const label = active ? visibleName(active, activeIndex) : "Основной вариант";

  return (
    <div className="branch-menu-wrap">
      <button type="button" className="branch-menu-trigger research-entry" onClick={onOpen} aria-expanded={open} aria-label={`Ход работы, текущий вариант: ${label}`}>
        <IconFork /><span>Ход работы · {label}</span>
        {Boolean(openDirections) && <b aria-label={`Открытых пунктов плана: ${openDirections}`}>{openDirections}</b>}
        <IconChevron />
      </button>
    </div>
  );
}

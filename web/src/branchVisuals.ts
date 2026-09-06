export const BRANCH_COLORS = ["#6ea8ff", "#8bd5ca", "#c6a0f6", "#f5bde6", "#eed49f", "#91d7e3"];

export function branchColor(index: number): string {
  return BRANCH_COLORS[Math.max(0, index) % BRANCH_COLORS.length];
}
